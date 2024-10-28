import torch
import torch.optim as optim
import numpy as np
import argparse

from evaluate import eval, write_pot_masks, eval_psnr_roi
from datetime import datetime
from unwrap_utils import get_tuples, pre_train_mapping, load_input_data, save_video
import sys

from config_utils import config_load, config_save

from pathlib import Path

from networks import build_network
from losses import *
from unwrap_utils import Timer

from PIL import Image
import cv2

import csv
import time
import skimage.metrics

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config):
   
    cfg = config_load(config)
    data_folder = Path(cfg["data_folder"])
    resx = np.int64(cfg["resx"])
    resy = np.int64(cfg["resy"])
    reference_frame = cfg["reference_frame"]

    iters_num = cfg["iters_num"]
    refine_iters_num = cfg["refine_iters_num"]
    loss_cfg = cfg["losses"]
    samples = cfg["samples_batch"]
    load_checkpoint = cfg["load_checkpoint"] # set to true to continue from a checkpoint
    checkpoint_path = cfg["checkpoint_path"]
    results_folder_name = cfg["results_folder_name"] # the folder (under the code's folder where the experiments will be saved.
    folder_suffix = cfg["folder_suffix"] # for each experiment folder (saved inside "results_folder_name") add this string
    pretrain_iter_number = cfg["pretrain_iter_number"]
    uv_mapping_scales = cfg["uv_mapping_scales"]
    mappings_cfg = cfg["model_mapping"]
    alpha_cfg = cfg["alpha"]
    eval_cfg = cfg["evaluation"]
    use_residual = cfg["use_residual"]
    refine_th = cfg["refine_th"]
    
    num_of_maps = len(mappings_cfg)
    timer = Timer()
    vid_name = data_folder.name
    
    results_folder = Path(f'./{results_folder_name}/{vid_name}_{folder_suffix}')
    results_folder.mkdir(parents=True, exist_ok=True)
    config_save(cfg, '%s/config.py'%results_folder)

    optical_flows_mask, video_frames, optical_flows_reverse_mask, mask_frames, video_frames_dx, video_frames_dy,  optical_flows_reverse, optical_flows, ref_opt_flows, ref_opt_flows_mask, scale_factor = load_input_data(
        resy, resx, data_folder, True,  True)
    cfg["scale_factor"] =scale_factor

    number_of_frames=video_frames.shape[3]

    # save the video in the working resolution
    save_video(video_frames, results_folder, cfg = cfg)

    model_F_mappings = torch.nn.ModuleList()
    for mapping_cfg in mappings_cfg:
        model_F_mappings.append(
            build_network(device=device, **mapping_cfg))

    model_alpha = build_network(device=device, **alpha_cfg)
  
    optimizer_all = list()
    optimizer_all.extend(model_alpha.get_optimizer_list())
    for model_F_mapping in model_F_mappings:
        optimizer_all.extend(model_F_mapping.get_optimizer_list())
    optimizer_all = optim.Adam(optimizer_all)

    rez = np.maximum(resx, resy)

    # get losses
    loss_funcs = dict()
    if loss_cfg.get('rgb'):
        loss_funcs['rgb'] = RGBLoss(loss_cfg['rgb']['weight'])
    
    if loss_cfg.get('alpha_bootstrapping'):
        loss_funcs['alpha_bootstrapping'] = AlphaBootstrappingLoss(loss_cfg['alpha_bootstrapping']['weight'])
    
    if loss_cfg.get('flow_alpha'):
        loss_funcs['flow_alpha'] = FlowAlphaLoss(rez, number_of_frames, loss_cfg['flow_alpha']['weight'])
    if loss_cfg.get("ref_flow_alpha"):
        loss_funcs['ref_flow_alpha'] = RefFlowAlphaLoss(rez, number_of_frames, loss_cfg['ref_flow_alpha']['weight'], reference_frame)
    if loss_cfg.get('ref_mask'):
        loss_funcs['ref_mask'] = RefMaskLoss(loss_cfg['ref_mask']['weight'])

    if loss_cfg.get('optical_flow'):
        loss_funcs['optical_flow'] = FlowMappingLoss(rez, number_of_frames, loss_cfg['optical_flow']['weight'])
    if loss_cfg.get('ref_optical_flow'):
        loss_funcs['ref_optical_flow'] = RefFlowMappingLoss(rez, number_of_frames, loss_cfg['ref_optical_flow']['weight'], reference_frame)

    if loss_cfg.get('rigidity'):
        loss_funcs['rigidity'] = RigidityLoss( rez, number_of_frames, loss_cfg['rigidity']['derivative_amount'], loss_cfg['rigidity']['weight'])
    if loss_cfg.get('position'):
        loss_funcs['position'] = PositionLoss(loss_cfg['position']['weight'])

    if use_residual :
        if loss_cfg.get('residual_reg'):
            loss_funcs['residual_reg'] = ResidualRegLoss(loss_cfg['residual_reg']['weight'])

    if loss_cfg.get('refine'):
        loss_funcs['refine'] = RefineLoss(rez, number_of_frames,loss_cfg['refine']['weight'], refine_th, use_residual)

    jif_all = get_tuples(number_of_frames, video_frames)   

    
    for i, model_F_mapping in enumerate(model_F_mappings):
        if model_F_mapping.pretrain:
            pre_train_mapping(
                model_F_mapping, number_of_frames, uv_mapping_scales[i],
                resx=resx, resy=resy, rez=rez,
                device=device, pretrain_iters=pretrain_iter_number)
    
    refine_step = False
    start_iteration = 1
    max_psnr_roi = 0
    
    # Start training!
    print("start training")
    for iteration in range(start_iteration, iters_num + refine_iters_num + 1):
        
        #Mask Refinement Step
        if iteration == iters_num +1 :
            print("start mask refinemet from %d" %iteration)
            refine_step = True
            checkpoint_path = "./results/" + str(Path(cfg["data_folder"]).name + "_" + cfg["folder_suffix"]) + "/checkpoint"
            # load checkpoint with max psnr roi
            init_file = torch.load(checkpoint_path)
            model_F_mappings.load_state_dict(init_file["model_F_mappings_state_dict"])
            model_alpha.load_state_dict(init_file["model_F_alpha_state_dict"])
            optimizer_all.load_state_dict(init_file["optimizer_all_state_dict"])

            # freeze mapping netwroks
            for model_F_mapping in model_F_mappings:
                for name, param in model_F_mapping.named_parameters():
                    param.requires_grad = False
            
            #load potential mask
            write_pot_masks(model_F_mappings, model_alpha, jif_all, video_frames, number_of_frames, rez, 
                                eval_cfg['samples_batch'], cfg, results_folder, device)
            init_mask_dir = str(results_folder) + "/pot_mask/"
            init_mask_files = sorted(Path(init_mask_dir).glob('*.png'))
            init_masks = torch.zeros((resy, resx,number_of_frames))
            for t in range(len(init_mask_files)):
                init_mask = np.array(Image.open(init_mask_files[t]).convert('L'))
                init_masks[:,:,t] = torch.tensor(init_mask)
            loss_funcs['flow_alpha'].set_loss_weight(loss_cfg['refine_flow_alpha']['weight'])

        losses = dict()
        inds_foreground = torch.randint(
            jif_all.shape[1], (np.int64(samples * 1.0), 1))
        jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)
        xyt_current = torch.cat((
            jif_current[0] / (rez / 2) - 1,
            jif_current[1] / (rez / 2) - 1,
            jif_current[2] / (number_of_frames / 2) - 1
        ), dim=1).to(device)  # size (batch, 3)

        if use_residual :
            uvs, residuals, rgb_textures = zip(*[i(xyt_current, return_residual = True, return_rgb = True) for i in model_F_mappings])
        else : 
            uvs, _, rgb_textures = zip(*[i(xyt_current, return_residual = True, return_rgb = True) for i in model_F_mappings])
        
        alpha = model_alpha(xyt_current)
        rgb_ori = video_frames[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]].squeeze(1).to(device)

        # reconstruct final colors
        if use_residual :
            rgb_output = (residuals[0] * rgb_textures[0]) * alpha[:, [0]] + (rgb_ori) * ( 1 - alpha[:, [0]])
        else : 
            rgb_output = (rgb_textures[0]) * alpha[:, [0]] + (rgb_ori) * (1-alpha[:, [0]])

        ref_loc = torch.where(jif_current[2,:,:].squeeze() == reference_frame)[0]
        ref_alpha_GT = mask_frames[jif_current[1], jif_current[0], jif_current[2]].squeeze(1)[ref_loc].to(device)
        ref_alpha = alpha[ref_loc]   

        # alpha bootstrapping loss
        if loss_funcs.get('alpha_bootstrapping'):
            if iteration <= loss_cfg['alpha_bootstrapping']['stop_iteration']:
                alpha_GT = mask_frames[jif_current[1], jif_current[0], jif_current[2]].squeeze(1).to(device)
                losses['alpha_bootstrapping'] = loss_funcs['alpha_bootstrapping'](alpha_GT, alpha)     
        
        # RGB loss
        if loss_funcs.get('rgb'):
            rgb_GT = video_frames[jif_current[1], jif_current[0], :, jif_current[2]].squeeze(1).to(device)
            losses['rgb'] = loss_funcs['rgb'](rgb_GT, rgb_output)

        if loss_funcs.get('flow_alpha'):
            losses['flow_alpha'] = loss_funcs['flow_alpha'](
                optical_flows, optical_flows_mask,
                optical_flows_reverse, optical_flows_reverse_mask,
                jif_current, alpha, device, model_alpha)  

        if refine_step != True :
            if loss_funcs.get('ref_flow_alpha'):
                losses['ref_flow_alpha'] = loss_funcs['ref_flow_alpha'](
                    ref_opt_flows, ref_opt_flows_mask,
                    jif_current, alpha, device, model_alpha)
            if loss_funcs.get('ref_mask'):
                losses['ref_mask'] = loss_funcs['ref_mask'](ref_alpha_GT, ref_alpha)
            
            if loss_funcs.get('ref_optical_flow'):
                losses['ref_optical_flow'] = loss_funcs['ref_optical_flow'](
                    ref_opt_flows, ref_opt_flows_mask,
                    jif_current, uvs[0], uv_mapping_scales[0],
                    device, model_F_mappings[0], True, alpha[:, [0]])
            if loss_funcs.get('optical_flow'):
                for i in range(num_of_maps):
                    losses['optical_flow_%d'%i] = loss_funcs['optical_flow'](
                        optical_flows, optical_flows_mask,
                        optical_flows_reverse, optical_flows_reverse_mask,
                        jif_current, uvs[0], uv_mapping_scales[0],
                        device, model_F_mappings[0], True, alpha[:, [0]])
            
            if loss_funcs.get('rigidity'):
                for i in range(num_of_maps):
                    losses['rigidity_%d'%i] = loss_funcs['rigidity'](
                        jif_current,
                        uvs[i], uv_mapping_scales[i],
                        device, model_F_mappings[i])
            if loss_funcs.get('position'):
                losses['position'] = loss_funcs['position'](ref_alpha_GT, xyt_current[ref_loc], uvs[0][ref_loc])

            if use_residual :
                if loss_funcs.get('residual_reg'):
                    losses['residual_reg'] = loss_funcs['residual_reg'](residuals[0])

        else :
            if loss_funcs.get('refine'):
                losses['refine'] = loss_funcs['refine'](model_F_mappings, video_frames, xyt_current, jif_current, init_masks, model_alpha, device)
            
        loss = sum(losses.values())

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()

        if iteration % eval_cfg['interval'] == 0:
            psnr, psnr_roi = eval(
                model_F_mappings, model_alpha,
                jif_all, video_frames, number_of_frames, rez, eval_cfg['samples_batch'],
                num_of_maps, str(results_folder/('%06d'%iteration)),
                iteration, optimizer_all, device, use_residual = use_residual, cfg = cfg, refine_th = refine_th
            )
            print('Iter = %d, PSNR = %.6f, PSNR ROI = %.6f' %(iteration, psnr, psnr_roi))
        
        if refine_step != True and iteration % eval_cfg['save_ckpt_interval'] == 0:
            psnr_roi = eval_psnr_roi(model_F_mappings, model_alpha, jif_all, video_frames,number_of_frames,rez, device, 
                                    use_residual, cfg)
            if psnr_roi >= max_psnr_roi :
                max_psnr_roi = psnr_roi
                saved_dict = {
                    'iteration': iteration,
                    'model_F_mappings_state_dict': model_F_mappings.state_dict(),
                    'model_F_alpha_state_dict': model_alpha.state_dict(),
                    'optimizer_all_state_dict': optimizer_all.state_dict()
                }
                torch.save(saved_dict, '%s/checkpoint' % (results_folder))
                with open(os.path.join(str(results_folder), 'MAX_PSNR_ROI.txt'), 'w') as f:
                    f.write('%d\n'%iteration)
                    f.write('MAX_PSNR_ROI = %.2f\n'%max_psnr_roi)

        model_F_mappings.train()
        model_alpha.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    args = parser.parse_args()

    main(args.config)
