from pathlib import Path
import argparse
import numpy as np
from raft_wrapper import RAFTWrapper

from tqdm import tqdm
import cv2
from torchvision.utils import save_image
from PIL import Image
import sys
import json
import torch
from config_utils import config_load, config_save
import sys
from raft import RAFT
from utils.utils import InputPadder
import os
import torch.nn.functional as F
import imageio.v2 as imageio
from time import time

DEVICE = 'cuda'

def gen_grid(h, w, device, normalize=False, homogeneous=False):
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h, device=device)
        lin_x = torch.linspace(-1., 1., steps=w, device=device)
    else:
        lin_y = torch.arange(0, h, device=device)
        lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x), indexing="ij")
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]

def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
    else:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

def load_image(fn, new_w, new_h):
    img = np.array(Image.open(fn)).astype(np.uint8)

    im_h = img.shape[0]
    im_w = img.shape[1]
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)\


def run_extract_reference_flow(config):
    
    start_time = time()
    vid_path = Path(config["data_folder"])
    vid_name = vid_path.name
    vid_root = vid_path.parent
    
    input_dir = vid_root / vid_name / 'video_frames'

    resx = config["resx"]
    resy = config["resy"]
    max_long_edge = max(resx, resy)
    model_path = 'thirdparty/RAFT/models/raft-things.pth'
    
    reference_frame = config["reference_frame"]
    use_dino = True

    args = argparse.Namespace()
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    args.model = model_path

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model.to(DEVICE)
    model.eval()

    accum_flow_out_dir = vid_root / vid_name / 'ref_flow'
    accum_flow_mask_out_dir = vid_root / vid_name / 'ref_flow_mask'
    os.makedirs(accum_flow_out_dir, exist_ok=True)
    os.makedirs(accum_flow_mask_out_dir, exist_ok=True)

    img_files = sorted(input_dir.glob('*.*'))
    num_imgs = len(img_files)
    
    ref_imgname = os.path.basename(img_files[reference_frame])
    save_file = os.path.join(accum_flow_out_dir,'{}_{}.npy'.format(ref_imgname, ref_imgname))
    ref_frame_flow = np.zeros((resy,resx,2))
    np.save(save_file, ref_frame_flow)
    ref_frame_flow_mask = np.ones((resy,resx))
    imageio.imwrite('{}/{}_{}.png'.format(accum_flow_mask_out_dir, ref_imgname, ref_imgname), (255*ref_frame_flow_mask).astype(np.uint8))
            
    cycle_th = 5.0
    intervals = [1,2,3,4,5,6,7,8,9]
    max_interval = max(intervals)
    mask_th = 255
    ref_dino_th = 0.6
    
    print("reference_frame : ", reference_frame)
    print("intervals : ", intervals, ", max_interval : ", max_interval)
    
    grid = gen_grid(resy, resx, device=DEVICE).permute(2, 0, 1)[None]

    if use_dino == True :   
        grid_normed = normalize_coords(grid.squeeze().permute(1, 2, 0), resy, resx)
        features = [torch.from_numpy(np.load(os.path.join(vid_root / vid_name / 'dino',
                                                          os.path.basename(img_file) + '.npy'))).float().to(DEVICE)
                    for img_file in img_files]
        dino_th = 0.5
    
    flow_with_ref = True

    with torch.no_grad():
        for i in range(reference_frame+1, num_imgs):
            cur_imfile = img_files[i]
            cur_image = load_image(cur_imfile, resx, resy)
            padder = InputPadder(cur_image.shape)

            accum_flow_tensor = torch.zeros((1, 2, resy,resx),dtype=torch.float32).cuda()
            accum_flow_mask_tensor = torch.zeros((1, resy,resx),dtype=torch.float32).cuda()
            if flow_with_ref == True:
                if abs(i - reference_frame) > max_interval :
                    ref_image = load_image(ref_imfile, resx, resy)
                    cur_image_pad, ref_image_pad = padder.pad(cur_image, ref_image)

                    _, flow_12 = model(cur_image_pad, ref_image_pad, iters=20, test_mode=True)
                    _, flow_21 = model(ref_image_pad, cur_image_pad, iters=20, test_mode=True)
            
                    flow_12_np = flow_12.squeeze().permute(1, 2, 0).cpu().numpy()
                    flow_21_np = flow_21.squeeze().permute(1, 2, 0).cpu().numpy()

                    flow_12_tensor = torch.from_numpy(flow_12_np).float().permute(2, 0, 1)[None].cuda()
                    flow_21_tensor = torch.from_numpy(flow_21_np).float().permute(2, 0, 1)[None].cuda()

                    coord2 = flow_12_tensor + grid
                    coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
                    flow_21_sampled = F.grid_sample(flow_21_tensor, coord2_normed[None],  mode='bilinear', align_corners=True)
                    map_i = flow_12_tensor + flow_21_sampled
                    fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
                    flow12_mask = fb_discrepancy < cycle_th
            
                    if use_dino == True :
                        feature_current = features[i].permute(2, 0, 1)[None]
                        feature_current_sampled = F.grid_sample(feature_current, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                        feature_current_sampled = feature_current_sampled.permute(1, 2, 0)
                
                        feature_reference = features[reference_frame].permute(2, 0, 1)[None]
                        feature_reference_sampled = F.grid_sample(feature_reference, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                        feature_sim = torch.cosine_similarity(feature_current_sampled, feature_reference_sampled, dim=-1)
                        feature_mask = feature_sim > ref_dino_th
                        flow12_mask = flow12_mask * feature_mask

                    accum_flow_tensor = flow_12_tensor.clone()
                    accum_flow_mask_tensor = 255 * flow12_mask.clone().float()[None].cuda()

            flow12_low_prev = None
            for k in intervals:
                mid = i - k
                if mid < reference_frame : 
                    break
                mid_imfile = img_files[mid]
                ref_imfile = img_files[reference_frame] 
                mid_image = load_image(mid_imfile, resx, resy)
                cur_image_pad, mid_image_pad = padder.pad(cur_image, mid_image)
                flow12_low, flow_12 = model(cur_image_pad, mid_image_pad, iters=20, test_mode=True, flow_init=flow12_low_prev)
                _, flow_21 = model(mid_image_pad, cur_image_pad, iters=20, test_mode=True, flow_init=None)
            
                flow12_low_prev = flow12_low


                flow_12_np = flow_12.squeeze().permute(1, 2, 0).cpu().numpy()
                flow_21_np = flow_21.squeeze().permute(1, 2, 0).cpu().numpy()

                flow_12_tensor = torch.from_numpy(flow_12_np).float().permute(2, 0, 1)[None].cuda()
                flow_21_tensor = torch.from_numpy(flow_21_np).float().permute(2, 0, 1)[None].cuda()

                coord2 = flow_12_tensor + grid
                coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
                flow_21_sampled = F.grid_sample(flow_21_tensor, coord2_normed[None],  mode='bilinear', align_corners=True)
                map_i = flow_12_tensor + flow_21_sampled
                fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
                flow12_mask = fb_discrepancy < cycle_th
            
                if use_dino == True :
                    feature_cur = features[i].permute(2, 0, 1)[None]
                    feature_cur_sampled = F.grid_sample(feature_cur, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                    feature_cur_sampled = feature_cur_sampled.permute(1, 2, 0)
                
                    feature_mid = features[mid].permute(2, 0, 1)[None]
                    feature_mid_sampled = F.grid_sample(feature_mid, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                    feature_sim = torch.cosine_similarity(feature_cur_sampled, feature_mid_sampled, dim=-1)
                    feature_mask = feature_sim > dino_th
                    flow12_mask = flow12_mask * feature_mask
                cur_mid_flow_tensor = flow_12_tensor.clone()
                cur_mid_flow_mask_tensor = 255 * flow12_mask.clone().float()[None].cuda() # [1, 432,768], float
                
                if mid == reference_frame :
                    chain_flow_tensor = cur_mid_flow_tensor
                    chain_flow_mask_tensor = cur_mid_flow_mask_tensor
                else : 
                    mid_ref_flow_file = os.path.join(vid_path, 'ref_flow', '{}_{}.npy'.format(os.path.basename(mid_imfile), os.path.basename(ref_imfile)))
                    mid_ref_flow_np = np.load(mid_ref_flow_file)
                    mid_ref_flow_tensor = torch.from_numpy(mid_ref_flow_np).float()[None].permute(0, 3, 1, 2).cuda()
                
                    mid_ref_flow_mask_file = mid_ref_flow_file.replace('ref_flow', 'ref_flow_mask').replace('.npy', '.png') 
                    mid_ref_flow_mask_tensor = torch.from_numpy(imageio.imread(mid_ref_flow_mask_file)).float()[None].cuda() 

                    coords = grid + cur_mid_flow_tensor
                    coords_normed = normalize_coords(coords.squeeze().permute(1, 2, 0), resy, resx) 
                            
                    mid_ref_flow_warped_tensor = F.grid_sample(mid_ref_flow_tensor, coords_normed[None], mode='bilinear',  align_corners=True)
                    chain_flow_tensor =  cur_mid_flow_tensor + mid_ref_flow_warped_tensor
                    
                    mid_ref_flow_mask_warped_tensor = F.grid_sample(mid_ref_flow_mask_tensor[None], coords_normed[None], mode='bilinear',  align_corners=True)[0]
                    chain_flow_mask_tensor =  ((mid_ref_flow_mask_warped_tensor >= mask_th) & (cur_mid_flow_mask_tensor >= 255)).float()*255.0
                
                mask_conf_tensor = (accum_flow_mask_tensor + chain_flow_mask_tensor).float().squeeze()/255.0
                avg = torch.zeros((1, 2, resy,resx),dtype=torch.float32).cuda()
                avg[:,:,  (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()==255)] = (accum_flow_tensor + chain_flow_tensor)[:,:,  (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()==255)]/2.0
                avg[:,:,  (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()!=255)] = accum_flow_tensor[:,:, (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()!=255)]
                avg[:,:,  (accum_flow_mask_tensor.squeeze()!=255)&(chain_flow_mask_tensor.squeeze()==255)] = chain_flow_tensor[:,:, (accum_flow_mask_tensor.squeeze()!=255)&(chain_flow_mask_tensor.squeeze()==255)]
                accum_flow_tensor[:,:, mask_conf_tensor!=0] = avg[:,:, mask_conf_tensor!=0]
                accum_flow_mask_tensor = ((accum_flow_mask_tensor + chain_flow_mask_tensor)>0).float()*255.0

            if use_dino == True :
                feature_cur = features[i].permute(2, 0, 1)[None]
                feature_cur_sampled = F.grid_sample(feature_cur, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                feature_cur_sampled = feature_cur_sampled.permute(1, 2, 0)

                coord2 = accum_flow_tensor + grid
                coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
                
                feature_ref = features[reference_frame].permute(2, 0, 1)[None]
                feature_ref_sampled = F.grid_sample(feature_ref, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                feature_sim = torch.cosine_similarity(feature_cur_sampled, feature_ref_sampled, dim=-1)
                feature_mask = feature_sim > dino_th
                accum_flow_mask_tensor = (accum_flow_mask_tensor>0) * feature_mask
                accum_flow_mask_tensor = accum_flow_mask_tensor.float()*255.0


            save_file = os.path.join(accum_flow_out_dir,'{}_{}.npy'.format(os.path.basename(cur_imfile), os.path.basename(ref_imfile)))
            np.save(save_file, accum_flow_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
            imageio.imwrite('{}/{}_{}.png'.format(accum_flow_mask_out_dir,os.path.basename(cur_imfile), os.path.basename(ref_imfile)), (accum_flow_mask_tensor.squeeze().cpu().numpy().astype(np.uint8))) 

        for i in range(reference_frame-1, -1, -1):
            cur_imfile = img_files[i]
            cur_image = load_image(cur_imfile, resx, resy)
            padder = InputPadder(cur_image.shape)

            accum_flow_tensor = torch.zeros((1, 2, resy,resx),dtype=torch.float32).cuda()
            accum_flow_mask_tensor = torch.zeros((1, resy,resx),dtype=torch.float32).cuda()
            if flow_with_ref == True:
                if abs(i - reference_frame) > max_interval :
                    ref_image = load_image(ref_imfile, resx, resy)
                    cur_image_pad, ref_image_pad = padder.pad(cur_image, ref_image)

                    _, flow_12 = model(cur_image_pad, ref_image_pad, iters=20, test_mode=True)
                    _, flow_21 = model(ref_image_pad, cur_image_pad, iters=20, test_mode=True)
            
                    flow_12_np = flow_12.squeeze().permute(1, 2, 0).cpu().numpy()
                    flow_21_np = flow_21.squeeze().permute(1, 2, 0).cpu().numpy()

                    flow_12_tensor = torch.from_numpy(flow_12_np).float().permute(2, 0, 1)[None].cuda()
                    flow_21_tensor = torch.from_numpy(flow_21_np).float().permute(2, 0, 1)[None].cuda()

                    coord2 = flow_12_tensor + grid
                    coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
                    flow_21_sampled = F.grid_sample(flow_21_tensor, coord2_normed[None],  mode='bilinear', align_corners=True)
                    map_i = flow_12_tensor + flow_21_sampled
                    fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
                    flow12_mask = fb_discrepancy < cycle_th
            
                    if use_dino == True :
                        feature_current = features[i].permute(2, 0, 1)[None]
                        feature_current_sampled = F.grid_sample(feature_current, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                        feature_current_sampled = feature_current_sampled.permute(1, 2, 0)
                
                        feature_reference = features[reference_frame].permute(2, 0, 1)[None]
                        feature_reference_sampled = F.grid_sample(feature_reference, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                        feature_sim = torch.cosine_similarity(feature_current_sampled, feature_reference_sampled, dim=-1)
                        feature_mask = feature_sim > ref_dino_th
                        flow12_mask = flow12_mask * feature_mask

                    accum_flow_tensor = flow_12_tensor.clone()
                    accum_flow_mask_tensor = 255 * flow12_mask.clone().float()[None].cuda()
            
            flow12_low_prev = None
            for k in intervals:
                mid = i + k
                if mid > reference_frame : 
                    break
                mid_imfile = img_files[mid]
                ref_imfile = img_files[reference_frame] 
                mid_image = load_image(mid_imfile, resx, resy)
                cur_image_pad, mid_image_pad = padder.pad(cur_image, mid_image)
                flow_12_low, flow_12 = model(cur_image_pad, mid_image_pad, iters=20, test_mode=True, flow_init=flow12_low_prev)
                _, flow_21 = model(mid_image_pad, cur_image_pad, iters=20, test_mode=True)

                flow12_low_prev = flow_12_low

                flow_12_np = flow_12.squeeze().permute(1, 2, 0).cpu().numpy()
                flow_21_np = flow_21.squeeze().permute(1, 2, 0).cpu().numpy()

                flow_12_tensor = torch.from_numpy(flow_12_np).float().permute(2, 0, 1)[None].cuda()
                flow_21_tensor = torch.from_numpy(flow_21_np).float().permute(2, 0, 1)[None].cuda()

                coord2 = flow_12_tensor + grid
                coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
                flow_21_sampled = F.grid_sample(flow_21_tensor, coord2_normed[None],  mode='bilinear', align_corners=True)
                map_i = flow_12_tensor + flow_21_sampled
                fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
                flow12_mask = fb_discrepancy < cycle_th
            
                if use_dino == True :
                    feature_cur = features[i].permute(2, 0, 1)[None]
                    feature_cur_sampled = F.grid_sample(feature_cur, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                    feature_cur_sampled = feature_cur_sampled.permute(1, 2, 0)
                
                    feature_mid = features[mid].permute(2, 0, 1)[None]
                    feature_mid_sampled = F.grid_sample(feature_mid, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                    feature_sim = torch.cosine_similarity(feature_cur_sampled, feature_mid_sampled, dim=-1)
                    feature_mask = feature_sim > dino_th
                    flow12_mask = flow12_mask * feature_mask
                
                cur_mid_flow_tensor = flow_12_tensor.clone()
                cur_mid_flow_mask_tensor = 255 * flow12_mask.clone().float()[None].cuda() # [1, 432,768], float
                
                if mid == reference_frame :
                    chain_flow_tensor = cur_mid_flow_tensor
                    chain_flow_mask_tensor = cur_mid_flow_mask_tensor
                else : 
                    mid_ref_flow_file = os.path.join(vid_path, 'ref_flow', '{}_{}.npy'.format(os.path.basename(mid_imfile), os.path.basename(ref_imfile)))
                    mid_ref_flow_np = np.load(mid_ref_flow_file)
                    mid_ref_flow_tensor = torch.from_numpy(mid_ref_flow_np).float()[None].permute(0, 3, 1, 2).cuda()
                
                    mid_ref_flow_mask_file = mid_ref_flow_file.replace('ref_flow', 'ref_flow_mask').replace('.npy', '.png') 
                    mid_ref_flow_mask_tensor = torch.from_numpy(imageio.imread(mid_ref_flow_mask_file)).float()[None].cuda() 

                    coords = grid + cur_mid_flow_tensor
                    coords_normed = normalize_coords(coords.squeeze().permute(1, 2, 0), resy, resx) 
                            
                    mid_ref_flow_warped_tensor = F.grid_sample(mid_ref_flow_tensor, coords_normed[None], mode='bilinear',  align_corners=True)
                    chain_flow_tensor =  cur_mid_flow_tensor + mid_ref_flow_warped_tensor
                    
                    mid_ref_flow_mask_warped_tensor = F.grid_sample(mid_ref_flow_mask_tensor[None], coords_normed[None], mode='bilinear',  align_corners=True)[0]
                    chain_flow_mask_tensor =  ((mid_ref_flow_mask_warped_tensor >= mask_th) & (cur_mid_flow_mask_tensor >= 255)).float()*255.0
                
                mask_conf_tensor = (accum_flow_mask_tensor + chain_flow_mask_tensor).float().squeeze()/255.0
                avg = torch.zeros((1, 2, resy,resx),dtype=torch.float32).cuda()
                avg[:,:,  (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()==255)] = (accum_flow_tensor + chain_flow_tensor)[:,:,  (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()==255)]/2.0
                avg[:,:,  (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()!=255)] = accum_flow_tensor[:,:, (accum_flow_mask_tensor.squeeze()==255)&(chain_flow_mask_tensor.squeeze()!=255)]
                avg[:,:,  (accum_flow_mask_tensor.squeeze()!=255)&(chain_flow_mask_tensor.squeeze()==255)] = chain_flow_tensor[:,:, (accum_flow_mask_tensor.squeeze()!=255)&(chain_flow_mask_tensor.squeeze()==255)]
                accum_flow_tensor[:,:, mask_conf_tensor!=0] = avg[:,:, mask_conf_tensor!=0]
                accum_flow_mask_tensor = ((accum_flow_mask_tensor + chain_flow_mask_tensor)>0).float()*255.0

            
            if use_dino == True :
                coord2 = accum_flow_tensor + grid
                coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
                
                feature_cur = features[i].permute(2, 0, 1)[None]
                feature_cur_sampled = F.grid_sample(feature_cur, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                feature_cur_sampled = feature_cur_sampled.permute(1, 2, 0)
                
                feature_ref = features[reference_frame].permute(2, 0, 1)[None]
                feature_ref_sampled = F.grid_sample(feature_ref, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                feature_sim = torch.cosine_similarity(feature_cur_sampled, feature_ref_sampled, dim=-1)
                feature_mask = feature_sim > dino_th
                accum_flow_mask_tensor = (accum_flow_mask_tensor>0) * feature_mask
                accum_flow_mask_tensor = accum_flow_mask_tensor.float()*255.0


            save_file = os.path.join(accum_flow_out_dir,'{}_{}.npy'.format(os.path.basename(cur_imfile), os.path.basename(ref_imfile)))
            np.save(save_file, accum_flow_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
            imageio.imwrite('{}/{}_{}.png'.format(accum_flow_mask_out_dir,os.path.basename(cur_imfile), os.path.basename(ref_imfile)), (accum_flow_mask_tensor.squeeze().cpu().numpy().astype(np.uint8))) 

    print("extract reference flow time : " , time() - start_time)

    start_time = time()
    
    flow_dir = 'ref_flow'
    flow_mask_dir = 'ref_flow_mask'
    save_mask_dir = vid_root / vid_name / 'init_masks'
    
    os.makedirs(save_mask_dir, exist_ok=True)

    roi = config["roi"]
    image = imageio.imread(img_files[0])
    ori_h = image.shape[0]
    ori_w = image.shape[1]

    scale_factor = max(ori_w, ori_h)/max_long_edge
    lt = [int(roi[0]//scale_factor),int(roi[1]//scale_factor)]
    rt = [int(roi[2]//scale_factor),int(roi[3]//scale_factor)]
    rb = [int(roi[4]//scale_factor),int(roi[5]//scale_factor)]
    lb = [int(roi[6]//scale_factor),int(roi[7]//scale_factor)]
    roi = [lt, rt, rb, lb]
    print("scale factor : ", scale_factor, " roi : ", roi)

    ref_mask = np.zeros((resy,resx),dtype=np.float32)
    ref_mask = cv2.fillPoly(ref_mask,[np.array([roi[0],roi[1],roi[2],roi[3]])],255.0)
    imageio.imwrite(str(save_mask_dir) + "/" + str(reference_frame).zfill(5) + ".png",ref_mask.astype(np.uint8))

    for query_id in range(0, num_imgs):
        if query_id == reference_frame :
            continue
        target_id = reference_frame
        imgname_query = os.path.basename(img_files[query_id])
        imgname_target = os.path.basename(img_files[target_id])
            
        flows = np.load(os.path.join(vid_path, flow_dir, '{}_{}.npy'.format(imgname_query, imgname_target)))
        raft_mask = imageio.imread(os.path.join(vid_path, flow_mask_dir, '{}_{}.png'.format(imgname_query, imgname_target)))
        raft_mask = raft_mask[:,:] > 0 

        flow = flows.copy()
        flow[:, :, 0] += np.arange(resx)
        flow[:, :, 1] += np.arange(resy)[:, np.newaxis]
        mask_remap = cv2.remap(ref_mask, flow, None, cv2.INTER_NEAREST)* raft_mask
        imageio.imwrite(str(save_mask_dir) + "/" + str(query_id).zfill(5) + ".png",mask_remap.astype(np.uint8))

    print("make init mask : " , time() - start_time)

if __name__ == '__main__':
    run_extract_reference_flow(config_load(sys.argv[1]))