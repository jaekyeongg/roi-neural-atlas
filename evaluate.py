import skimage.metrics
import torch

import skimage.measure
import os
from PIL import Image
import numpy as np
import imageio

from tqdm import tqdm
import cv2

# taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
# sample coordinates x,y from image im.
def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

def eval_psnr_roi(model_F_mappings,model_alpha, jif_all, video_frames,number_of_frames,rez, device, use_residual, cfg) :
    reference_frame = cfg["reference_frame"]
    with torch.no_grad():
        model_F_mappings.eval()
        model_alpha.eval()
        ref_loc = torch.where(jif_all[2,:,None].squeeze() == reference_frame)[0]
        rec_x = jif_all[0, :, None] / (rez / 2) - 1.0
        rec_y = jif_all[1, :, None] / (rez / 2) - 1.0
        rec_t = jif_all[2, :, None] / (number_of_frames / 2) - 1.0
        rec_xyt = torch.cat((rec_x, rec_y, rec_t), dim=1)     
        ref_xyt= rec_xyt[ref_loc,:].to(device)
        _, residuals, rgbs = zip(*[i(ref_xyt, True, True) for i in model_F_mappings])
        alpha = model_alpha(ref_xyt)
        rgb_ori = video_frames[:,:,:,reference_frame].reshape(rgbs[0].shape).to(device)
        if use_residual :
            output = ((residuals[0]*rgbs[0]) * alpha[:, [0]] + (rgb_ori) *(1- alpha[:, [0]])).reshape(video_frames.shape[:-1]).cpu().numpy()
        else :
            output = ((rgbs[0]) * alpha[:, [0]] + (rgb_ori) *(1- alpha[:, [0]])).reshape(video_frames.shape[:-1]).cpu().numpy()

        roi_ori = cfg["roi"]
        scale_factor = cfg["scale_factor"]
        lt = [int(roi_ori[0]//scale_factor),int(roi_ori[1]//scale_factor)]
        rt = [int(roi_ori[2]//scale_factor),int(roi_ori[3]//scale_factor)]
        rb = [int(roi_ori[4]//scale_factor),int(roi_ori[5]//scale_factor)]
        lb = [int(roi_ori[6]//scale_factor),int(roi_ori[7]//scale_factor)]
        roi = [lt, rt, rb, lb]
        x_s = min(roi[0][0], roi[3][0])
        y_s = min(roi[0][1], roi[1][1])
        x_e = max(roi[2][0], roi[1][0])
        y_e = max(roi[2][1], roi[3][1])
        psnr_roi  = skimage.metrics.peak_signal_noise_ratio(
                video_frames[y_s:y_e,x_s:x_e, :, reference_frame].numpy(),
                output[y_s:y_e,x_s:x_e, :], data_range=1)
    return psnr_roi

def write_pot_masks(model_F_mappings, model_alpha, jif_all, video_frames, number_of_frames, rez, samples_batch, cfg, save_dir, device):
    model_F_mappings.eval()
    model_alpha.eval()
    with torch.no_grad():
        rec_x = jif_all[0, :, None] / (rez / 2) - 1.0
        rec_y = jif_all[1, :, None] / (rez / 2) - 1.0
        rec_t = jif_all[2, :, None] / (number_of_frames / 2) - 1.0
        rec_xyt = torch.cat((rec_x, rec_y, rec_t), dim=1)
        batch_xyt = rec_xyt.split(samples_batch, dim=0)
        xyts = list()
        alphas = list()
        uvs = list()
        residuals = list()
        rgbs = list()
        for idx in range(len(batch_xyt)):
            now_xyt = batch_xyt[idx].to(device)
            rec_alpha = model_alpha(now_xyt)
            rec_maps, rec_residuals, rec_rgbs = zip(*[i(now_xyt, True, True) for i in model_F_mappings])
            alphas.append(rec_alpha.cpu().numpy())
            uvs.append(np.stack([i.cpu().numpy() for i in rec_maps]))
            residuals.append(np.stack([i.cpu().numpy() for i in rec_residuals]))
            rgbs.append(np.stack([i.cpu().numpy() for i in rec_rgbs]))
        uvs = np.split(np.concatenate(uvs, axis=1), number_of_frames, axis=1)
        residuals = np.split(np.concatenate(residuals, axis=1), number_of_frames, axis=1)
        rgbs = np.split(np.concatenate(rgbs, axis=1), number_of_frames, axis=1)
        alphas = np.split(np.concatenate(alphas), number_of_frames)

        reference_frame = cfg["reference_frame"]
        scale_factor = cfg["scale_factor"]
        roi_ori = cfg["roi"]
        lt = [int(roi_ori[0]//scale_factor),int(roi_ori[1]//scale_factor)]
        rt = [int(roi_ori[2]//scale_factor),int(roi_ori[3]//scale_factor)]
        rb = [int(roi_ori[4]//scale_factor),int(roi_ori[5]//scale_factor)]
        lb = [int(roi_ori[6]//scale_factor),int(roi_ori[7]//scale_factor)]
        roi = [lt, rt, rb, lb]
        ref_mask = np.zeros((cfg["resy"], cfg["resx"]),dtype=np.float32)
        ref_rec_alpha = cv2.fillPoly(ref_mask,[np.array([roi[0],roi[1],roi[2],roi[3]])],1.0)
        ref_rec_alpha = ref_rec_alpha.reshape(-1,1)
        ref_loc = torch.where(jif_all[2,:,None].squeeze() == reference_frame)[0]
        ref_xyt = rec_xyt[ref_loc,:].to(device)
        ref_rec_maps, _,  _ = zip(*[i(ref_xyt, return_residual = True, return_rgb = True) for i in model_F_mappings])
        rec_idxs = [np.clip(np.floor((i * 0.5 + 0.5).cpu().numpy() * 1000).astype(np.int64), 0, 999) for i in ref_rec_maps]
        _idx = np.stack((rec_idxs[0][:, 1], rec_idxs[0][:, 0]))
        texture_size = (1000, 1000)
        video_size = video_frames.shape[:2]
        rec_masks = np.zeros((1, *texture_size), dtype=np.uint8)
        for d in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            _idx_now = _idx + np.array(d)[:, None]
            _idx_now[0] = np.clip(_idx_now[0], 0, texture_size[1]-1)
            _idx_now[1] = np.clip(_idx_now[1], 0, texture_size[0]-1)
            mask_now = (ref_rec_alpha[..., 0] * 255).astype(np.uint8)
            mask_now = np.max((mask_now, rec_masks[0][_idx_now[0], _idx_now[1]]), axis=0)
            rec_masks[0][_idx_now[0], _idx_now[1]] = mask_now
               
        im =  rec_masks[0].copy()
        pot_mask_path = str(save_dir) + "/pot_mask/"
        os.makedirs(pot_mask_path, exist_ok=True)
        pot_mask_all = np.zeros((len(uvs), *video_size))
        for t, (uv, alpha) in enumerate(zip(uvs, alphas)):
            pot_mask = bilinear_interpolate_numpy(im, (uv[0][:, 0]*0.5+0.5)*im.shape[1], (uv[0][:, 1]*0.5+0.5)*im.shape[0])
            pot_mask_all[t] += (pot_mask).reshape(*video_size)
            pot_mask_all[t][pot_mask_all[t]>128]= 255
            pot_mask_all[t][pot_mask_all[t]<=128]= 0
            cv2.imwrite(pot_mask_path + str(t).zfill(5) + ".png" , pot_mask_all[t])
        model_F_mappings.train()
        model_alpha.train()


def eval_data_gen(
    model_F_mappings, model_alpha,
    jif_all, number_of_frames, rez, samples_batch,
    num_of_maps, texture_size, device, cfg = None):
    '''
    Given the whole model settings.
    Return:
        - rec_masks: Maximum alpha value sampled of UV map of each layer.
        - xyts: Normalized spatial and temporal location of each frame.
        - alphas: Alpha value of each layer of each frame.
            `[ndarray, ndarray, ...]`, len = number of frames
        - uvs: Color of UV map of each layer of each frame.
            `[[uv1, uv2, ...], [uv1, uv2, ...], ...]`, len = number of frames
        - residuals: Residual value of each layer of each frame, corresponding to video coordinate.
            `[[residual1, residual2, ...], [residual1, residual2, ...], ...]`, len = number of frames
        - rgbs: Color of each layer of each frame, corresponding to video coordinate.
            `[[rgb1, rgb2, ...], [rgb1, rgb2, ...], ...]`, len = number of frames
    '''
    model_F_mappings.eval()
    model_alpha.eval()
    with torch.no_grad():
        rec_x = jif_all[0, :, None] / (rez / 2) - 1.0
        rec_y = jif_all[1, :, None] / (rez / 2) - 1.0
        rec_t = jif_all[2, :, None] / (number_of_frames / 2) - 1.0
        rec_xyt = torch.cat((rec_x, rec_y, rec_t), dim=1)

        batch_xyt = rec_xyt.split(samples_batch, dim=0)

        # init results
        rec_masks = np.zeros((1, *texture_size), dtype=np.uint8)
        if cfg is not None :
            reference_frame = cfg["reference_frame"]
            ref_loc = torch.where(jif_all[2,:,None].squeeze() == reference_frame)[0]
            ref_xyt= rec_xyt[ref_loc,:].to(device)
            #ref_rec_alpha = model_alpha(ref_xyt).cpu().numpy()
 
            ref_mask = np.zeros((cfg["resy"], cfg["resx"]),dtype=np.float32)
            roi = cfg["roi"]
            scale_factor = cfg["scale_factor"]
            lt = [int(roi[0]//scale_factor),int(roi[1]//scale_factor)]
            rt = [int(roi[2]//scale_factor),int(roi[3]//scale_factor)]
            rb = [int(roi[4]//scale_factor),int(roi[5]//scale_factor)]
            lb = [int(roi[6]//scale_factor),int(roi[7]//scale_factor)]
            roi = [lt, rt, rb, lb]
            ref_rec_alpha = cv2.fillPoly(ref_mask,[np.array([roi[0],roi[1],roi[2],roi[3]])],1.0)
            ref_rec_alpha = ref_rec_alpha.reshape(-1,1)

            ref_rec_maps, _,  _ = zip(*[i(ref_xyt, return_residual = True, return_rgb = True) for i in model_F_mappings])
            rec_idxs = [np.clip(np.floor((i * 0.5 + 0.5).cpu().numpy() * 1000).astype(np.int64), 0, 999) for i in ref_rec_maps]
            _idx = np.stack((rec_idxs[0][:, 1], rec_idxs[0][:, 0]))
            for d in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                _idx_now = _idx + np.array(d)[:, None]
                _idx_now[0] = np.clip(_idx_now[0], 0, texture_size[1]-1)
                _idx_now[1] = np.clip(_idx_now[1], 0, texture_size[0]-1)
                mask_now = (ref_rec_alpha[..., 0] * 255).astype(np.uint8)
                mask_now = np.max((mask_now, rec_masks[0][_idx_now[0], _idx_now[1]]), axis=0)
                rec_masks[0][_idx_now[0], _idx_now[1]] = mask_now
      
        xyts = list()
        alphas = list()
        uvs = list()
        residuals = list()
        rgbs = list()

        # run eval: split by batch size
        pbar = tqdm(range(number_of_frames), 'Generating')
        progress = 0

        for idx in range(len(batch_xyt)):
            now_xyt = batch_xyt[idx].to(device)
            progress += len(now_xyt) * number_of_frames
            if pbar.n != int(progress / len(rec_xyt)):
                pbar.update(int(progress / len(rec_xyt)) - pbar.n)
            xyts.append(now_xyt.cpu().numpy())
            rec_alpha = model_alpha(now_xyt)
            alphas.append(rec_alpha.cpu().numpy())
            rec_maps, rec_residuals, rec_rgbs = zip(*[i(now_xyt, True, True) for i in model_F_mappings])
            uvs.append(np.stack([i.cpu().numpy() for i in rec_maps]))
            residuals.append(np.stack([i.cpu().numpy() for i in rec_residuals]))
            rgbs.append(np.stack([i.cpu().numpy() for i in rec_rgbs]))

        pbar.close()
    
    # re-split the data by frame number
    xyts = np.split(np.concatenate(xyts), number_of_frames)
    uvs = np.split(np.concatenate(uvs, axis=1), number_of_frames, axis=1)
    residuals = np.split(np.concatenate(residuals, axis=1), number_of_frames, axis=1)
    rgbs = np.split(np.concatenate(rgbs, axis=1), number_of_frames, axis=1)
    alphas = np.split(np.concatenate(alphas), number_of_frames)

    return rec_masks, xyts, alphas, uvs, residuals, rgbs

def eval(
    model_F_mappings, model_alpha,
    jif_all, video_frames, number_of_frames, rez, samples_batch,
    num_of_maps, save_dir,
    iteration, optimizer_all, device, use_residual = True, save_checkpoint=True, cfg = None, refine_th = 0.025):
    
    os.makedirs(save_dir, exist_ok=True)
    texture_size = (1000, 1000)
    with torch.no_grad():
        # init results
        texture_maps = list()
        checkerboard = np.array(Image.open('checkerboard.png'))[..., :3]*0.8
        checkerboard_texture = [checkerboard for _ in range(num_of_maps)]
        _m = np.ones(shape = (texture_size[0], texture_size[1],3))
        grey = _m[:,:] * np.array([255.0, 255.0, 255.0])*0.75
        grey_texture = [grey for _ in range(num_of_maps)]
        # generate necessary evaluation components
        rec_masks, xyts, alphas, uvs, residuals, rgbs = eval_data_gen(model_F_mappings, model_alpha, jif_all, number_of_frames, rez, samples_batch, num_of_maps, texture_size, device, cfg=cfg)
      
        # write results
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, texture_size[1]), torch.linspace(-1, 1, texture_size[0]), indexing='ij')
        grid_xy = torch.hstack((grid_y.reshape(-1, 1), grid_x.reshape(-1, 1))).to(device)
        
        for i in range(num_of_maps):
            texture_map = model_F_mappings[i].model_texture(grid_xy
                ).detach().cpu().numpy().reshape(*texture_size, 3)
            texture_map = (texture_map * 255).astype(np.uint8)
            texture_maps.append(texture_map)
            Image.fromarray(np.concatenate((texture_map, rec_masks[i][..., None]), axis=-1)).save(os.path.join(save_dir, 'tex%d.png'%i))

        _write_alpha(save_dir, alphas, video_frames.shape[:2], 1)
        if use_residual :
            _write_residual(save_dir, residuals, alphas, video_frames.shape[:2], 1)
        
        psnr, psnr_roi, atlas_diff_th_masks  = _write_video(save_dir, rgbs, residuals, alphas, video_frames.numpy(), num_of_maps, video_frames.shape[:2],
                                                            use_residual= use_residual, cfg = cfg, refine_th = refine_th)

        _write_edited(save_dir, texture_maps, grey_texture, 0.5, uvs, residuals, alphas, video_frames, video_frames.shape[:2], rec_masks, use_residual, "grey")
        _write_edited(save_dir, texture_maps, checkerboard_texture, 0.5, uvs, residuals, alphas, video_frames, video_frames.shape[:2], rec_masks, use_residual, "checkboard")

        with open(os.path.join(save_dir, 'PSNR.txt'), 'w') as f:
            f.write('PSNR = %.6f\n'%psnr)
            f.write('PSNR ROI = %.6f\n'%psnr_roi)

        # save current model
        if save_checkpoint:
            saved_dict = {
                'iteration': iteration,
                'model_F_mappings_state_dict': model_F_mappings.state_dict(),
                'model_F_alpha_state_dict': model_alpha.state_dict(),
                'optimizer_all_state_dict': optimizer_all.state_dict()
            }
            torch.save(saved_dict, '%s/checkpoint' % (save_dir))
    return psnr, psnr_roi

def _write_residual(save_dir, residuals, alphas, video_size, num_layers):
    writers = [imageio.get_writer(os.path.join(save_dir, 'residual%d.mp4'%i), fps=10) for i in range(num_layers)]
    for alpha, residual in zip(alphas, residuals):
        for i in range(num_layers):
            writers[i].append_data(np.clip(residual[i]*128, 0, 255).astype(np.uint8).reshape(*video_size, 3))
    for i in writers: i.close()


def _write_alpha(save_dir, alphas, video_size, num_layers):
    writers = [imageio.get_writer(os.path.join(save_dir, 'alpha%d.mp4'%i), fps=10) for i in range(num_layers)]
    t = 0 
    mask_path = str(save_dir) + "/mask/"
    os.makedirs(mask_path, exist_ok=True)
    for alpha in alphas:
        alpha = alpha.reshape(*video_size, num_layers)
        for i in range(num_layers):
            writers[i].append_data((alpha[..., [i]] * 255).astype(np.uint8))
            mask_frame = (alpha[..., [i]] * 255).astype(np.uint8)
            cv2.imwrite(mask_path + str(t).zfill(5) + ".png" , mask_frame)
        t = t+1
    for i in writers: i.close()


def _write_video(save_dir, rgbs, residuals, alphas, video_frames, num_layers, video_size, write_compare=False, use_residual = True, cfg = None,  refine_th = 0.025):
    writer = imageio.get_writer(os.path.join(save_dir, 'rec.mp4'), fps=10)
    writer_atlas = imageio.get_writer(os.path.join(save_dir, 'atlas_rec.mp4'), fps=10)
    writer_atlas_diff_th = imageio.get_writer(os.path.join(save_dir, 'atlas_diff_th.mp4'), fps=10)

    psnr = np.zeros((len(rgbs), 1))
    atlas_diff_th_all = np.zeros((len(rgbs), *video_size))

    for t, (rgb, residual, alpha) in enumerate(zip(rgbs, residuals, alphas)):
        rgb_ori = video_frames[:,:,:,t].reshape(rgb[0].shape)

        if use_residual :
            output = ((residual[0]*rgb[0]) * alpha[:, [0]] + (rgb_ori) *(1- alpha[:, [0]])).reshape(video_frames.shape[:-1])
            atlas_diff_th = np.mean(np.abs((residual[0]*rgb[0])- rgb_ori),axis=1).squeeze().reshape(video_frames.shape[:-2]) < refine_th
            writer_atlas.append_data((np.clip((residual[0]*rgb[0]).reshape(video_frames.shape[:-1]), 0, 1) * 255).astype(np.uint8))
        else :
            output = ((rgb[0]) * alpha[:, [0]] + (rgb_ori) *(1- alpha[:, [0]])).reshape(video_frames.shape[:-1])
            writer_atlas.append_data((np.clip(rgb[0].reshape(video_frames.shape[:-1]), 0, 1) * 255).astype(np.uint8)) 
            atlas_diff_th = np.mean(np.abs(rgb[0]- rgb_ori),axis=1).squeeze().reshape(video_frames.shape[:-2]) < refine_th

        atlas_diff_th_all[t] = atlas_diff_th
        writer.append_data((np.clip(output, 0, 1) * 255).astype(np.uint8))
        writer_atlas_diff_th.append_data((np.clip(atlas_diff_th, 0, 1) * 255).astype(np.uint8))
        
        psnr[t] = skimage.metrics.peak_signal_noise_ratio(
            video_frames[:, :, :, t],
            output,
            data_range=1)
        
        if cfg is not None and t == cfg["reference_frame"]:
            scale_factor = cfg["scale_factor"]
            x_s = min(int(cfg["roi"][0]//scale_factor), int(cfg["roi"][6]//scale_factor))
            y_s = min(int(cfg["roi"][1]//scale_factor), int(cfg["roi"][3]//scale_factor))
            x_e = max(int(cfg["roi"][4]//scale_factor), int(cfg["roi"][2]//scale_factor))
            y_e = max(int(cfg["roi"][5]//scale_factor), int(cfg["roi"][7]//scale_factor))
            psnr_roi  = skimage.metrics.peak_signal_noise_ratio(
                video_frames[y_s:y_e,x_s:x_e, :, t],
                output[y_s:y_e,x_s:x_e, :],
                data_range=1)
            refer_frame = cv2.cvtColor((np.clip(output, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            refer_frame_rect = cv2.rectangle(refer_frame,(x_s,y_s),(x_e, y_e),(0,255,0),2)
            cv2.imwrite(str(save_dir) + "/refer_frame_recon_rect.png" , refer_frame_rect)
 
    writer.close()
    writer_atlas.close()
    writer_atlas_diff_th.close()

    return psnr.mean(), psnr_roi.mean(), atlas_diff_th_all

def _write_edited(save_dir, maps1, maps2, ratio, uvs, residuals, alphas, video_frames, video_size, rec_masks, ues_residual, texture_name):

    writer_edited_all = imageio.get_writer(os.path.join(save_dir, 'edited_' + texture_name + '.mp4'), fps=10)
    edited_all = np.zeros((len(uvs), *video_size, 3))

    rec_mask = rec_masks[0].copy()
    for i in range(len(maps1)):
        im =  maps2[0].copy()
        for t, (uv, alpha) in enumerate(zip(uvs, alphas)):
            pot_mask = bilinear_interpolate_numpy(rec_mask, (uv[0][:, 0]*0.5+0.5)*rec_mask.shape[1], (uv[0][:, 1]*0.5+0.5)*rec_mask.shape[0])
            pot_mask[pot_mask>0] = 1.0
            pot_mask = np.expand_dims(pot_mask,axis=1)
            alpha = alpha*pot_mask
            rgb = bilinear_interpolate_numpy(im, (uv[i][:, 0]*0.5+0.5)*im.shape[1], (uv[i][:, 1]*0.5+0.5)*im.shape[0])
            rgb_ori = (video_frames[:,:,:,t].cpu().numpy()*255.0).reshape(rgb.shape)
            if ues_residual :
                edited_all[t] += ((residuals[t][0]*rgb ) * alpha[..., [0]]  +  (rgb_ori) * (1-alpha[..., [0]]) ).reshape(*video_size, 3)
            else : 
                edited_all[t] +=  (rgb * alpha[..., [0]]  +  (rgb_ori) * (1-alpha[..., [0]]) ).reshape(*video_size, 3)

    for t in range(len(uvs)):
        writer_edited_all.append_data(np.clip(edited_all[t], 0, 255).astype(np.uint8))
    
    writer_edited_all.close()


def recon_edited_video(
    model_F_mappings, model_alpha,
    jif_all, number_of_frames, rez, samples_batch, video_frames,
    num_of_maps, save_dir, device, uv_maps, reconstruct,  mask_frames, use_residual = True, edit_mask = None):

    texture_size = (1000, 1000)
    video_size = video_frames.shape[:2]
    _, _, alphas, uvs, residuals, rgbs_ = eval_data_gen(model_F_mappings, model_alpha, jif_all, number_of_frames, rez, samples_batch, num_of_maps, texture_size, device)

    #Replace alpha value to mask
    for t in range(number_of_frames):
        mask_frame = mask_frames[:,:,t, :].reshape(alphas[t].shape)
        alphas[t] = mask_frame

    writer = imageio.get_writer(os.path.join(save_dir, 'edit.mp4'))

    texture_maps = list()
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, texture_size[1]), torch.linspace(-1, 1, texture_size[0]), indexing='ij')
    grid_xy = torch.hstack((grid_y.reshape(-1, 1), grid_x.reshape(-1, 1))).to(device)
    for i in range(num_of_maps):
        texture_map = model_F_mappings[i].model_texture(grid_xy
            ).detach().cpu().numpy().reshape(*texture_size, 3)
        texture_maps.append(texture_map)
    
    if edit_mask is not None :
        for t in range(number_of_frames):
            valid = bilinear_interpolate_numpy(edit_mask, (uvs[t][0][:, 0]*0.5+0.5)*edit_mask.shape[1], (uvs[t][0][:, 1]*0.5+0.5)*edit_mask.shape[0])
            valid[valid>0] = 1.0
            valid = np.expand_dims(valid,axis=1)
            alphas[t] = alphas[t]*valid

    for t in range(number_of_frames):
        rgbs = list()
        for uv, uv_map, texture_map in zip(uvs[t], uv_maps, texture_maps):
            if not reconstruct:
                map = uv_map - texture_map
            else: map = uv_map
            rgb = bilinear_interpolate_numpy(map, (uv[:, 0]*0.5+0.5)*uv_map.shape[1], (uv[:, 1]*0.5+0.5)*uv_map.shape[0])
            rgbs.append(rgb)

        rgb_ori = (video_frames[:,:,:,t].cpu().numpy()).reshape(rgbs[0].shape)
        
        th = 0.95
        alpha_1_indices = np.where(alphas[t][:, [0]] >= th) 
        alpha_indices = np.where(alphas[t][:, [0]] < th)

        rgb_all = np.zeros_like(rgbs[0])
        
        if use_residual : 
            rgb_all[alpha_1_indices[0],:] =  residuals[t][0][alpha_1_indices[0], :]*rgbs[0][alpha_1_indices[0], :]
            rgb_all[alpha_indices[0],:] = alphas[t][:, [0]][alpha_indices[0],:] * residuals[t][0][alpha_indices[0],:]* (rgbs[0][alpha_indices[0],:] -  rgbs_[t][0][alpha_indices[0],:]) + rgb_ori[alpha_indices[0],:]
        else :
            rgb_all[alpha_1_indices[0],:] =  rgbs[0][alpha_1_indices[0], :]
            rgb_all[alpha_indices[0],:] = alphas[t][:, [0]][alpha_indices[0],:] * (rgbs[0][alpha_indices[0],:] -  rgbs_[t][0][alpha_indices[0],:]) + rgb_ori[alpha_indices[0],:] 
            
        edit_frame= cv2.cvtColor(np.clip((rgb_all.reshape(*video_size, 3) * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.append_data(np.clip((rgb_all.reshape(*video_size, 3) * 255), 0, 255).astype(np.uint8))
    writer.close()

