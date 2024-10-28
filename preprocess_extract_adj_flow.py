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


def run_extract_adj_flow(config):

    start_time = time()
    vid_path = Path(config["data_folder"])
    vid_name = vid_path.name
    vid_root = vid_path.parent
    
    input_dir = vid_root / vid_name / 'video_frames'

    resx = config["resx"]
    resy = config["resy"]
    max_long_edge = max(resx, resy)
    model_path = 'thirdparty/RAFT/models/raft-things.pth'

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

    adj_flow_out_dir = vid_root / vid_name / 'adj_flow'
    adj_flow_mask_out_dir = vid_root / vid_name / 'adj_flow_mask'
    os.makedirs(adj_flow_out_dir, exist_ok=True)
    os.makedirs(adj_flow_mask_out_dir, exist_ok=True)

    img_files = sorted(input_dir.glob('*.*'))
    num_imgs = len(img_files)
    
    grid = gen_grid(resy, resx, device=DEVICE).permute(2, 0, 1)[None]
    
    if use_dino == True :   
        grid_normed = normalize_coords(grid.squeeze().permute(1, 2, 0), resy, resx)
        features = [torch.from_numpy(np.load(os.path.join(vid_root / vid_name / 'dino',
                                                          os.path.basename(img_file) + '.npy'))).float().to(DEVICE)
                    for img_file in img_files]
        dino_th = 0.5
    
    cycle_th = 1.0
    with torch.no_grad():
        for i in range(num_imgs - 1):
            j = i + 1
            imfile1 = img_files[i]
            imfile2 = img_files[j]

            image1 = load_image(imfile1, resx, resy)
            image2 = load_image(imfile2, resx, resy)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_12 = model(image1, image2, iters=20, test_mode=True)
            _, flow_21 = model(image2, image1, iters=20, test_mode=True)
            
            flow_12_np = flow_12.squeeze().permute(1, 2, 0).cpu().numpy()
            flow_21_np = flow_21.squeeze().permute(1, 2, 0).cpu().numpy()

            save_12_file = os.path.join(adj_flow_out_dir,
                                         '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
            np.save(save_12_file, flow_12_np)
            save_21_file = os.path.join(adj_flow_out_dir,
                                         '{}_{}.npy'.format(os.path.basename(imfile2), os.path.basename(imfile1)))
            np.save(save_21_file, flow_21_np)

            flow_12_tensor = torch.from_numpy(flow_12_np).float().permute(2, 0, 1)[None].cuda()
            flow_21_tensor = torch.from_numpy(flow_21_np).float().permute(2, 0, 1)[None].cuda()

            #flow12 mask
            coord2 = flow_12_tensor + grid
            coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
            flow_21_sampled = F.grid_sample(flow_21_tensor, coord2_normed[None],  mode='bilinear', align_corners=True)
            map_i = flow_12_tensor + flow_21_sampled
            fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
            flow12_mask = fb_discrepancy < cycle_th
            
            if use_dino == True :
                feature_i = features[i].permute(2, 0, 1)[None]
                feature_i_sampled = F.grid_sample(feature_i, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                feature_i_sampled = feature_i_sampled.permute(1, 2, 0)
                
                feature_j = features[j].permute(2, 0, 1)[None]
                feature_j_sampled = F.grid_sample(feature_j, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                feature_sim = torch.cosine_similarity(feature_i_sampled, feature_j_sampled, dim=-1)
                feature_mask = feature_sim > dino_th
                flow12_mask = flow12_mask * feature_mask

            imageio.imwrite('{}/{}_{}.png'.format(adj_flow_mask_out_dir, os.path.basename(imfile1), os.path.basename(imfile2)), (255 * flow12_mask.cpu().numpy().astype(np.uint8)))
            
            #flow21 mask
            coord2 = flow_21_tensor + grid
            coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), resy, resx)  # [h, w, 2]
            flow_12_sampled = F.grid_sample(flow_12_tensor, coord2_normed[None],  mode='bilinear', align_corners=True)
            map_i = flow_21_tensor + flow_12_sampled
            fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
            flow21_mask = fb_discrepancy < cycle_th
            
            if use_dino == True :
                feature_j = features[j].permute(2, 0, 1)[None]
                feature_j_sampled = F.grid_sample(feature_j, grid_normed[None],  mode='bilinear', align_corners=True)[0]
                feature_j_sampled = feature_j_sampled.permute(1, 2, 0)
                
                feature_i = features[i].permute(2, 0, 1)[None]
                feature_i_sampled = F.grid_sample(feature_i, coord2_normed[None],  mode='bilinear', align_corners=True)[0].permute(1, 2, 0)
                    
                feature_sim = torch.cosine_similarity(feature_j_sampled, feature_i_sampled, dim=-1)
                feature_mask = feature_sim > dino_th
                flow21_mask = flow21_mask * feature_mask

            imageio.imwrite('{}/{}_{}.png'.format(adj_flow_mask_out_dir, os.path.basename(imfile2), os.path.basename(imfile1)), (255 * flow21_mask.cpu().numpy().astype(np.uint8)))
    
    print("extract adj flow time : " , time() - start_time)

if __name__ == '__main__':
    run_extract_adj_flow(config_load(sys.argv[1]))