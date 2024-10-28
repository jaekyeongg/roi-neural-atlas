
import os
from PIL import Image
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import cv2
from pathlib import Path
from config_utils import config_load
import argparse
import sys
sys.path.append("./thirdparty/VitMatte")

def init_model(model, checkpoint, device):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    if model == 'vitmatte-s':
        config = './thirdparty/VitMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = './thirdparty/VitMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    return model


def generate_trimap(mask, erode_k, erode_iter, dilate_k, dilate_iter, inversion = True):
    
    if inversion == True : 
        e_kernel = np.ones((dilate_k, dilate_k), np.uint8)
        eroded = cv2.erode(mask, e_kernel, iterations=dilate_iter)
        d_kernel = np.ones((erode_k, erode_k), np.uint8)
        dilated = cv2.dilate(mask, d_kernel, iterations=erode_iter)
    else :
        e_kernel = np.ones((erode_k, erode_k), np.uint8)
        eroded = cv2.erode(mask, e_kernel, iterations=erode_iter)
        d_kernel = np.ones((dilate_k, dilate_k), np.uint8)
        dilated = cv2.dilate(mask, d_kernel, iterations=dilate_iter)

    trimap = np.zeros_like(mask)
    trimap[dilated==255] = 128
    trimap[eroded==255] = 255
    return trimap


def mask_debug(img, mask, index, dir):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    mask_debug = mask.copy()
    mask_debug = np.expand_dims(mask_debug,axis=2)
    
    img_dump = img_bgr*mask_debug
    cv2.imwrite(dir + "/" + str(i).zfill(5) + ".png",img_dump)

    mask_debug_inv = mask_debug.copy()
    mask_debug_inv = -mask_debug_inv + 1.0

    img_dump_inv = img_bgr*mask_debug_inv
    cv2.imwrite(dir + "/" + str(i).zfill(5) + "_inv.png",img_dump_inv)



def main(config):

    cfg = config_load(config)

    erode_k = cfg["erode_kernel"]
    erode_iter = cfg["erode_iter"]
    dilate_k = cfg["dilate_kernel"]
    dilate_iter = cfg["dilate_iter"]

    print("erode : {},{} - dilate: {},{}".format(erode_k,erode_iter,dilate_k, dilate_iter) )
    mask_folder = cfg['mask_folder']
    
    input_dir =  os.path.dirname(mask_folder) + "/input"
    mask_dir = mask_folder + "/mask"

    trimap_dir = mask_dir + "/trimap"
    output = mask_folder + "/refined_mask"
    vis_mask_dir = output + "/vis"

    os.makedirs(trimap_dir, exist_ok=True)
    os.makedirs(output, exist_ok=True)
    os.makedirs(vis_mask_dir, exist_ok=True)
    
    input_files = sorted(Path(input_dir).glob('*.png'))
    mask_files = sorted(Path(mask_dir).glob('*.png'))

    model =  'vitmatte-s'
    checkpoint_dir = './thirdparty/VitMatte/models/ViTMatte_S_Com.pth'
    
    for i in range(len(input_files)):
        image_ori = np.array(Image.open(input_files[i]).convert('RGB'))
        image = image_ori.copy()
        mask = np.array(Image.open(mask_files[i]).convert('L'))

        #Mask Inversion
        tmp = mask.copy()
        mask[tmp>=128] = 0
        mask[tmp<128] = 255

        trimap = mask.copy()
        trimap = generate_trimap(trimap, erode_k, erode_iter, dilate_k, dilate_iter, inversion = True)
        cv2.imwrite(trimap_dir + "/" + str(i).zfill(5) + "_trimap.png",trimap)

        image_torch = F.to_tensor(image).unsqueeze(0)
        trimap_torch = F.to_tensor(trimap).unsqueeze(0)
        
        input =  {
            'image': image_torch,
            'trimap': trimap_torch
        }

        model = init_model(model, checkpoint_dir, 'cuda')
        alpha = model(input)['phas'].flatten(0, 2)
        alpha = alpha.cpu().detach().numpy()

        alpha[trimap == 255] = 1.0
        alpha[trimap == 0] = 0.0
        
        #Mask Reinversion
        alpha = -alpha + 1.0
        cv2.imwrite(vis_mask_dir + "/" + str(i).zfill(5) + "_alpha.png",(alpha*255).astype(np.uint8))
        np.save(output + "/" + str(i).zfill(5) + ".npy", alpha)
       

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    args = parser.parse_args()

    main(args.config)





