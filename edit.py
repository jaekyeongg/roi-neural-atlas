import torch
import numpy as np
from pathlib import Path
from networks import build_network
from unwrap_utils import load_input_data, get_tuples
from config_utils import config_load
from evaluate import recon_edited_video
import argparse
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config):

    #config
    cfg = config_load(config)
    data_folder = Path(cfg["data_folder"])
    resx = np.int64(cfg["resx"])
    resy = np.int64(cfg["resy"])
    vid_name = data_folder.name
    vid_root = data_folder.parent

    config_mappings = cfg["model_mapping"]
    config_alpha = cfg["alpha"]
    samples = cfg["samples_batch"]
    use_residual = cfg["use_residual"]

    #Editing config
    edit_checkpoint = cfg["edit_checkpoint"]
    ori_texture_file = cfg["ori_texture"]
    edit_texture_file = cfg["edit_texture"]
    use_alpha_mask = cfg["use_alpha_mask"]

    results_folder = Path(('/').join(edit_checkpoint.split('/')[:-1])+"/")
    _, video_frames, _, _, _, _, _, _, _, _, scale_factor = load_input_data(
        resy, resx, data_folder, True,  True)
    cfg["scale_factor"] = scale_factor

    ori_texture = np.array(Image.open(ori_texture_file))
    ori_texture_rgb = ori_texture[:,:,:3]
    ori_texture_a = np.expand_dims(ori_texture[:,:,3],axis=2)

    edit_texture = np.array(Image.open(edit_texture_file))
    edit_texture_rgb = edit_texture[:,:,:3]
    edit_texture_a = np.expand_dims(edit_texture[:,:,3],axis=2)

    edit_mask = np.mean(np.abs((ori_texture_rgb*ori_texture_a-edit_texture_rgb*edit_texture_a)),axis=2)>0
    edit_mask = (edit_mask*255).astype(np.uint8)
    cv2.imwrite(str(results_folder) + "/edit_mask.png", edit_mask)

    number_of_frames = video_frames.shape[3]
    jif_all = get_tuples(number_of_frames, video_frames)
    rez = np.maximum(resx, resy)
    
    num_of_maps = len(config_mappings)

    model_F_mappings = torch.nn.ModuleList()
    for config_mapping in config_mappings:
        model_F_mappings.append(
            build_network(device=device, **config_mapping))

    model_alpha = build_network(
        device=device,
        **config_alpha)
    
    init_file = torch.load(edit_checkpoint)
    model_F_mappings.load_state_dict(init_file["model_F_mappings_state_dict"])
    model_alpha.load_state_dict(init_file["model_F_alpha_state_dict"])
    start_iteration = init_file["iteration"]
    
    save_dir = str(results_folder)

    mask_frames = np.zeros((resy, resx, number_of_frames, 1))
    if use_alpha_mask == True :
        mask_dir = str(results_folder) + "/refined_mask/"
        mask_files = sorted(Path(mask_dir).glob('*.npy'))
        for i in range(number_of_frames):
            mask = np.load(str(mask_files[i]))
            mask_frames[:, :, i, 0] = torch.from_numpy(mask)
    else : 
        mask_dir = str(results_folder) + "/mask/"
        mask_files = sorted(Path(mask_dir).glob('*.png'))
        for i in range(number_of_frames):
            mask = np.array(Image.open(mask_files[i]).convert('L'))/255.0
            mask_frames[:, :, i, 0] = torch.from_numpy(mask)
        
    custom_uvs = [edit_texture_rgb / 255.0]

    recon_edited_video(
        model_F_mappings, model_alpha,
        jif_all, number_of_frames, rez, samples, video_frames,
        num_of_maps, save_dir, device, custom_uvs, True, mask_frames, use_residual=use_residual, edit_mask = edit_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    args = parser.parse_args()

    args = parser.parse_args()
    main(args.config)
