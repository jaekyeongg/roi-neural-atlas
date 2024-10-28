import numpy as np
import torch
import cv2
import torch.optim as optim
import imageio

from PIL import Image
import os


def load_input_data(
        resy, resx,
        data_folder, use_mask_rcnn_bootstrapping,
        filter_optical_flow):

    video_frames_dir = data_folder / 'video_frames'
    flow_dir = data_folder / 'adj_flow'
    flow_mask_dir = data_folder / 'adj_flow_mask'
    mask_dirs = data_folder / 'init_masks'

    input_files = sorted(list(video_frames_dir.glob('*.jpg')) + list(video_frames_dir.glob('*.png')))

    number_of_frames=len(input_files)
    video_frames = torch.zeros((resy, resx, 3, number_of_frames))
    video_frames_dx = torch.zeros((resy, resx, 3, number_of_frames))
    video_frames_dy = torch.zeros((resy, resx, 3, number_of_frames))

    mask_frames = torch.zeros((resy, resx, number_of_frames, 1))

    optical_flows = torch.zeros((resy, resx, 2, number_of_frames,  1))
    optical_flows_mask = torch.zeros((resy, resx, number_of_frames,  1))
    optical_flows_reverse = torch.zeros((resy, resx, 2, number_of_frames,  1))
    optical_flows_reverse_mask = torch.zeros((resy, resx, number_of_frames, 1))

    mask_files = sorted(list(mask_dirs.glob('*.jpg')) + list(mask_dirs.glob('*.png')))

    for i in range(number_of_frames):
        file1 = input_files[i]
        im = np.array(Image.open(str(file1))).astype(np.float64) / 255.
        if use_mask_rcnn_bootstrapping:
            mask = np.array(Image.open(str(mask_files[i]))).astype(np.float64) / 255.
            mask = cv2.resize(mask, (resx, resy), cv2.INTER_NEAREST)
            mask_frames[:, :, i, 0] = torch.from_numpy(mask)

        video_frames[:, :, :, i] = torch.from_numpy(cv2.resize(im[:, :, :3], (resx, resy)))
        video_frames_dy[:-1, :, :, i] = video_frames[1:, :, :, i] - video_frames[:-1, :, :, i]
        video_frames_dx[:, :-1, :, i] = video_frames[:, 1:, :, i] - video_frames[:, :-1, :, i]

    for i in range(number_of_frames - 1):
        file1 = input_files[i]
        j = i + 1
        file2 = input_files[j]

        fn1 = file1.name
        fn2 = file2.name

        flow12_fn = flow_dir / f'{fn1}_{fn2}.npy'
        flow21_fn = flow_dir / f'{fn2}_{fn1}.npy'
        flow12 = np.load(flow12_fn)
        flow21 = np.load(flow21_fn)

        mask_flow = Image.open(flow_mask_dir / f'{fn1}_{fn2}.png')
        mask_flow = np.array(mask_flow).astype(np.float32)
        mask_flow_reverse = Image.open(flow_mask_dir / f'{fn2}_{fn1}.png')
        mask_flow_reverse = np.array(mask_flow_reverse).astype(np.float32)

        optical_flows[:, :, :, i, 0] = torch.from_numpy(flow12)
        optical_flows_reverse[:, :, :, j, 0] = torch.from_numpy(flow21)

        if filter_optical_flow:
            optical_flows_mask[:, :, i, 0] = torch.from_numpy(mask_flow)
            optical_flows_reverse_mask[:, :, j, 0] = torch.from_numpy(mask_flow_reverse)
        else:
            optical_flows_mask[:, :, i, 0] = torch.ones_like(mask_flow)
            optical_flows_reverse_mask[:, :, j, 0] = torch.ones_like(mask_flow_reverse)

    
    ref_opt_flows_dir = data_folder /'ref_flow'
    ref_opt_flows_mask_dir = data_folder /'ref_flow_mask'
    ref_opt_files = sorted(list(ref_opt_flows_dir.glob('*.npy')))
    ref_opt_mask_files = sorted(list(ref_opt_flows_mask_dir.glob('*.png')))
    ref_opt_flows = torch.zeros((resy, resx, 2, number_of_frames,  1))
    ref_opt_flows_mask = torch.zeros((resy, resx, number_of_frames,  1))
    for i in range(number_of_frames):
        ref_opt = np.load(ref_opt_files[i])
        ref_opt_mask = Image.open(ref_opt_mask_files[i])
        ref_opt_mask = np.array(ref_opt_mask).astype(np.float32)
        ref_opt_flows[:, :, :, i, 0] = torch.from_numpy(ref_opt)
        ref_opt_flows_mask[:, :, i, 0] = torch.from_numpy(ref_opt_mask)
    
    scale_factor =  max(im[0].shape[0], im[0].shape[1])/max(resx, resy)
  
    return optical_flows_mask, video_frames, optical_flows_reverse_mask, mask_frames, video_frames_dx, video_frames_dy, optical_flows_reverse, optical_flows, ref_opt_flows, ref_opt_flows_mask, scale_factor


def get_tuples(number_of_frames, video_frames):
    # video_frames shape: (resy, resx, 3, num_frames), mask_frames shape: (resy, resx, num_frames)
    jif_all = []
    for f in range(number_of_frames):
        mask = (video_frames[:, :, :, f] > -1).any(dim=2)
        relis, reljs = torch.where(mask > 0.5)
        jif_all.append(torch.stack((reljs, relis, f * torch.ones_like(reljs))))
    return torch.cat(jif_all, dim=1)

def pre_train_mapping(
        model_F_mapping, frames_num, uv_mapping_scale,
        resx, resy, rez, device, pretrain_iters=100):
    optimizer_mapping = optim.Adam(model_F_mapping.get_optimizer_list())
    for i in range(pretrain_iters):
        for f in range(frames_num):
            i_s_int = torch.randint(resy, (np.int64(10000), 1))
            j_s_int = torch.randint(resx, (np.int64(10000), 1))

            i_s = i_s_int / (rez / 2) - 1
            j_s = j_s_int / (rez / 2) - 1

            xyt = torch.cat((
                j_s,
                i_s,
                (f / (frames_num / 2) - 1) * torch.ones_like(i_s)
            ), dim=1).to(device)
            uv_temp = model_F_mapping(xyt)

            optimizer_mapping.zero_grad()

            loss = (xyt[:, :2] * uv_mapping_scale - uv_temp).norm(dim=1).mean()
            loss.backward()
            optimizer_mapping.step()
    return model_F_mapping

def save_video(video_frames, results_folder, cfg = None):
    input_path = str(results_folder) + "/input/"
    os.makedirs(input_path, exist_ok=True)
    input_video = imageio.get_writer(
        "%s/input_video.mp4" % (results_folder), fps=10)
    for i in range(video_frames.shape[3]):
        cur_frame = video_frames[:, :, :, i].clone()
        input_video.append_data((cur_frame.numpy() * 255).astype(np.uint8))
        cv2.imwrite(input_path + str(i).zfill(5) + ".png", cv2.cvtColor((cur_frame.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if cfg is not None and i == cfg["reference_frame"]:
            scale_factor = cfg["scale_factor"]
            poly =np.array([[int(cfg["roi"][0]//scale_factor),int(cfg["roi"][1]//scale_factor)],
                        [int(cfg["roi"][2]//scale_factor),int(cfg["roi"][3]//scale_factor)],
                        [int(cfg["roi"][4]//scale_factor),int(cfg["roi"][5]//scale_factor)],
                        [int(cfg["roi"][6]//scale_factor),int(cfg["roi"][7]//scale_factor)]
                   ])
            roi_frmae = cv2.cvtColor((cur_frame.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            roi_frmae_rect = cv2.polylines(roi_frmae,[poly],True,(0,255,0),2)


            cv2.imwrite(str(results_folder) + "/roiframe_rect.png" , roi_frmae_rect)

    input_video.close()

from time import time
from datetime import timedelta

class Timer():
    def __init__(self):
        self.time_list = list()
        self.start_time = None

    def start(self):
        assert self.start_time is None
        self.start_time = time()

    def stop(self):
        assert self.start_time is not None
        self.time_list.append(time() - self.start_time)
        self.start_time = None

    def average(self):
        assert len(self.time_list) != 0
        return sum(self.time_list) / len(self.time_list)

    def last_period(self, period):
        assert len(self.time_list) >= period
        return sum(self.time_list[-period:]) / period

    def ETA(self, iterations, sample_num):
        assert len(self.time_list) >= sample_num
        ETA = self.last_period(sample_num) * iterations
        return timedelta(seconds=int(ETA))
