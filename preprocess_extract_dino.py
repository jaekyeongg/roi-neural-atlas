# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

import cv2
import torch
import sys
from config_utils import config_load, config_save
from pathlib import Path


sys.path.insert(0, './thirdparty/dino/')
import utils
import vision_transformer as vits
sys.path.remove('./thirdparty/dino/')


def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out


def read_frame(frame_dir, scale_size=[432]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    #img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if (ori_h > ori_w):
        tw = scale_size[0]
        th = (tw * ori_h) / ori_w
        th = int((th // 64) * 64)
    else:
        th = scale_size[0]
        tw = (th * ori_w) / ori_h
        tw = int((tw // 64) * 64)

    img = cv2.resize(img, (tw, th),  interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


if __name__ == '__main__':

    config = config_load(sys.argv[1])
    resx = config["resx"]
    resy = config["resy"]
    
    args = argparse.Namespace()
    args.pretrained_weights = '.'
    args.arch = 'vit_small'
    args.patch_size = 8
    args.checkpoint_key = "teacher"
    args.n_last_frames = 7
    args.size_mask_neighborhood = 12
    args.topk = 5
    args.bs = 6

    # building network
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    vid_path = Path(config["data_folder"])
    vid_name = vid_path.name
    vid_root = vid_path.parent
    input_dir = vid_root / vid_name / 'video_frames'
    frame_list = sorted(input_dir.glob('*.*'))
    save_dir = vid_root / vid_name / 'dino'
    os.makedirs(save_dir, exist_ok=True)

    for frame_path in tqdm(frame_list):
        frame, ori_h, ori_w = read_frame(str(frame_path), scale_size = [resy])
        frame_feat, h, w = extract_feature(model, frame, return_h_w=True)  # dim x h*w
        frame_feat = frame_feat.reshape(h, w, -1)
        frame_feat = frame_feat.cpu().numpy()
        frame_name = os.path.basename(frame_path)
        np.save(os.path.join(save_dir, frame_name + '.npy'), frame_feat)

