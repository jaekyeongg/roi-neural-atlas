#data and roi config
data_folder = 'data/lucia'
reference_frame = 20
roi = [860,200, 1310,200, 1310,750, 860,750]

#training config
results_folder_name = 'results'
maximum_number_of_frames = 70
resx = 768
resy = 432
iters_num = 60000

#mask refinement config
refine_iters_num = 20000
refine_th = 0.025

samples_batch = 10000
uv_mapping_scales = [1.0]
pretrain_iter_number = 100
load_checkpoint = False
checkpoint_path = ''
folder_suffix = 'test'
use_residual = True

evaluation = dict(
    interval = 10000,
    save_ckpt_interval = 1000,
    samples_batch = 10000)

losses = dict(
    alpha_bootstrapping = dict(
        weight = 0.6,
        stop_iteration = 10000),
        
    rgb = dict(
        weight = 1),
    rigidity = dict(
        weight = 0.002,
        derivative_amount = 1),

    ref_mask = dict(
        weight = 0.4),
    flow_alpha = dict(
        weight = 0.8),
    ref_flow_alpha = dict(
        weight = 0.8),

    optical_flow = dict(
        weight = 0.1),
    ref_optical_flow = dict(
        weight = 0.02),

    position = dict(
        weight = 0.6),
    residual_reg = dict(
        weight = 0.8),

    refine = dict(
        weight = 0.01),
    refine_flow_alpha = dict(
        weight = 0.1)
    )

config_xyt = {
    'base_resolution': 16,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'n_levels': 16,
    'otype': 'HashGrid',
    'per_level_scale': 1.25}

config_uv = {
    'base_resolution': 16,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'n_levels': 16,
    'otype': 'HashGrid',
    'per_level_scale': 1.25}

model_mapping = [{
    'model_type': 'EncodingMappingNetwork',
    'pretrain': True,
    'texture': {
        'model_type': 'EncodingTextureNetwork',
        'encoding_config': config_uv},
    'residual': {
        'model_type': 'ResidualEstimator',
        'encoding_config': None},
    'encoding_config': None,
    'num_layers': 4,
    'num_neurons': 256
}]

alpha = {
    'model_type': 'EncodingAlphaNetwork',
    'encoding_config': config_xyt,
    'num_layers': 4,
    'num_neurons': 32
}


#This is the morphology config for creating the trimap used in matting as post processing.
#you may change dilation iter or erosion iter to get more natural editing results.
mask_folder = './results/lucia_test/080000'
dilate_kernel = 3
dilate_iter = 2
erode_kernel = 3
erode_iter = 1

#editing config
edit_checkpoint = './results/lucia_test/080000/checkpoint'
ori_texture = './results/lucia_test/080000/tex0.png'
edit_texture = './results/lucia_test/080000/tex0_edit.png'
use_alpha_mask = True