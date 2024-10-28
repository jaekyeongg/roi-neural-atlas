import torch
from torch.nn import functional as F

class RGBLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, GT, pred):
        return(torch.norm(pred - GT, dim=1) ** 2).mean() * self.loss_weight

class AlphaBootstrappingLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, GT, pred):
        return F.binary_cross_entropy(pred, GT) * self.loss_weight

class RefMaskLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, GT, pred):
        return F.binary_cross_entropy(pred, GT) * self.loss_weight

class PositionLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, ref_mask_GT, ref_xyt, ref_uv):
        return  (ref_uv - ref_xyt[:,:2]).norm(dim=1).mean()* self.loss_weight


class FlowAlphaLoss():
    def __init__(self, rez, num_of_frames, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight
    
    def set_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight
        
    def __call__(self,
            of, of_mask, of_rev, of_rev_mask,
            jif_current, alpha,
            device, model_alpha):
        # forward
        xyt_forward_match, indices_forward = get_flow_match(of, of_mask, jif_current, self.rez, self.num_of_frames, True)
        alpha_forward_match = model_alpha(xyt_forward_match.to(device))
        loss_forward = (alpha[indices_forward] - alpha_forward_match).abs().mean()

        # backward
        xyt_backward_match, indices_backward = get_flow_match(of_rev, of_rev_mask, jif_current, self.rez, self.num_of_frames, False)
        alpha_backward_match = model_alpha(xyt_backward_match.to(device))
        loss_backward = (alpha[indices_backward] - alpha_backward_match).abs().mean()

        return (loss_forward + loss_backward) * 0.5 * self.loss_weight

class RefFlowAlphaLoss():
    def __init__(self, rez, num_of_frames, loss_weight, roi_frame):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight
        self.roi_frame = roi_frame

    def __call__(self,
            ref_of, ref_of_mask, jif_current, alpha,
            device, model_alpha):

        ref_of_indice = torch.where(
            ref_of_mask[jif_current[1].ravel(), jif_current[0].ravel(), jif_current[2].ravel()])
        indices = ref_of_indice[0]
        ref_frames = ref_of_indice[1] + self.roi_frame
        
        jif_current_next = jif_current[:, indices, 0]
        ref_flows = ref_of[jif_current_next[1], jif_current_next[0], :, jif_current_next[2], 0]

        jif_ref_match = torch.stack((
            jif_current_next[0] + ref_flows[:, 0],
            jif_current_next[1] + ref_flows[:, 1],
            ref_frames
        ))
        xyt_ref_match = torch.stack((
            jif_ref_match[0] / (self.rez / 2) - 1,
            jif_ref_match[1] / (self.rez / 2) - 1,
            jif_ref_match[2] / (self.num_of_frames / 2) - 1
        )).T

        ref_alpha_match = model_alpha(xyt_ref_match.to(device))
        loss_forward = (alpha[indices] - ref_alpha_match).abs().mean()
        return (loss_forward) * self.loss_weight

class FlowMappingLoss():
    def __init__(self, rez, num_of_frames, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight
        
    def __call__(self,
            of, of_mask, of_rev, of_rev_mask,
            jif_current, uv, uv_scale,
            device, model_F_mapping, use_alpha=False, alpha=None):
        # forward

        xyt_forward_match, indices_forward = get_flow_match(of, of_mask, jif_current, self.rez, self.num_of_frames, True)

        uv_forward = uv[indices_forward]
        uv_forward_match = model_F_mapping(xyt_forward_match.to(device))
        loss_forward = (uv_forward_match - uv_forward).norm(dim=1) * self.rez / (2 * uv_scale)

        # backward
        xyt_backward_match, indices_backward = get_flow_match(of_rev, of_rev_mask, jif_current, self.rez, self.num_of_frames, False)
        uv_backward = uv[indices_backward]
        uv_backward_match = model_F_mapping(xyt_backward_match.to(device))
        loss_backward = (uv_backward_match - uv_backward).norm(dim=1) * self.rez / (2 * uv_scale)

        if use_alpha:
            loss_forward = loss_forward * alpha[indices_forward].squeeze()
            loss_backward = loss_backward * alpha[indices_backward].squeeze()

        return (loss_forward.mean() + loss_backward.mean()) * 0.5 * self.loss_weight

class RefFlowMappingLoss():
    def __init__(self, rez, num_of_frames, loss_weight,roi_frame):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight
        self.roi_frame = roi_frame

    def __call__(self,
            ref_of, ref_of_mask,
            jif_current, uv, uv_scale,
            device, model_F_mapping, use_alpha=True, alpha=None):
        
        ref_of_indice = torch.where(
            ref_of_mask[jif_current[1].ravel(), jif_current[0].ravel(), jif_current[2].ravel()])
        indices = ref_of_indice[0]
        ref_frames = ref_of_indice[1] + self.roi_frame

        jif_current_next = jif_current[:, indices, 0]
        ref_flows = ref_of[jif_current_next[1], jif_current_next[0], :, jif_current_next[2], 0]

        jif_ref_match = torch.stack((
            jif_current_next[0] + ref_flows[:, 0],
            jif_current_next[1] + ref_flows[:, 1],
            ref_frames
        ))
        xyt_ref_match = torch.stack((
            jif_ref_match[0] / (self.rez / 2) - 1,
            jif_ref_match[1] / (self.rez / 2) - 1,
            jif_ref_match[2] / (self.num_of_frames / 2) - 1
        )).T
    
        uv_forward = uv[indices]
        uv_forward_match = model_F_mapping(xyt_ref_match.to(device))

        loss_forward = (uv_forward_match - uv_forward).norm(dim=1) * self.rez / (2 * uv_scale)
        loss_forward = loss_forward * alpha[indices].squeeze()

        return loss_forward.mean() * self.loss_weight


class RigidityLoss():
    def __init__(self, rez, num_of_frames, d, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.d = d
        self.loss_weight = loss_weight

    def __call__(self,
            jif_current,
            uv, uv_scale, device, model_F_mapping):
        x_patch = torch.cat((jif_current[0], jif_current[0] - self.d))
        y_patch = torch.cat((jif_current[1] - self.d, jif_current[1]))
        t_patch = torch.cat((jif_current[2], jif_current[2]))
        xyt_p = torch.cat((
            x_patch / (self.rez / 2) - 1,
            y_patch / (self.rez / 2) - 1,
            t_patch / (self.num_of_frames / 2) - 1
        ), dim=1).to(device)

        uv_p = model_F_mapping(xyt_p)
        u_p = uv_p[:, 0].reshape(2, -1) # u(x, y-d, t), u(x-d, y, t)
        v_p = uv_p[:, 1].reshape(2, -1) # v(x, y-d, t), v(x-d, y, t)

        u_p_d = (uv[:, 0].unsqueeze(0) - u_p) * self.rez / 2
        v_p_d = (uv[:, 1].unsqueeze(0) - v_p) * self.rez / 2

        du_dx = u_p_d[1]
        du_dy = u_p_d[0]
        dv_dy = v_p_d[0]
        dv_dx = v_p_d[1]

        J = torch.stack((
            du_dx, du_dy, dv_dx, dv_dy
        ), dim=-1).reshape(-1, 2, 2)
        J = J / uv_scale / self.d
        JtJ = torch.matmul(J.transpose(1, 2), J)

        # 2x2 matrix inverse for faster computation
        A = JtJ[:, 0, 0] + 0.001
        B = JtJ[:, 0, 1]
        C = JtJ[:, 1, 0]
        D = JtJ[:, 1, 1] + 0.001
        JtJinv = torch.stack((
            D, -B, -C, A
        ), dim=-1).reshape(-1, 2, 2) / (A * D - B * C).reshape(-1, 1, 1)

        loss = (JtJ ** 2).sum(dim=[1, 2]).sqrt() + (JtJinv ** 2).sum(dim=[1, 2]).sqrt()

        return loss.mean() * self.loss_weight


class RefineLoss():
    def __init__(self, rez, num_of_frames, loss_weight, refine_th, use_residual):
        self.loss_weight = loss_weight
        self.reine_th = refine_th
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.use_residual = use_residual

    def __call__(self, model_F_mappings, video_frames, xyt_current, jif_current, init_masks, model_alpha, device):
        sample_indices = torch.where(init_masks[jif_current[1].ravel(), jif_current[0].ravel(), jif_current[2].ravel()]>250.0)
        jif_samples = jif_current[:, sample_indices[0], 0]
        xyt_samples = torch.stack((
            jif_samples[0] / (self.rez  / 2) - 1,
            jif_samples[1] / (self.rez  / 2) - 1,
            jif_samples[2] / (self.num_of_frames / 2) - 1
        )).T.to(device)

        #print("xyt_samples shape : ", xyt_samples.shape)
        uvs, residuals, rgb_textures = zip(*[i(xyt_samples, True, True) for i in model_F_mappings])
        atlas = rgb_textures[0]
        rgb_ori = video_frames[jif_samples[1, :], jif_samples[0, :], :, jif_samples[2, :]].squeeze(1).to(device)
        if self.use_residual == True :
            indices = torch.where(torch.mean(((residuals[0]*atlas) - rgb_ori).abs(),dim=1) < self.reine_th)
        else :
            indices = torch.where(torch.mean((atlas- rgb_ori).abs(),dim=1) < self.reine_th)

        if len(indices[0]) == 0 : 
            return 0.0

        refine_points = xyt_samples[indices[0], :]
        refine_alphas = model_alpha(refine_points)
        refine_loss = (1.0 - refine_alphas.squeeze())**2

        return refine_loss.mean()* self.loss_weight


class ResidualRegLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, residual):
        return ((residual - 1) ** 2).mean() * self.loss_weight


def get_flow_match(of, of_mask, jif_current, rez, num_of_frames, is_forward):
    next_mask = torch.where(
        of_mask[jif_current[1].ravel(), jif_current[0].ravel(), jif_current[2].ravel()])

    indices = next_mask[0]
    num_next_frames = 2 ** next_mask[1]

    jif_next = jif_current[:, indices, 0]
    next_flows = of[jif_next[1], jif_next[0], :, jif_next[2], next_mask[1]]

    if is_forward == False:
        num_next_frames *= -1
    jif_next_match = torch.stack((
        jif_next[0] + next_flows[:, 0],
        jif_next[1] + next_flows[:, 1],
        jif_next[2] + num_next_frames
    ))
    xyt_next_match = torch.stack((
        jif_next_match[0] / (rez / 2) - 1,
        jif_next_match[1] / (rez / 2) - 1,
        jif_next_match[2] / (num_of_frames / 2) - 1
    )).T

    return xyt_next_match, indices
