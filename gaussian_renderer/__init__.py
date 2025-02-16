#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import time
import torch.nn.functional as F
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def min_max_normalize_torch(points):
    min_vals = points.min(dim=0).values  # 每个坐标轴的最小值
    max_vals = points.max(dim=0).values  # 每个坐标轴的最大值
    # 将数据归一化到 [-1, 1] 范围
    normalized_points = 2 * (points - min_vals) / (max_vals - min_vals) - 1
    return normalized_points

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None,sentence=None,ratio=0.03):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # print('viewpoint_camera:', viewpoint_camera)
    # print('viewpoint_camera.image_height:', viewpoint_camera.image_height)
    # print('viewpoint_camera.image_width:', viewpoint_camera.image_width)
    # print('viewpoint_camera.FoVx:', viewpoint_camera.FoVx)
    # print('viewpoint_camera.FoVy:', viewpoint_camera.FoVy)
    # print('viewpoint_camera.world_view_transform:', viewpoint_camera.world_view_transform)
    # print('viewpoint_camera.full_proj_transform:', viewpoint_camera.full_proj_transform)
    # print('viewpoint_camera.camera_center:', viewpoint_camera.camera_center)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    t_token=pc.get_text(sentence).to("cuda")
    t_token=pc.mlp1(t_token)
    #t_token = t_token/ (t_token.norm(dim=-1, keepdim=True) + 1e-9)        
    
    #print('new_opacity:', new_opacity[0])
    # print('means3D11111111111111:', means3D)
    # print('means2D:', means2D)
    # print('opacity:', opacity)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # l_xyz=min_max_normalize_torch(pc.get_xyz)
    # x=torch.cat((l_xyz,pc._language_feature),dim=1)

    #p=pc.position(pc.get_xyz.unsqueeze(0),128)
    p=pc.mlp3(pc.get_xyz)
    p=F.normalize(p,dim=-1)
    #import pdb;pdb.set_trace()
    x=pc.mlp2(pc._language_feature)
    # t_token=F.normalize(t_token,dim=-1)
    g=pc.cross_attention(x,p,t_token)
    #g=pc.cross_attention(x,t_token)
    #features=torch.matmul(x,t_token.transpose(-1,-2)).squeeze(0)
    features=torch.matmul(g,t_token.transpose(-1,-2)).squeeze(0)
    features=features.sum(dim=-1,keepdim=True)

    # if ratio>0.01:
    sorted_indices = torch.argsort(features, descending=True)
    indices = sorted_indices[:int(len(sorted_indices) * ratio)].squeeze(1)
    # else:
    #     valid_indices = (features > 0.5).nonzero(as_tuple=True)[0]
    #     indices = valid_indices
    selected_tensors=x[indices]
    #selected_tensors = g[indices]
    #selected_tensors = g[top_ten_percent_indices]

    mean_tensor = torch.mean(selected_tensors, dim=0, keepdim=True)

    

    rendered_image, language_feature_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        language_feature_precomp = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    #print('rendered_image:', rendered_image)
    # print('language_feature_image:', language_feature_image)
    # print('radii:', radii)
    # end_time = time.time()
    # print('render_init_rasterizer程序运行时间为: %s Seconds'%(end_time-start_time))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    return {"render": rendered_image,
            "language_feature_image": language_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "mean_tensor": mean_tensor}