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
import re
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torch.nn as nn
import torch.nn.functional as F
from gaussian_renderer import render
import torchvision
import random
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads):
#         """
#         Cross Attention 模块
#         参数：
#         - dim (int): 特征维度，例如 256。
#         - num_heads (int): 注意力头的数量。
#         """
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5  # 缩放因子

#         # 线性变换
#         self.q_linear = nn.Linear(dim, dim)  # Query
#         self.k_linear = nn.Linear(dim, dim)  # Key
#         self.v_linear = nn.Linear(dim, dim)  # Value

#     def forward(self, point_features, text_embeddings):
#         """
#         执行 Cross Attention
#         参数：
#         - point_features (torch.Tensor): 点特征，形状为 [5000, 256]。
#         - text_embeddings (torch.Tensor): 文本特征，形状为 [x, 256]。
#         返回：
#         - output (torch.Tensor): 融合后的特征，形状为 [5000, x]。
#         """
#         # 获取形状
#         B, C = point_features.shape  # B=5000, C=256
#         T, _ = text_embeddings.shape  # T=x, C=256

#         # 线性变换
#         Q = self.q_linear(point_features)  # [5000, 256]
#         K = self.k_linear(text_embeddings)  # [x, 256]
#         V = self.v_linear(text_embeddings)  # [x, 256]

#         # 计算注意力分数
#         # Q @ K^T -> [5000, x]
#         attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

#         # 注意力权重归一化
#         attention_weights = F.softmax(attention_scores, dim=-1)  # [5000, x]

#         # 加权 Value
#         # attention_weights @ V -> [5000, x]
#         output = torch.matmul(attention_weights, V)+point_features  # [5000, x]

#         return output
    


def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args,model=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    #print('views:', views)
    ans=0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #print('idx',idx)
        for i in range(len(view.sentence)):
            #print('view.sentence[i]:', view.sentence[i])
            ans+=1
            sn=view.image_name
            number = re.findall(r'\d+', sn)
            number_int = int(number[0])
            #import pdb;pdb.set_trace()
            output = render(view, gaussians, pipeline, background, args,sentence=view.sentence[i])
            #print('output:', output)
            if not args.include_feature:
                rendering = output["render"]
                #print('rendering:', rendering)
            else:
                rendering = output["language_feature_image"]
                rendering = torch.sigmoid(rendering)
                rendering = (rendering>=0.5).float()
                
            if not args.include_feature:
                gt = view.original_image[0:3, :, :]
                
            else:
                #gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)
                gt=view.gt_mask[view.category[i]]
                #gt=(gt>=0.00000001).float()
                # with open(os.path.join(gts_path, '{0:05d}'.format(idx) + "_gt.txt"), 'w') as f:
                #     f.write(','.join(map(str, gt.squeeze().tolist())))
                # return 
            np.save(os.path.join(render_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),rendering.permute(1,2,0).cpu().numpy())
            np.save(os.path.join(gts_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),gt.permute(1,2,0).cpu().numpy())
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
    print('ans',ans)           
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():  
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt_cconcroskit251'+str(iteration)+'.pth')
        (model_params, first_iter) = torch.load(checkpoint,map_location=f'cuda:{torch.cuda.current_device()}')
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # if not skip_train:
        #      render_set(dataset.model_path, dataset.source_path, "trainb", iteration, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "testccc", iteration, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    random.seed(0)

    # 设置NumPy的随机种子
    np.random.seed(0)

    # 设置PyTorch的随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    #parser.add_argument('--model_path', type=str, default='output/bed')
    args = get_combined_args(parser)
    #print(args.source_path)
    #print("Rendering " + args.model_path)
    args.include_feature=True
    args.iteration=5
    
    iteration=5

    #safe_state(args.quiet)
    for i in range(iteration):
        args.iteration=i
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
