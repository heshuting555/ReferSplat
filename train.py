import os
import torch
import matplotlib.pyplot as plt
from random import randint
from utils.loss_utils import l1_loss, ssim,bce_loss,dice_loss,multi_pos_cross_entropy
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,epoch):
    
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color,dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, 100000), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    ratio=0.1
    total_loss=[]
    iteration=1
    for epoch in range(epoch_num):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        while len(viewpoint_stack)!=0:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            text_feature=gaussians.get_text(viewpoint_cam.sentence).to("cuda")
            for i in range(len(viewpoint_cam.sentence)):
                iter_start.record()
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt,sentence=viewpoint_cam.sentence[i],ratio=ratio)
                language_feature,mean_tensor=render_pkg["language_feature_image"],render_pkg["mean_tensor"]
                if opt.include_feature:
                    features=gaussians.mlp1(text_feature)
                    features=torch.mean(features, dim=1)
                    mean_tensor=F.normalize(mean_tensor,dim=1)
                    features=F.normalize(features,dim=1)

                    cosine_similarities=(torch.matmul(mean_tensor,features.T)/0.1).to("cuda")
                    
                    sentence_tensor = torch.zeros(len(viewpoint_cam.sentence))
                    
                    sentence_tensor[i] = 1
                    current_category = viewpoint_cam.category[i]
                    category_indices = [idx for idx, cat in enumerate(viewpoint_cam.category) if cat == current_category]
                    sentence_tensor[category_indices] = 1
                    sentence_tensor = sentence_tensor.unsqueeze(0).to("cuda")
                    com_loss = multi_pos_cross_entropy(cosine_similarities, sentence_tensor)
                    gt_mask = viewpoint_cam.gt_mask[viewpoint_cam.category[i]].to("cuda")
                    loss = bce_loss(language_feature, gt_mask)+0.1*com_loss
                    loss.backward()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                iter_end.record()
                iteration+=1
                if iteration%2000==0 and ratio>0.005:
                    ratio=ratio*0.6
                    if ratio<0.005:
                        ratio=0.005
                with torch.no_grad():
                    ema_loss_for_log = 0.4*loss.item()+0.6*ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(10)
                        total_loss.append(ema_loss_for_log)
        
        torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt_cbasetea251" + str(epoch) + ".pth")
    progress_bar.close()
    
if __name__ == "__main__":
    # Set up command line argument parser
    torch.set_default_dtype(torch.float32)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = 'output/teatime/chkpnt30000.pth')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    epoch_num=5
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,epoch_num)

    print("\nTraining complete.")