
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
    


def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args,model=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    ans=0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for i in range(len(view.sentence)):
            ans+=1
            sn=view.image_name
            number = re.findall(r'\d+', sn)
            number_int = int(number[0])
            output = render(view, gaussians, pipeline, background, args,sentence=view.sentence[i])
            if not args.include_feature:
                rendering = output["render"]
            else:
                rendering = output["language_feature_image"]
                rendering = torch.sigmoid(rendering)
                rendering = (rendering>=0.5).float()
                
            if not args.include_feature:
                gt = view.original_image[0:3, :, :]
                
            else:
                gt=view.gt_mask[view.category[i]] 
            np.save(os.path.join(render_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),rendering.permute(1,2,0).cpu().numpy())
            np.save(os.path.join(gts_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),gt.permute(1,2,0).cpu().numpy())
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
               
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
    random.seed(0)
    np.random.seed(0)
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
    args = get_combined_args(parser)
    args.include_feature=True
    args.iteration=5
    
    iteration=5

    for i in range(iteration):
        args.iteration=i
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
