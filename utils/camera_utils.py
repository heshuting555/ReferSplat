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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch,masktotensor
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # TODO 变更条件
            if orig_h > 1080:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    resized_mask={k: masktotensor(mask, resolution) for k, mask in cam_info.gt_mask.items()}
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_mask=resized_mask
    sentence=cam_info.sentence
    # with open('gt_mask.txt', 'w') as f:
    #     f.write(','.join(map(str, cam_info.gt_mask.getdata())))
    
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    #print('gt_mask:', gt_mask)
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    # print('cam_info.uid:', cam_info.uid)
    # print('cam_info.R:', cam_info.R)
    # print('cam_info.T:', cam_info.T)
    # print('cam_info.FovX:', cam_info.FovX)
    # print('cam_info.FovY:', cam_info.FovY)
    # print('cam_info.image_name:', cam_info.image_name)
    # print('cam_info.uid:', cam_info.uid)
    # print('id:', id)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,sentence=sentence, 
                  image=gt_image, gt_alpha_mask=loaded_mask,gt_mask=gt_mask,
                  category=cam_info.category,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
