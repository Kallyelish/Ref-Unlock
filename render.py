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
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.graphics_utils import getWorld2View2
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    ref_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ref_map")
    ref_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ref_color")
    trans_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "trans_color")
    comp_ref_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "comp_ref_color")
    comp_trans_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "comp_trans_color")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(ref_map_path, exist_ok=True)
    makedirs(ref_color_path, exist_ok=True)
    makedirs(trans_color_path, exist_ok=True)
    makedirs(comp_ref_color_path, exist_ok=True)
    makedirs(comp_trans_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t0 = time.time()
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1 - t0)
        print(f'Render time: \033[1;35m{t1 - t0:.5f}\033[0m')
        renderings = rendering["render"]
        ref_map = rendering["trans_weights"]
        ref_color = rendering["ref_color"]
        trans_color = rendering["trans_color"]
        comp_ref_color = ref_map * ref_color
        comp_trans_color = (1- ref_map) * trans_color
        depth = rendering["depth"]
        depth = depth / depth.max()

        gt = view.original_image[0:3, :, :]

        # if args.train_test_exp:
        #     renderings = rendering[..., renderings.shape[-1] // 2:]
        #     gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(renderings, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ref_map, os.path.join(ref_map_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ref_color, os.path.join(ref_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(trans_color, os.path.join(trans_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(comp_ref_color, os.path.join(comp_ref_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(comp_trans_color, os.path.join(comp_trans_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
    t = np.array(t_list)
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.ref_sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--render_images", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--camera_type", default='spiral', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)