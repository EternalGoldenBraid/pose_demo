import os
from time import perf_counter
import sys
import warnings
import json
from itertools import product
from pathlib import Path
from numba import njit, prange
import cv2
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
import pyrealsense2 as rs
from scipy import stats

from ipdb import iex
from matplotlib import pyplot as plt

from os.path import join as pjoin
from bop_toolkit_lib import inout
warnings.filterwarnings("ignore")

#breakpoint()
#base_path = os.path.dirname(os.path.abspath("."))
#sys.path.append(base_path)
#
#breakpoint()
from utility import (timeit,  cam_control,
                     #load_segmentation_model,
                     load_segmentation_model_chroma)
from utility.load_pose_estimator import PoseEstimator

from lib.render_cloud import load_cloud, render_cloud # TODO Use this
from lib import ove6d

from dataset import demo_dataset
from configs import config as cfg

#DEVICE = torch.device('cuda')
#DEVICE = torch.device('cpu')
#DEVICE = 'cpu'
DEVICE = 'cuda'

def main(args):

    cfg.DATASET_NAME = 'huawei_box'        # dataset name
    cfg.USE_ICP = args.icp

    # Load camera module
   #timeit.log("Realsense initialization.")
   # TODO change to row_major for numpy...? What's torch
    cam = cam_control.Camera(size=(cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT), framerate=60)
    depth_scale, cam_K =  cam.depth_scale, cam.cam_K
    cam_K_np = cam_K.numpy()
   #timeit.endlog()

    # load segmentation module
    if args.segment_method == 'chromakey':
        segmentator = load_segmentation_model_chroma.load(cfg=cfg, device=DEVICE)
    else:
        raise NotImplementedError("Enable segmentator selection (detectron)")
        #segmentator = load_segmentation_model.load(model=args.segment_method, cfg=cfg, device=DEVICE)

    # Load mesh models
    #timeit.log("Loading data.")
    dataroot: Path = Path(os.path.dirname(__file__))/Path(cfg.DATA_PATH)
    #dataroot = Path(os.path.realpath(__file__)).parent.parent/Path(cfg.DATA_PATH)
    dataset = demo_dataset.Dataset(data_dir=dataroot/ cfg.DATASET_NAME, cfg=cfg,
                cam_K=cam_K, cam_height=cfg.RENDER_HEIGHT, cam_width=cfg.RENDER_WIDTH,
                n_triangles=args.n_triangles)

    # Load pose estimation module
    obj_id: int = args.obj_id.value
    codebook_path: Path = dataroot/'object_codebooks'/ cfg.DATASET_NAME / \
        'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR) / \
        'views_{}'.format(str(cfg.RENDER_NUM_VIEWS))

    #timeit.endlog()
    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, obj_id=obj_id,
            model_path=Path('checkpoints','OVE6D_pose_model.pth'),
            device=DEVICE, dataset=dataset, codebook_path=codebook_path)

    # TODO Renderer usable for renering mesh 
    # Passing initialized renderer? Implications?
    #dataset.object_renderer = pose_estimator.obj_renderer

    # Streaming loop
    mod_count: int = 0
    buffer_size: int = args.buffer_size
    frame_buffer = np.empty([buffer_size, cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH])
    d_max = 1

    try:
        while True:

            fps_start = perf_counter()
            
            depth_image, color_image = cam.get_image()

            depth_image[depth_image*depth_scale > d_max] = 0

            masks, masks_gpu, scores = segmentator(color_image)

            #breakpoint()
            #breakpoint()
            cv2.imshow('mask', masks[0].astype(float))

            #depth_colormap = cv2.applyColorMap(
            #        cv2.convertScaleAbs(depth_image,
            #                            #alpha=0.03),
            #                            alpha=0.9, beta=0.0),
            #        cv2.COLORMAP_JET) 
            #depth_colormap = (255*((depth_image - d_max)/d_max)).astype(np.uint8)[..., None].repeat(repeats=3,axis=2)
            #depth_colormap = cv2.applyColorMap(
            #        (255*((depth_image - d_max)/d_max)).astype(np.uint8),
            #        cv2.COLORMAP_JET) 
            depth_colormap = cv2.applyColorMap(
                    (255*(depth_image).astype(np.uint8)),
                    cv2.COLORMAP_JET) 
            if masks.size != 0:

                ### TODO: Can we get depth_image dircetly to gpu from sensor and skip gpu --> cpu with <mask>
                R, t = pose_estimator.estimate_pose(obj_mask=masks_gpu[0][None,...],
                            obj_depth=torch.tensor(
                                (depth_image*masks[0]*depth_scale).astype(np.float32)).squeeze()[None,...]
                            )

                ### TODO Multi object support.
                #obj_depths = torch.tensor([(depth_image*mask*depth_scale).astype(np.float32) for mask in masks])
                #R, t = pose_estimator.estimate_poses(obj_masks=masks_gpu, scores=scores,
                #            obj_depths=obj_depths.squeeze())

                #timeit.endlog()
                #timeit.log("Rendering.")

                for transform_idx in range(R.shape[0]):
                    #color_image, done = dataset.render_cloud(obj_id=obj_id, 
                    #        R=R[transform_idx].numpy().astype(np.float32), 
                    #        t=t[transform_idx].numpy()[...,None].astype(np.float32),
                    #        image=color_image)

                    color_image, done = dataset.render_mesh(obj_id=obj_id, 
                             R=R[transform_idx].numpy().astype(np.float32), 
                             t=t[transform_idx].numpy()[...,None].astype(np.float32),
                             image=color_image.copy())


                images = np.hstack([ 
                    color_image, 
                    depth_colormap*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                    #color_image*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                    ])
            else:
                images = np.hstack((color_image, depth_colormap))

            
            cv2.putText(images, f"fps: {(1/(perf_counter()-fps_start)):2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        del cam

if __name__=="__main__":
    import argparse

    from enum import Enum, unique
    class ArgTypeMixin(Enum):

        @classmethod
        def argtype(cls, s: str) -> Enum:
            try:
                return cls[s]
            except KeyError:
                raise argparse.ArgumentTypeError(
                    f"{s!r} is not a valid {cls.__name__}")

        def __str__(self):
            return self.name

    @unique
    class ObjectIds(ArgTypeMixin, Enum):
        box = 1
        head_phones = 3
        engine_main = 4
        dual_sphere = 5
        tea = 6
        bolt = 7
        wrench = 8
        lego = 9
        eraser_lowq = 10
        eraser_highq = 11
        #eraser_lowq = 10
        box_synth = 12
        gear_assembled = 13
        clipper = 14
        pot = 15


    
    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')

    parser.add_argument('-o','--object', dest='obj_id',
                        type=ObjectIds.argtype, default=ObjectIds.box,
                        choices=ObjectIds,
                        help='Object names')
    #parser.add_argument('-o', '--object ', dest='obj_id',
    #                    required=False,default='box',
    #                    choices = ['box', 'head_phones', 'engine_main', 'dual_sphere','6', 'box'],
    #                    help='Object names')
    parser.add_argument('-b', '--buffer_size', dest='buffer_size',  
                        type=int, required=False, default=3,
                        help='Frame buffer for smoothing.')
    parser.add_argument('-n', '--n_triangles', dest='n_triangles',
                        type=int, required=False, default=2000,
                        help='Number of triangles for cloud/mesh.')
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                        required=False, default='maskrcnn',
                        choices = ['chromakey','bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'],
                        help="""Method of segmentation.
                        contour: OpenCV based edge detection ...,
                        TODO:
                        """)
    ### Python < 3.9 TODO: Set this up.
    #parser.add_argument('--feature', action='store_true', dest='render_mesh')
    #parser.add_argument('--no-feature', dest='render_mesh', action='store_false')
    #parser.set_defaults(render_mesh=True)
    ### Python >= 3.9
    parser.add_argument('-rm', '--render-mesh', dest='render_mesh', action=argparse.BooleanOptionalAction)
    parser.add_argument('-icp', dest='icp', action=argparse.BooleanOptionalAction)

    
    args = parser.parse_args()

    main(args)
