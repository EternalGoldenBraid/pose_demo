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

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from utility import timeit, load_segmentation_model, cam_control
from lib import (rendering, network, triangulate)

from lib.render_cloud import load_cloud, render_cloud

from dataset import LineMOD_Dataset, demo_dataset
from evaluation import utils
from evaluation import config as cfg

#DEVICE = torch.device('cuda')
#DEVICE = torch.device('cpu')
#DEVICE = 'cpu'
DEVICE = 'cuda'

class PoseEstimator:
    def __init__(self, cfg, cam_K, obj_codebook, model_net, device='cpu'):
        self.obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)
        self.cam_K = cam_K
        self.obj_codebook = obj_codebook
        self.device = device
        self.model_net = model_net
        self.cfg = cfg

    def estimate_pose(self, obj_depth, obj_mask):
        """
        B*height*width
        """

        #tar_obj_depth = (view_depth * obj_mask).squeeze()
        pose_ret = utils.OVE6D_mask_full_pose(
            model_func=self.model_net, 
            #obj_depth=tar_obj_depth[None,:,:],
            obj_depth=obj_depth, # 1*h*w
            obj_mask=obj_mask, # 1*h*w
            obj_codebook=self.obj_codebook,
            cam_K=self.cam_K,
            config=self.cfg,
            obj_renderer=self.obj_renderer,
            device=DEVICE)

        return pose_ret['raw_R'][None,...], pose_ret['raw_t'][None,...]

    def estimate_poses(self, obj_depths, obj_masks, scores):

        #tar_obj_depth = (view_depth * obj_mask).squeeze()
        pose_ret = utils.OVE6D_rcnn_full_pose(
            model_func=self.model_net, 
            #obj_depth=tar_obj_depth[None,:,:],
            obj_depths=obj_depths, # N*h*w
            obj_masks=obj_masks, # N*h*w
            obj_rcnn_scores=scores,
            obj_codebook=self.obj_codebook,
            cam_K=self.cam_K,
            config=self.cfg,
            obj_renderer=self.obj_renderer,
            device=DEVICE,
            return_rcnn_idx=False # TODO role?
            )

        # TODO Add multiobject support.
        return pose_ret['raw_R'][None,...], pose_ret['raw_t'][None,...]

    def __del__(self):
        del self.obj_renderer

def load_codebooks(model_net, eval_dataset):
    codebook_saving_dir = pjoin(base_path,'evaluation/object_codebooks',
                                cfg.DATASET_NAME, 
                                'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR), 
                                'views_{}'.format(str(cfg.RENDER_NUM_VIEWS)))
    
    
    
    object_codebooks = utils.OVE6D_codebook_generation(codebook_dir=codebook_saving_dir, 
                                                        model_func=model_net,
                                                        dataset=eval_dataset, 
                                                        config=cfg, 
                                                        device=DEVICE)
    print('Object codebooks have been loaded!')
    print(object_codebooks.keys())

    return object_codebooks

def load_model_ove6d(model_path, model_file=None):

    assert type(model_file) == str

    ### MODEL
    #ckpt_file = pjoin(base_path, 'checkpoints', "OVE6D_pose_model.pth")
    ckpt_file = pjoin(base_path, model_path, model_file)

    model_net = network.OVE6D().to(DEVICE)
    model_net.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
    model_net.eval()
    print('OVE6D has been loaded!')
    return model_net

def main(args):

    #cfg.RENDER_WIDTH = eval_dataset.cam_width    # the width of rendered images
    #cfg.RENDER_HEIGHT = eval_dataset.cam_height  # the height of rendered images
    cfg.DATASET_NAME = 'huawei_box'        # dataset name
    cfg.USE_ICP = args.icp

   #timeit.log("Realsense initialization.")
   # TODO change to row_major for numpy...? What's torch
    cam = cam_control.Camera(size=(cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT), framerate=60)
    depth_scale, cam_K =  cam.depth_scale, cam.cam_K
    cam_K_np = cam_K.numpy()
   #timeit.endlog()

    ### Segmentation
    segmentator = load_segmentation_model.load(model=args.segment_method, cfg=cfg, device=DEVICE)

   #timeit.log("Loading data.")
    #dataroot = Path(os.path.dirname(__file__)).parent/Path(cfg.DATA_PATH)
    dataroot = Path(os.path.realpath(__file__)).parent.parent/Path(cfg.DATA_PATH)

    dataset = demo_dataset.Dataset(data_dir=dataroot/ 'huawei_box', cfg=cfg,
                cam_K=cam_K, cam_height=cfg.RENDER_HEIGHT, cam_width=cfg.RENDER_WIDTH,
                n_points=args.n_points)

    model_path = 'checkpoints'
    model_file = "OVE6D_pose_model.pth"
    model_net_ove6d = load_model_ove6d(model_path=model_path, model_file=model_file)

    obj_id: int = args.obj_id
    obj_codebook = load_codebooks(model_net=model_net_ove6d, eval_dataset=dataset)[obj_id]

    #timeit.endlog()
    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, 
            obj_codebook=obj_codebook, 
            model_net=model_net_ove6d,
            device=DEVICE)

    dataset.object_renderer = pose_estimator.obj_renderer

    # Streaming loop
    mod_count: int = 0
    buffer_size: int = args.buffer_size
    frame_buffer = np.empty([buffer_size, cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH])

    try:
        while True:

            fps_start = perf_counter()
            
            depth_image, color_image = cam.get_image()

            masks, masks_gpu, scores = segmentator(color_image)

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
            if masks.size != 0:

                #print((depth_image[mask]*depth_scale).mean())
                #import pdb; pdb.set_trace()
                ### TODO: Can we get depth_image dircetly to gpu from sensor and skip gpu --> cpu with <mask>
                R, t = pose_estimator.estimate_pose(obj_mask=masks_gpu[0][None,...],
                            obj_depth=torch.tensor(
                                (depth_image*masks[0]*depth_scale).astype(np.float32)).squeeze()[None,...]
                            )

                ### TODO Multi object support.
                #obj_depths = torch.tensor([(depth_image*mask*depth_scale).astype(np.float32) for mask in masks])
                #R, t = pose_estimator.estimate_poses(obj_masks=masks_gpu, scores=scores,
                #            obj_depths=obj_depths.squeeze())

                #if count % args.buffer_size == 0:
                #    R = R/args.buffer_size; t = t/args.buffer_size
                #    count = 0
                #else:
                #    R = R+R_old; t = t+t_old
                #    continue

                #timeit.endlog()
                #timeit.log("Rendering.")

                for transform_idx in range(R.shape[0]):
                    color_image, done = dataset.render_cloud(obj_id=obj_id, 
                            R=R[transform_idx].numpy().astype(np.float32), 
                            t=t[transform_idx].numpy()[...,None].astype(np.float32),
                            image=color_image)

                    #color_image, done = dataset.render_mesh(obj_id=obj_id, 
                    #        R=R[transform_idx].numpy().astype(np.float32), 
                    #        t=t[transform_idx].numpy()[...,None].astype(np.float32),
                    #        image=color_image)

                #import pdb; pdb.set_trace()
                if args.render_mesh:
                    pass

                images = np.hstack([ 
                    color_image, 
                    color_image*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                    ])
                #timeit.endlog()
            else:
                images = np.hstack((color_image, color_image))

            
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
    
    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')
    parser.add_argument('-o', '--obj_id', dest='obj_id',
                        type=int, required=False,default=1,
                        help='Object index: {box, basket, headphones}')
    parser.add_argument('-b', '--buffer_size', dest='buffer_size',  
                        type=int, required=False, default=3,
                        help='Frame buffer for smoothing.')
    parser.add_argument('-n', '--n_points', dest='n_points',
                        type=int, required=False, default=2000,
                        help='Number of points for cloud/mesh.')
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                        required=False, default='maskrcnn',
                        choices = ['bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'],
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
