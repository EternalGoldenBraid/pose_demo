import os
import cv2
import sys
import json
import math
import time
import torch
import warnings
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import pyrealsense2 as rs
from scipy import stats

from ipdb import iex
from matplotlib import pyplot as plt
#from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from os.path import join as pjoin
from bop_toolkit_lib import inout
warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from lib import (rendering, network,
        agnostic_segmentation)

from lib.render_cloud import load_cloud, render_cloud

from dataset import LineMOD_Dataset, prototype_Dataset
from evaluation import utils
from evaluation import config as cfg

DEVICE = torch.device('cuda')

class PoseEstimator:
    def __init__(self, cfg, cam_K, obj_codebook, model_net, device='cpu'):
        self.obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)
        self.cam_K = cam_K
        self.obj_codebook = obj_codebook
        self.device = device
        self.model_net = model_net
        self.cfg = cfg

    def estimate_pose(self, obj_depth, obj_mask):

        #tar_obj_depth = (view_depth * obj_mask).squeeze()
        pose_ret = utils.OVE6D_mask_full_pose(
            model_func=self.model_net, 
            #obj_depth=tar_obj_depth[None,:,:],
            obj_depth=obj_depth[None,:,:],
            obj_mask=obj_mask[None,:,:],
            obj_codebook=self.obj_codebook,
            cam_K=self.cam_K,
            config=self.cfg,
            obj_renderer=self.obj_renderer,
            device=DEVICE)

        return pose_ret['raw_R'], pose_ret['raw_t']

    def __del__(self):
        del self.obj_renderer

def load_data(datapath=None):

    ### DATA
    eval_dataset = prototype_Dataset.Dataset(datapath / 'huawei_box')
    #eval_dataset = LineMOD_Dataset.Dataset(datapath / 'lm')
    
    return eval_dataset

def load_codebooks(model_net, eval_dataset):
    ### CODEBOOKS
    cfg.VIEWBOOK_BATCHSIZE = 200 # reduce this if out of GPU memory, 
    cfg.RENDER_NUM_VIEWS = 4000 # reduce this if out of GPU memory, 
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

    
def main(obj_id, clipping_distance_in_meters):

    #cfg.RENDER_WIDTH = eval_dataset.cam_width    # the width of rendered images
    #cfg.RENDER_HEIGHT = eval_dataset.cam_height  # the height of rendered images
    cfg.RENDER_WIDTH = 640    # the width of rendered images
    cfg.RENDER_HEIGHT = 480  # the height of rendered images
    cfg.DATASET_NAME = 'huawei_box'        # dataset name
    cfg.HEMI_ONLY = False
    cfg.VIEWBOOK_BATCHSIZE = 200 # reduce this if out of GPU memory, 
    cfg.RENDER_NUM_VIEWS = 4000 # reduce this if out of GPU memory,
    cfg.VP_NUM_TOPK = 50   # the retrieval number of viewpoint 
    cfg.RANK_NUM_TOPK = 5


    dataroot = Path(cfg.DATA_PATH)
    dataset = prototype_Dataset.Dataset(data_dir=dataroot/ 'huawei_box', cfg=cfg)
    #dataset = load_data(datapath=dataroot, cfg)

    model_path = 'checkpoints'
    model_file = "OVE6D_pose_model.pth"
    model_net_ove6d = load_model_ove6d(model_path=model_path, model_file=model_file)

    obj_codebook = load_codebooks(model_net=model_net_ove6d, eval_dataset=dataset)[obj_id]

    segmentator = agnostic_segmentation.load_model('checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth')

    data_path = dataroot/'huawei_box'
    #cloud *= 10 # TODO SCALE IS OFF
    dataset.cam_K.to(DEVICE)
    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, 
            obj_codebook=obj_codebook, 
            model_net=model_net_ove6d,
            device=DEVICE)

    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    
    framerate = 30
    config.enable_stream(rs.stream.depth, cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT, rs.format.z16, framerate)
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance = clipping_distance_in_meters / depth_scale
    grey_color = 0
    
    ### QUERY FOR FRAMES FOR INTRINSICS. 
    ### TODO: Query sensor conf directly instead of loading a frame first.
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Get frames
    depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    i = depth_frame.profile.as_video_stream_profile().intrinsics
    #color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics ### These should be same.
    K = torch.tensor([  [i.fx,  0,  i.ppx],
                        [0, i.fy, i.ppy],
                        [0, 0,  1]], device='cpu') 
    dataset.cam_K = K
    
    # Validate that both frames are valid
    if not depth_frame:
        print("Couldn't load camera intrinsics")
        return


    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
    
            # Get aligned frames
            depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    
            # Validate that both frames are valid
            if not depth_frame:
                continue
    
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
            #depth_image = np.asanyarray(aligned_depth_frame.get_data())*depth_scale/100 ### depth_scale --> cm --> (100) meters?
            #depth_image = np.asanyarray(depth_frame.get_data())*depth_scale ### depth_scale --> cm --> (100) meters?
            #depth_image /= 100
    
            # Remove background - Set pixels further than clipping_distance to grey
            #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            mask = np.where((depth_image > clipping_distance) | (depth_image <= 0),
                    grey_color, depth_image)

            #plt.hist(depth_image)
            #plt.show(block=False)
            #import pdb; pdb.set_trace()
    
            # Render images:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
            mask_colormap = cv2.applyColorMap(cv2.convertScaleAbs(mask, alpha=0.03), cv2.COLORMAP_JET) 
            #masked_img = (color_image*0.9 + mask*0.5*255).astype(np.uint8)

            images = np.hstack((depth_colormap, mask_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Superimpose rotated pointcloud onto video.')
    parser.add_argument('--obj_id',
                        type=int,
                        required=False,
                        help='Object index: {box, basket, headphones}',
                        default=1)
    parser.add_argument('-d',
                        type=float,
                        required=False,
                        help='Clipping distance in meters',
                        default=0.5)
    
    args = parser.parse_args()
    
    main(obj_id=args.obj_id, clipping_distance_in_meters = args.d) #1 meter
