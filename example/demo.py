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

from utility import timeit
from lib import (rendering, network,
        agnostic_segmentation)

from lib.render_cloud import load_cloud, render_cloud

from dataset import LineMOD_Dataset, demo_dataset
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
            obj_depth=obj_depth[None,:,:], # 1*h*w
            obj_mask=obj_mask[None,:,:], # 1*h*w
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

def init_cam():
    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
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
    config.enable_stream(rs.stream.color, cfg.RENDER_WIDTH, cfg.RENDER_HEIGHT, rs.format.bgr8, framerate)
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    #align_to = rs.stream.depth
    align = rs.align(align_to)

    return pipeline, config, align

def get_scale_intrinsics(pipeline,config, align, bgsubtract=True):

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    

    ### QUERY FOR FRAMES FOR INTRINSICS. 
    ### TODO: Query sensor conf directly instead of loading a frame first.
    # Get frameset of color and depth
    if bgsubtract:
        input("Capture background. Enter")
    frames = pipeline.wait_for_frames()
    
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        print("Couldn't load camera intrinsics")
        return

    i = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    #color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics ### These should be same.
    K = torch.tensor([  [i.fx,  0,  i.ppx],
                        [0, i.fy, i.ppy],
                        [0, 0,  1]], device='cpu') 

    print("Camera intrinsics:", K)

    return depth_scale, K, aligned_depth_frame
    
def main(obj_id):

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

    timeit.log("Realsense initialization.")
    pipeline, rs_config, align = init_cam()

    #segment_method = 'bgsubtract'
    #segment_method = 'maskrcnn'
    segment_method = 'bgsMOG2'
    if segment_method == 'bgsubtract':
        bgsubtract=True
    elif segment_method == 'bgsMOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
        bgsubtract=False
    else: bgsubtract=False

    depth_scale, cam_K, background = get_scale_intrinsics(pipeline=pipeline, 
            config=rs_config, align=align, bgsubtract=bgsubtract)

    timeit.endlog()
    timeit.log("Loading data.")
    dataroot = Path(cfg.DATA_PATH)
    dataset = demo_dataset.Dataset(data_dir=dataroot/ 'huawei_box', cfg=cfg,
                cam_K=cam_K, cam_height=cfg.RENDER_HEIGHT, cam_width=cfg.RENDER_WIDTH)


    model_path = 'checkpoints'
    model_file = "OVE6D_pose_model.pth"
    model_net_ove6d = load_model_ove6d(model_path=model_path, model_file=model_file)


    obj_codebook = load_codebooks(model_net=model_net_ove6d, eval_dataset=dataset)[obj_id]

    timeit.endlog()

    segmentator = agnostic_segmentation.load_model('checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth')

    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, 
            obj_codebook=obj_codebook, 
            model_net=model_net_ove6d,
            device=DEVICE)


    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
    
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
    
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
    
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
    
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
    
            timeit.log("Pose estimation.")
            # Segment
            if segment_method == 'bgsubtract':
                mask = depth_image - np.asanyarray(background.get_data())
                plt.imshow(mask/mask.max())
                plt.show()
                import pdb; pdb.set_trace()
            elif segment_method == 'maskrcnn':
                masks = segmentator(color_image)['instances'].get('pred_masks')[0]
            elif segment_method == 'bgsMOG2':
                mask = backSub.apply(color_image)
                #import pdb; pdb.set_trace()
                cv2.rectangle(color_image, (10, 2), (100,20), (255,255,255), -1)

            else:
                print("No segmentator")
                return -1
    
            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
            if mask.all() == None:
                #mask = mask.cpu().numpy().astype(np.uint8)
                mask = mask.numpy().astype(np.uint8)
                masked_img = (color_image*0.9 + mask[...,None]*0.5*255).astype(np.uint8)

                obj_depth=torch.tensor((depth_image*mask*depth_scale).astype(np.float32)).squeeze()
                R, t = pose_estimator.estimate_pose(
                            obj_depth=obj_depth,
                            obj_mask=masks)

                #timeit.endlog()
                #timeit.log("Rendering.")
                dataset.render_cloud(obj_id=obj_id, R=R, t=t, image=color_image)
                #images = np.hstack((color_image, masked_image))
                #import pdb; pdb.set_trace()
                #images = np.hstack((color_image, color_image*mask[...,None], depth_colormap*mask[...,None]))
                #color_image = color_image.astype(float)
                #color_image = color_image/color_image.max()
                #images = np.hstack((
                #    color_image,
                #    color_image*mask[...,None],
                #    depth_image*mask[...,None]
                #    ))
                images = color_image

                timeit.endlog()

                #est_depth = dataset.render_depth(obj_id=obj_id, R=R, t=t, mesh=obj_codebook['obj_mesh'])
                #est_depth /= est_depth.max()
                #images = np.hstack((color_image, np.dstack((est_depth,est_depth,est_depth))))

                #images = np.hstack((color_image, mask*255))
                #images = color_image
            else:
                #images = np.hstack((color_image, depth_colormap))
                images = np.hstack((color_image, color_image, depth_colormap))
            import pdb; pdb.set_trace()

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
    
    args = parser.parse_args()
    
    main(obj_id=args.obj_id)
