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

from utility import timeit, load_segmentation_model
from lib import (rendering, network, triangulate)

from lib.render_cloud import load_cloud, render_cloud

from dataset import LineMOD_Dataset, demo_dataset
from evaluation import utils
from evaluation import config as cfg

DEVICE = torch.device('cuda')

class Camera()

    def __init__(self, size=(640, 480), framerate=60):
        self.pipeline = None
        self.config = None
        self.align = None

        # Create a pipeline
        self.pipeline = rs.pipeline()
        
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()
        
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
        
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, framerate)
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, framerate)
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        #align_to = rs.stream.depth
        self.align = rs.align(align_to)
    

        ### Get scale intrinsics
        ### TODO: Get directly from device without capturing frame.

        # Start streaming
        profile = pipeline.start(self.config)
    
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)
    
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
    
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            print("Couldn't load camera frame.")
            return
    
        i = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        #color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics ### These should be same.
        self.cam_K = torch.tensor([  [i.fx,  0,  i.ppx],
                            [0, i.fy, i.ppy],
                            [0, 0,  1]], device='cpu') 
    
        print("Camera intrinsics:", K)
    
        return depth_scale, K, np.asanyarray(color_frame.get_data())

        def get_frame(self):
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return None
            
            return aligned_depth_frame, color_frame

        def get_image(self):
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Get aligned frames
            # aligned_depth_frame is a 640x480 depth image
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return None

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return depth_image, color_image

    
if __name__=="__main__":

    cam = Camera(size=(640, 480), framerate=60)
    depth_scale, cam_K =  cam.depth_scale, cam.cam_K

    try:
        while True:

            depth_image, color_image

            #timeit.log("Pose estimation.")
            mask, mask_gpu = segmentator(color_image)

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
            if mask.size != 0:

                ### TODO: Can we get depth_image dircetly to gpu from sensor and skip gpu --> cpu with <mask>
                R, t = pose_estimator.estimate_pose(obj_mask=mask_gpu,
                            obj_depth=torch.tensor( (depth_image*mask*depth_scale).astype(np.float32)).squeeze())

                #if count % args.buffer_size == 0:
                #    count = -1
                #else:
                #    continue

                #timeit.endlog()
                #timeit.log("Rendering.")

                color_image, done = dataset.render_cloud(obj_id=obj_id, 
                        R=R.numpy().astype(np.float32), 
                        t=t.numpy().astype(np.float32),
                        image=color_image)

                #color_image, done = dataset.render_mesh(obj_id=obj_id, 
                #        R=R.numpy().astype(np.float32), 
                #        t=t.numpy().astype(np.float32),
                #        image=color_image)

                # Point cloud matching failed.
                if not done: continue

                #import pdb; pdb.set_trace()
                if args.render_mesh:
                    pass

                images = np.hstack([ 
                    color_image, color_image*(mask[...,None]) ])
                #timeit.endlog()
            else:
                images = np.hstack((color_image, color_image))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()

