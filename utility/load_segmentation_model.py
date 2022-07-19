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

from ipdb import iex
from matplotlib import pyplot as plt
#from detectron2 import model_zoo
from detectron2.projects import point_rend

from os.path import join as pjoin
warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from lib import detectron_segmentation, contour_segmentation

def load(model, cfg, device):
    """
    TODO: ADD TYPES
    Segmentator needs to return:
    mask_gpu: torch.tensor on gpu
    mask_gpu: numpy.array
    """
    if model == 'bgs_hsv':
        def segmentator_(image): 
            #im_hsv = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)
            im_hsv = cv2.cvtColor(
                    1./255*image.astype(np.float32), code=cv2.COLOR_BGR2HSV_FULL)
            sub = im_hsv.copy()
            low = 140; up = 220
            mask = cv2.inRange(sub, np.array([low,0,0]), np.array([up,1,1]))

            center = (cfg.RENDER_WIDTH//2, cfg.RENDER_HEIGHT//2)
            radius = cfg.RENDER_WIDTH // 4
            color = 255
            thickness = -1
            line_type = 8
            #circle = cv2.circle(mask, center, radius, color, thickness, line_type)
            circle = cv2.circle(np.zeros_like(mask, dtype=np.uint8), center, radius, color, thickness, line_type)//255
            mask = mask//255; mask=mask.astype(bool)
            mask = ~mask
            mask = circle*mask

            return mask.astype(np.uint8), torch.tensor(mask, device=device)

        segmentator = segmentator_
    elif model == 'bgsKNN':
        def segmentator_(image):
            mask = cv2.createBackgroundSubtractorKNN().apply(image)
            return mask, torch.tensor(mask, device=device)
        segmentator = segmentator_
    elif model == 'bgsMOG2':
        def segmentator_(image):
            mask = cv2.createBackgroundSubtractorMOG2().apply(image)
            return mask, torch.tensor(mask, device=device)
        segmentator = segmentator_
    elif model == 'contour':
        #segmentator = lambda image: contour_segmentation.ContourSegmentator().get_mask(image), None
        def segmentator_(image):
            mask = contour_segmentation.ContourSegmentator().get_mask(image)
            return mask, torch.tensor(mask, device=device)
        segmentator = segmentator_
    elif model == 'maskrcnn':
        none_array = np.array(())
        model_seg = detectron_segmentation.load_model_image_agnostic(base_path+'/checkpoints/FAT_trained_Ml2R_bin_fine_tuned.pth')
        def segmentator_(image, model=model_seg):
            mask_gpu = model(image)['instances'].get('pred_masks')
            if mask_gpu.numel() == 0:
                return none_array, None
            mask_cpu = mask_gpu[0].to(
                    non_blocking=True, copy=True, device='cpu').numpy().astype(int).astype(np.uint8)
            return mask_cpu, mask_gpu[0]
        segmentator = segmentator_
    elif model == 'point_rend':
        none_array = np.array(())
        model_seg = detectron_segmentation.load_model_point_rend(base_path+'/checkpoints/model_final_ba17b9_pointrend.pkl',
                config_yaml=base_path+'/configs/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
                confidence=0.7, base_path=base_path)
        def segmentator_(image, model=model_seg):
            mask_gpu = model(image)['instances'].get('pred_masks')
            if mask_gpu.numel() == 0:
                return none_array, None
            mask_cpu = mask_gpu[0].to(
                    non_blocking=True, copy=True, device='cpu').numpy().astype(int).astype(np.uint8)
            return mask_cpu, mask_gpu[0]
        segmentator = segmentator_
    else: 
        print("Invalid segmentation option:", model)
        return None

    return segmentator

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
                        choices = ['bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn',],
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