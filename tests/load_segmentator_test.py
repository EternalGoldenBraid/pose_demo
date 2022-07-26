import os
from os.path import join as pjoin
from time import perf_counter
import sys
import warnings
import json
from pathlib import Path

import numpy as np
import cv2
from ipdb import iex
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects import point_rend

warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from utility import timeit, load_segmentation_model
from lib import detectron_segmentation, contour_segmentation
from evaluation import config as cfg


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
cfg.USE_ICP = False

models = np.array(
        #['bgs', 'bgs_hsv', 'bgsmog2', 'bgsknn', 'contour', 'maskrcnn', 'point_rend'],
        ['bgs', 'bgs_hsv', 'bgsmog2', 'bgsknn', 'contour', 'maskrcnn'],
        dtype=object)
models_load_ok = np.zeros(len(models), dtype=bool)
models_segment_ok = np.zeros(len(models), dtype=bool)
for m_idx, model_name in enumerate(models):
    print("Testing:", model_name)
    print("#"*30)
    model = load_segmentation_model.load(model=model_name, cfg=cfg, device='cuda')

    if model:
        models_load_ok[m_idx] = True

        ## TODO: Add True negative test!
        mask, mask_gpu, scores = model(cv2.imread('input.jpg'))
        if mask.size != 0:
            models_segment_ok[m_idx] = True
        else:
            models_segment_ok[m_idx] = False
    else:
        models_load_ok[m_idx] = False
        models_segment_ok[m_idx] = False

print(f"{models_load_ok.sum()}/{models_load_ok.size} models loading")
print(models[~models_load_ok], "failed")
print("#"*30)
print(f"{models_segment_ok.sum()}/{models_segment_ok.size} models segmenting")
print(models[~models_segment_ok], "failed")
