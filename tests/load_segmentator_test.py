import os
from os.path import join as pjoin
from time import perf_counter
import sys
import warnings
import json
from pathlib import Path
import numpy as np

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

models = np.array(['bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'], dtype=object)
models_ok = np.zeros(len(models), dtype=bool)
for m_idx, model in enumerate(models):
    print("Testing:", model)
    print("#"*30)
    if load_segmentation_model.load(model=model, cfg=cfg, device='cuda'):
        models_ok[m_idx] = True
    else:
        models_ok[m_idx] = False

print(f"{models_ok.sum()}/{models_ok.size} models loading")
print(models[~models_ok], "failed")
