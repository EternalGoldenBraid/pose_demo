import os
import time
import glob
import math
import torch
import numpy as np
from evaluation import config as cfg
from lib import preprocess, rendering

from ipdb import iex



intrinsics = torch.tensor([[603.8275,   0.0000, 329.7891],
        [  0.0000, 602.4884, 239.1700],
        [  0.0000,   0.0000,   1.0000]])

self.obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)
