import os
import sys
import json
from pathlib import Path
from typing import Any, TypedDict
import cv2
import torch
import numpy as np
from numpy.typing import NDArray

from ipdb import iex

from os.path import join as pjoin

base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from lib import chromakey

def load(cfg, device, **kwargs):
    """
    TODO: ADD TYPES
    Segmentator needs to return:
        TODO

    ## Returns
    mask_gpu: torch.tensor on gpu
    mask: numpy.array
    scores: confidence scores. None if not provided.
    """
    img_size=(cfg.RENDER_HEIGHT, cfg.RENDER_WIDTH)

    segmentator_setter = chromakey.Segmentator(
	        img_size=img_size
	        )
	
	#if object_name == 'test_gear':
	#    tola = 1.0; tolb = 1.53
	#elif object_name == 'test_clipper':
	#    tola = 0.66; tolb = 1.05
	#else:
	#    raise ValueError("What scene?")
    init_tola: int = 496
    init_tolb: int = 601
    init_Cb_key: int = 96
    init_Cr_key: int = 63
    pre_kwargs = {
	        'init_tola' : init_tola/10,
	        'init_tolb' : init_tolb/10,
	        'init_Cb_key' : init_Cb_key*1.0,
	        'init_Cr_key' : init_Cr_key*1.0
	        }

    mask_gpu = torch.zeros((1, img_size[0], img_size[1]), device=device, dtype=bool)
    mask: NDArray[np.bool_] = np.zeros((1, img_size[0], img_size[1]), dtype=np.bool_)

    #if kwargs['trackbars_on']:
    if True:
        window_name = 'mask'


        cv2.namedWindow(window_name)
        tola_max = 1000
        tolb_max = 1000
        Cb_key_max = 255
        Cb_key_max = 255
        def nothing(value) -> None:
            pass

        cv2.createTrackbar('tola', window_name, init_tola, tola_max, nothing)
        cv2.createTrackbar('tolb', window_name, init_tolb, tolb_max, nothing)
        cv2.createTrackbar('Cb_key', window_name, init_Cb_key, Cb_key_max, nothing)
        cv2.createTrackbar('Cr_key', window_name, init_Cr_key, Cb_key_max, nothing)

        filter_ = segmentator_setter.get_filter(colorspace='YCrCb')

        def segmentator(image) -> tuple[NDArray[np.bool_], Any, None]:

            tola = cv2.getTrackbarPos('tola',window_name)/10
            tolb = cv2.getTrackbarPos('tolb',window_name)/10
            Cr_key = cv2.getTrackbarPos('Cr_key',window_name)
            Cb_key = cv2.getTrackbarPos('Cb_key',window_name)
            
            kwargs: dict[str, Any] = {"tola": tola, "tolb": tolb, "Cr_key": Cr_key, "Cb_key": Cb_key}
        	
            mask[:] = filter_(image=image, **kwargs)
            mask_gpu[:] = torch.from_numpy(mask)
            return mask, mask_gpu, None
    else:
	    ### Load reference frames for segmentator
        filter_ = segmentator_setter.get_filter(colorspace='YCrCb', **pre_kwargs)
    	
    	
        def segmentator(image) -> tuple[NDArray[np.bool_], Any, None]:
        	
            mask[:] = filter_(image=image)
            mask_gpu[:] = torch.from_numpy(mask)
            return mask, mask_gpu, None
    
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
