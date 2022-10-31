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

#base_path = os.path.dirname(os.path.abspath("."))
#sys.path.append(base_path)

from lib import chromakey
#from ove6d.lib import chromakey

def load(cfg, device, **kwargs)-> tuple[NDArray, torch.tensor, None]:
    """
    Helper function that loads a chromakey segmentation.
    
    Parameters
    ----------
    
    :param string device: cuda or cpu.
    :param cfg: a .py file defining various configuration parameters as variables.

    Returns
    -------
    
    :return torch.tensor mask_gpu: torch.tensor on gpu of shape [N_objects, img_height, img_width]
    :return numpy.array mask: np.array on cpu [N_objects, img_height, img_width]
    :return scores: Confidence scores if provided my segmentation model.
    
    Notes
    -----
    
    Place the segmentation model that defines that implements the described return types to ./lib
    from which it will be imported.
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
    init_tola: int = 511
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
        filter_ = segmentator_setter.get_filter(colorspace='YCrCb', **pre_kwargs)
    	
    	
        def segmentator(image) -> tuple[NDArray[np.bool_], Any, None]:
        	
            mask[:] = filter_(image=image)
            mask_gpu[:] = torch.from_numpy(mask)
            return mask, mask_gpu, None
    
    return segmentator
