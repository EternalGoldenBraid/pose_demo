import os
import threading
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Generator, List
from numpy.random import default_rng

from sklearn import linear_model

import h5py
from time import perf_counter

def load_recorded_frames(filename, object_name, data_path=None):
    if data_path==None:
        data_path = Path('data','recordings',filename)
    else:
        raise NotImplementedError("data_path unsupported")

    with h5py.File(data_path, "r") as f:
        data = f[f'single_object/{object_name}']
        colors: NDArray = data['color_frames'][:].astype(np.uint8)
        depths: NDArray = data['depth_frames'][:]
        depth_scale: float = f['meta/depth_scale'][:]
        cam_K: NDArray = f['meta/camera_intrinsic'][:]
        fps: int = f['meta/framerate'][:]

    return colors, depths, depth_scale, cam_K, int(fps)

def draw_contours(image_size, mask: NDArray[np.uint8], color=(255,0,0), min_area=300, max_area=10000):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = np.array([cv2.contourArea(cnt) for cnt in contours])
    contours = [contours[i] for i in range(len(contours)) if area[i] > min_area and area[i] < max_area]

    new_mask = np.zeros([len(contours), img_size[0], img_size[1], 1], dtype=np.uint8)

    for c_idx in range(len(contours)):
        #new_mask[:,:,c_idx] = cv2.drawContours(mask, contours, c_idx, (1), -1)
        #breakpoint()
        #new_mask[c_idx] = cv2.fillPoly(new_mask[c_idx], pts =[contours[c_idx]], color=255)
        cv2.fillPoly(new_mask[c_idx], pts =[contours[c_idx]], color=255)
    return new_mask.squeeze(axis=-1).astype(bool)


def filter_out_of_plane_depth_map(
        estimator, hist_mask, depth_map,
        color_image: NDArray, grid: NDArray,
        outlier_color = (176,196,222), inlier_color = (255,105,180), max_residual=0.01):

    estimator.fit(
            grid[hist_mask][:,:2],
            depth_map[hist_mask]
            )

    w = np.r_[estimator.estimator_.coef_, estimator.estimator_.intercept_]

    # TODO Add sparsity here?
    plane = (grid.reshape(-1,3) @ w).reshape(depth_map.shape)
    plane_map_mask: NDArray = np.abs(plane - depth_map) <= max_residual

    return plane_map_mask

def get_hist_bgr(img: NDArray[np.uint8], hist_size: int = 256, hist_range: List[int] = [0,255]):
    """
    returns in rgb.
    """
    bgr_planes: tuple[NDArray, NDArray, NDArray] = cv2.split(img)
    b_hist = np.array(cv2.calcHist(bgr_planes, [0], None, [hist_size], hist_range)).squeeze()
    g_hist = np.array(cv2.calcHist(bgr_planes, [1], None, [hist_size], hist_range)).squeeze()
    r_hist = np.array(cv2.calcHist(bgr_planes, [2], None, [hist_size], hist_range)).squeeze()
    
    return np.array([r_hist, g_hist, b_hist])

def filter_histogram(hist_means: NDArray[np.int_],
                     image: NDArray[np.uint8],
                     eps: int = 10, ) -> NDArray:
    """
    Return segmented image and mask.

    Attributes:
    :param hist_means: RGB histogram averaged over one or more images.
    :type hist_means: NDArray[3, n_bins]
    """

    # Peaks of histograms per channel
    peaks = hist_means.argmax(axis=1)

    hist_mask = (image > (peaks - eps)) * (image < (peaks + eps))
    hist_mask = hist_mask[:,:,0]*hist_mask[:,:,1]*hist_mask[:,:,2]

    return hist_mask

def create_bg_filter(frames: NDArray[np.uint8], hist_size = 256, hist_range = [0,255]):

    bg_hists = np.array([get_hist_bgr(img=image, hist_size=hist_size, hist_range=hist_range) for image in frames])
    hist_means_ref: NDArray = np.mean(bg_hists, axis=0) # 3 x 

    def new_filter_histogram(image, hist_means=None , eps=10):
        return filter_histogram(hist_means=hist_means_ref, image=image, eps=eps)

    return new_filter_histogram

class Segmentator:

    def __init__(self, 
                 max_n_ref_frames: int =8,
                 img_size: tuple[int, int] = (480, 640),
                 seed=1337) -> None:
        self.rng = default_rng(seed)
        self.max_n_ref_frames: int = max_n_ref_frames
        self.img_size: tuple[int, int] = img_size
        self.ref_frames: NDArray = np.empty((max_n_ref_frames, img_size[0], img_size[1], 3), dtype=np.uint8)
        self.ref_peaks: NDArray[np.uint8] = np.zeros(3, dtype=np.uint8)

    def get_reference_frames(self, cam,) -> None:
        pass

    def set_reference_frames(self, frames: NDArray[np.uint8],
                hist_size: int = 256,
                hist_range: List[int] = [0,255]) -> 'rgbd_2_mask':
        """
        Compute histogram masking params from reference images.
        """

        idxs = self.rng.integers(low=0, high=frames.shape[0], size=self.max_n_ref_frames)
        new_filter_histogram = create_bg_filter(frames=frames[idxs], hist_size = 256, hist_range = [0,255])

        estimator = linear_model.RANSACRegressor(max_trials=1000)
        xx, yy = np.meshgrid(np.arange(self.img_size[1], dtype=int), np.arange(self.img_size[0], dtype=int))
        grid: NDArray = np.stack( [xx, yy, np.ones_like(xx, dtype=int)], axis=-1)



        if plane_fit_to == 'pointcloud':
            def rgbd_2_mask(
                    image: NDArray[np.uint8], depth: NDArray[np.uint8], 
                    eps: int = 9, 
                    max_residual=0.01,
                    min_area=300, max_area=10000
                    ):
                """
                TODO
                :param eps: Total displacement of pixel values from the most frequent pixel value
                    which will be considered as background.  peak-eps <= background_pixels <= peak+eps
                :param max_residual: Accepted projection error for plane projection inliers. 
                    Points within this distance will be considered part of the plane.
                """
                hist_mask = new_filter_histogram(image=image, eps=eps)

                plane_mask = filter_out_of_plane_depth_map(
                        estimator=estimator, hist_mask=hist_mask,
                        depth_map=depth,
                        color_image=image, grid=grid,
                        max_residual=max_residual)

                (plane_mask, plane_hat_pcl,  plane_map,
                result_plane, full_plane) = filter_out_of_plane_pcl(estimator=estimator, hist_mask=hist_mask,
                        depth=depth_image.copy()*depth_scale,
                        bg_canvas=bg_canvas,
                        color_image=color_image.copy(), cam_K=cam_K)

                final_mask = draw_contours(self.img_size, mask=plane_mask.astype(np.uint8), 
                                           color=(255,0,0), min_area=min_area, max_area=max_area)
                return final_mask

        elif plane_fit_to == 'depth_map':
            def rgbd_2_mask(
                    image: NDArray[np.uint8], depth: NDArray[np.uint8], 
                    eps: int = 9, 
                    max_residual=0.01,
                    min_area=300, max_area=10000
                    ):
                """
                TODO
                :param eps: Total displacement of pixel values from the most frequent pixel value
                    which will be considered as background.  peak-eps <= background_pixels <= peak+eps
                :param max_residual: Accepted projection error for plane projection inliers. 
                    Points within this distance will be considered part of the plane.
                """
                hist_mask = new_filter_histogram(image=image, eps=eps)

                plane_mask = filter_out_of_plane_depth_map(
                        estimator=estimator, hist_mask=hist_mask,
                        depth_map=depth,
                        color_image=image, grid=grid,
                        max_residual=max_residual)

                final_mask = draw_contours(self.img_size, mask=plane_mask.astype(np.uint8), 
                                           color=(255,0,0), min_area=min_area, max_area=max_area)
                return final_mask

        return rgbd_2_mask


if __name__=='__main__':

    #scene: str = 'green_canvas'
    scene: str = 'tabletop'
    
    if scene == 'green_canvas':
        filename_ref: str ='table_top_empty.hdf5'; object_name_ref: str ='table_top_empty'
        filename: str ='green_canvas_objects.hdf5'; object_name: str ='green_canvs_objects'
    elif scene == 'tabletop':
        filename_ref: str ='green_canvas_empty.hdf5'; object_name_ref: str ='green_canvs_empty'
        filename: str ='table_top_gears_clipper.hdf5'; object_name: str ='table_top_gears_clipper'
    else:
        raise ValueError("What scene?")
    
    maxlen: int = 8
    # Load reference frames
    #(colors_ref: NDArray[np.uint8],
    # depths_ref: NDArray[np.uint16],
    # depth_scale_ref: float, cam_K_ref NDArray[float], _) = load_recorded_frames(
    #        filename=filename_ref, object_name=object_name_ref )
    (colors_ref, depths_ref, depth_scale_ref, cam_K_ref, _) = load_recorded_frames(
            filename=filename_ref, object_name=object_name_ref )

    img_size: tuple[int, int] = depths_ref[0].shape

    segmentator = Segmentator(
            max_n_ref_frames=8,
            img_size=img_size
            )

    ### Compute histogram masking params from reference images.
    filter_ = segmentator.set_reference_frames(frames=colors_ref)

    # Load inference frames
    colors, depths, depth_scale, cam_K, fps = load_recorded_frames(
        filename=filename, object_name=object_name
    )
    
    assert depth_scale == depth_scale_ref
    assert (cam_K == cam_K_ref).all()
    cam_K = cam_K.astype(np.float64)
    frames = [(c, d) for (c, d) in zip(colors, depths)]
    n_frames = len(frames)
    print("Loaded", n_frames, "frames.")

    masked_color = np.zeros_like(colors[0])
    masked_depth = np.zeros_like(depths[0])

    window_name = 'win'
    def nothing(value) -> None:
        pass

    init_min_area = 300
    init_max_area = 30000
    init_max_distance = 1
    init_max_residual = 10 # In millimeters
    init_eps = 9
    init_max_trials = 2
    
    # create trackbars for color change
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('min_area',window_name,init_min_area,5000,nothing)
    cv2.createTrackbar('max_area',window_name,init_max_area, 100000,nothing)
    cv2.createTrackbar('max_distance',window_name,init_max_distance, 100,nothing)
    cv2.createTrackbar('max_residual',window_name,init_max_residual, 100,nothing)
    cv2.createTrackbar('eps',window_name,init_eps,256,nothing)


    try:
        idx = -1
        while True:
            idx += 1

            c, d = frames[idx % n_frames]

            time_a = perf_counter()

            # get current positions of four trackbars
            min_area = cv2.getTrackbarPos('min_area',window_name)
            max_area = cv2.getTrackbarPos('max_area',window_name)
            max_dist = cv2.getTrackbarPos('max_distance',window_name)
            max_residual = cv2.getTrackbarPos('max_residual',window_name)/1000
            eps = cv2.getTrackbarPos('eps',window_name)

            mask = filter_(
                    image=c, depth=d*depth_scale,
                    eps = eps, 
                    max_residual=max_residual,
                    min_area=min_area, max_area=max_area
                    )

            processing_fps = 1/(perf_counter() - time_a)
            masked_color[:] = c

            for item_idx in range(mask.shape[0]):
                masked_color[mask[item_idx]] = (255,105,180)

            images = cv2.addWeighted(c, 0.5, masked_color, 0.5, 0)

            cv2.putText(images, f"fps: {processing_fps:2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, images)
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    except IndexError as e:
        print(e)
        breakpoint()
