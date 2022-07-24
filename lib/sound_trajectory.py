import numpy as np
import numba as nb
from numba import njit, prange
import cv2


#@njit(parallel=True)
def world2image(cam_K, image_size = (640,480, 3), 
        n_points=3, radius=3, 
        height=480, width=640, n_channels=3):

    n_channels = image_size[-1]
    P_rgb = np.zeros((n_channels, n_points), dtype=np.uint8)

    image_size = np.array([height, width, n_channels])

    # Padded kernel for slicing outside the image
    image_kernel = np.zeros(image_size)
    image_kernel = np.pad(image_kernel, constant_values=-1,
            pad_width=[(radius, radius), (radius, radius), (0,0)]
            )

    # Slice image sized slice for return
    kernel_slice: slice = np.s_[radius:radius+image_size[0], radius:radius+image_size[1]]

    def color_points(image, pts):
        """
        pts: (n_pts, channels)
        """
        nonlocal P_rgb #TODO Overhead?
        nonlocal radius

        #breakpoint()
        P = cam_K.dot(pts.T) # C x N
        #P = P // P[-1,:]

        if P[1].max() >= image_size[0] or P[0].max() >= image_size[1]:
            return image

        # Colorize points
        P_rgb[0,:] = P[0,:] % image_size[0]
        P_rgb[1,:] = P[1,:] % image_size[1]
        P_rgb[2,:] = P[2,:] % image_size[1] # TODO
        P_rgb = P_rgb.astype(int)
        P = P.astype(int)

        ### Copy the image to preallocated memory
        # TODO How to avoid copy?
        image_kernel[kernel_slice] = image

        ### Apply 3D pose (color) to pixels.
        for idx in prange(P.shape[1]):
            #TODO Figure out how to handle the coord < radius case.
            # This if clause is a dirty workaround to avoid padding issues.
            if P[:2,idx].min() < radius: continue
            #slice_: slice = np.s_[P[:2][0]-radius:P[:2][0]+radius, P[:2][1]-radius:P[:2][1]+radius]
            h_slice = np.s_[P[:2,idx][1]-radius:P[:2,idx][1]+radius]
            w_slice = np.s_[P[:2,idx][0]-radius:P[:2,idx][0]+radius]
            #slice_: slice = np.s_[P[:2,idx][0]-radius:P[:2,idx][0]+radius, P[:2,idx][1]-radius:P[:2,idx][1]+radius]
            slice_: slice = np.s_[h_slice, w_slice]
            #breakpoint()
            image_kernel[slice_] = \
                P_rgb[:,idx].repeat(repeats=(2*radius)**2, axis=0). \
                    reshape(image_kernel[slice_].shape, order='F')
            #image_kernel[slice_] = P_rgb[idx].repeat(repeats=radius, axis=0)
            #image[np.clip(P[idx][0], 0, image_size[0]), np.clip(P[idx][1], 0,image_size[1])]
            #image[np.clip(P[idx][0], 0, image_size[0]), np.clip(P[idx][1], 0,image_size[1])]


        #image[P[1,:], P[0,:]]  = P_rgb.T
        image = image_kernel[kernel_slice]
        #image[p[1,:], p[0,:]]  = (P_rgb.repeat(repeats=radius, axis=0)).t
        return image.astype(int).astype(np.uint8)

    return color_points
