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
    image_kernel = np.zeros(image_size, dtype=np.uint8)
    image_kernel = np.pad(image_kernel, constant_values=0,
            pad_width=[(radius, radius), (radius, radius), (0,0)]
            )

    # Slice image sized slice for return
    kernel_slice: slice = np.s_[radius:radius+image_size[0], radius:radius+image_size[1]]

    # https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
    import pyaudio
    p = pyaudio.PyAudio()

    volume = 0.5     # range [0.0, 1.0]
    duration = 0.4   # in seconds, may be float
    fs = 44100
    #f = np.array([400, 4000, 12000])

    #f_min = 20
    #f_max = 16000
    #fx = np.clip(np.log(np.arange(0,width)), a_min=f_min, a_max=f_max) 
    #fy = np.clip(np.arange(0,height)**2, a_min=f_min, a_max=f_max)
    #fz = np.clip(np.arange(0,5)**2, a_min=f_min, a_max=f_max)

    freqs = np.linspace(20, 5000, num=10)
    
    # generate samples, note conversion to float32 array
    #samples = [(np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32).tobytes() \
    #samples = [(np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32) \
            #for f in freqs]
    sampler = lambda f, phi=0: (np.sin(2*np.pi*np.arange(fs*duration+phi)*f/fs)).astype(np.float32)
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    previous_mean: float = 0.
    max_norm = 0.8
    f_max = 1e3; f_min=20
    # linear regression for frequency.
    k = (f_max - f_min)/max_norm
    frequency = lambda norm: np.clip(k*norm+f_min, a_min=f_min, a_max=f_max)

    prev_signal = np.zeros(int(fs*duration))

    def color_points(image, pts):
        """
        pts: (n_pts, channels)
        """
        nonlocal P_rgb #TODO Overhead?
        nonlocal radius
        nonlocal frequency
        nonlocal volume

        #breakpoint()
        P = cam_K.dot(pts.T) # C x N
        #P = P // P[-1,:]


        if P[1].max() >= image_size[0] or P[0].max() >= image_size[1]:
            return image

        # Colorize points
        P_rgb[0,:] = P[0,:] % image_size[0]
        P_rgb[1,:] = P[1,:] % image_size[1]
        #P_rgb[2,:] = (P[2,:]+30) % image_size[1] # TODO
        P_rgb[2,:] = (P[2,:]+30)**2 # TODO
        P_rgb = P_rgb.astype(int)

        P = P // P[-1,:]
        P = P.astype(int)

        f = frequency(np.linalg.norm(pts,axis=1).mean())
        print(f)
        stream.write(volume*sampler(f))

        #y_sound = samples[int( (P[:,0][0]-P[:,1][0]) /(height/10)-1)]
        #x_sound = samples[int( (P[:,0][1]-P[:,0][1]) /(width/10)-1)]
        #stream.write(y_sound+x_sound)

        #stream.write((np.sin(2*np.pi*np.arange(fs*duration)*400/fs)).astype(np.float32))
        #print("y_sound:", y_sound)
        #print("x_sound:", x_sound)
        #print("y_sound", int(P[:,0][0]/(height/10)))
        #print("x_sound", int(P[:,0][1]/(width/10)))

        ### Copy the image to preallocated memory
        # TODO How to avoid copy?
        image_kernel[kernel_slice] = image


        ### Apply 3D pose (color) to pixels.
        for idx in prange(P.shape[1]):
            #TODO Figure out how to handle the coord < radius case.
            # This if clause is a dirty workaround to avoid padding issues.
            if P[:2,idx].min() < radius: continue
            slice_: slice = np.s_[ \
                P[:2,idx][1]-radius:P[:2,idx][1]+radius, \
                P[:2,idx][0]-radius:P[:2,idx][0]+radius  \
                ]
            # Broadcast colors to kernel.
            image_kernel[slice_] = \
                P_rgb[:,idx].repeat(repeats=(2*radius)**2, axis=0). \
                    reshape(image_kernel[slice_].shape, order='F')
                    #reshape(image.shape, order='F')

        image = image_kernel[kernel_slice]
        return image.astype(int).astype(np.uint8)

    return color_points
