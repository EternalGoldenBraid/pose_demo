import cv2
import numpy as np
from numba import prange, njit
from time import time

class ContourSegmentator():
    def __init__(self):
        self.image_area = 480*640
        self.blur: int = 21
        self.canny_low: int = 15; self.canny_high: int = 150
        self.min_area: float = 0.0005; self.max_area: int = 0.95
        self.dilate_iter: int = 10; self.erode_iter: int = 10
        self.mask_color = (0.0,0.0,0.0);
        self.mask_dilate_iter: int = 5; self.mask_erode_iter: int = 5
        self.max_area = self.max_area * self.image_area
        self.min_area = self.min_area * self.image_area

    def get_mask(self, frame):
        # Convert image to grayscale        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, self.canny_low, self.canny_high)
    
        # Post process
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
    
        # get the contours and their areas
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    
        mask = np.zeros([480,640], dtype = np.uint8)
        ### MAP HERE?
        ## Go through and find relevant contours and apply to mask
        #for contour in contour_info:
        #    # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        #    if contour[1] > min_area and contour[1] < max_area:
        #        # Add contour to mask
        #        mask = cv2.fillConvexPoly(mask, contour[0], (255))
    
        ### MAP HERE?
        # Go through and find relevant contours and apply to mask
        #for contour in contour_info:
        for idx in prange(len(contour_info)):
            # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
            if contour_info[idx][1] > self.min_area and contour_info[idx][1] < self.max_area:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour_info[idx][0], (255))
    
    
        ### POST PROCESSING USE THE ONELINER?
        # use dilate, erode, and blur to smooth out the mask
        mask = cv2.dilate(mask, None, iterations=self.mask_dilate_iter)
        mask = cv2.erode(mask, None, iterations=self.mask_erode_iter)
        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
    
        return mask

if __name__=='__main__':

    # initialize video from the webcam
    #video = cv2.VideoCapture(cv2.CAP_V4L2)
    video = cv2.VideoCapture(0)
    seg = ContourSegmentator()
    while True:
        start = time()
        ret, frame = video.read()
        if ret == True:
    
            mask = seg.get_mask(frame)
    
            # Ensures data types match up
            mask = (mask.astype('float32') / 255.0)
            frame = frame.astype('float32') / 255.0
    
            # Blend the image and the mask
            #masked = (mask[...,None] * frame) + ((1-mask[...,None]) * mask_color)
            #masked = (masked * 255).astype('uint8')
            #cv2.imshow("Foreground", np.hstack[(masked, (frame*255).astype('uint8'))])
            #import pdb; pdb.set_trace()
            #images = np.hstack((masked, (frame*255).astype('uint8')))
    
            edges = edges.astype('float32')/255
            edges = np.dstack( (edges, edges, edges) )
            images = np.hstack((frame*mask[...,None], frame, edges))
            #cv2.imshow("Foreground", np.hstack([(masked, (frame*255).astype('uint8'))]))
            cv2.imshow("Foreground", images)
            #cv2.imshow("Foreground", mask)
    
            print("Time: ", time()-start)
    
            ## Use the q button to quit the operation
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        else: 
            continue
    
    cv2.destroyAllWindows()
    video.release()
