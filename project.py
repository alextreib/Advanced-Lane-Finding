
# coding: utf-8

# # Udacity Self-Driving Car Nanodegree
# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 

# ### Imports

# In[1]:


import numpy as np
import cv2
import os
import pickle
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.image as mpimg
from collections import deque
# get_ipython().run_line_magic('matplotlib', 'inline')

print('Imports successful')


# ## Utils

# In[2]:


def compare_display_imgs(in_img, out_img):
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    
    # Visualize input vs. pipeline img
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    fig.subplots_adjust(hspace = .2, wspace=.05)
    
    ax1.imshow(in_img)
    ax1.set_title('Input Img', fontsize=30)
    
    ax2.imshow(out_img)
    ax2.set_title('Output img', fontsize=30)
    return True


# In[3]:


def compare_img_pipeline(imgFilename):
    img = cv2.imread(imgFilename)
    out_img=pipeline(img)
    
    compare_display_imgs(img,out_img)
    
    return out_img


# In[4]:


def run_pipeline_and_save(imgFilename):
    print (imgFilename)
    img = cv2.imread(imgFilename)
    out_img=pipeline(img)
    fileName,extension=os.path.splitext(imgFilename)
    
    cv2.imwrite(fileName+"_done"+extension,out_img)
    return out_img


# In[5]:


def save_dict(dict,filename):
    pickle.dump(dict, open(filename, "wb" ))
    return


# In[6]:


def load_dict(filename):
    dict = pickle.load(open(filename, "rb" ))
    return dict


# # Framework 

# ## Distortion correction

# In[6]:


# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# # termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d points in real world space
# imgpoints = [] # 2d points in image plane.

# # Make a list of calibration images
# images = glob.glob('./camera_cal/calibration*.jpg')

# # Step through the list and search for chessboard corners
# for i, filename in enumerate(images):
#     in_img = cv2.imread(filename)
#     img_chessboard = np.copy(in_img)
#     gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
    
#     # Find the chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

#     # If found, add object points, image points
#     if ret == True:
#         objpoints.append(objp)
#         # this step to refine image points was taken from:
#         # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#         corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners)
        
#         # Draw chessboard
#         cv2.drawChessboardCorners(img_chessboard, (9,6), corners, ret)
        
#         compare_display_imgs(in_img,img_chessboard)


# # *some of these images do not appear because the specified number of chessboard corners were not found*

# # In[ ]:


# # (Based on previous code snippet)
# # Calculate the camera matrix and distortion coefficients based on the chessboard calculations
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, in_img.shape[:2],None,None)
# dst = cv2.undistort(in_img, mtx, dist, None, mtx)

# # Save values
# calibDict = {}
# calibDict["mtx"] = mtx
# calibDict["dist"] = dist
# save_dict(calibDict,"calib.p")


# In[8]:


# Function for the undistorting the image
def undistort(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    calibDict=load_dict("calib.p")
    img_undistort = cv2.undistort(img, calibDict["mtx"], calibDict["dist"], None, calibDict["mtx"])
    return img_undistort

# # Test undistortion on an example image from the given camera
# img = cv2.imread('./camera_cal\\calibration1.jpg')
# img_undistort=undistort(img)

# compare_display_imgs(img, img_undistort)

# print ("Is the chessboard equidistant?")


# ## Perspective Transform

# In[9]:


# lanewidth -> 100 covers nearly one line, 600 all lines
lanewidth=300
h=720
w=1280

# define src points through getting a plane in the image
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])

# Defining the destinations points through offset and w/h of the given image
dst = np.float32([(lanewidth,0),
                  (w-lanewidth,0),
                  (lanewidth,h),
                  (w-lanewidth,h)])

def get_camera_matrix():
    return cv2.getPerspectiveTransform(src, dst)
    
def get_camera_matrix_inv():
    return cv2.getPerspectiveTransform(dst, src)


# In[10]:


def birds_eye(img):
    # Perform the transformation
    h,w=img.shape[:2]
    img_birds_eye = cv2.warpPerspective(img, get_camera_matrix(), (w,h), flags=cv2.INTER_LINEAR)
    return img_birds_eye

def birds_eye_inv(img):
    # Perform the transformation
    h,w=img.shape[:2]
    img_birds_eye_inv = cv2.warpPerspective(img, get_camera_matrix_inv(), (w,h), flags=cv2.INTER_LINEAR)
    return img_birds_eye_inv
    
    
## Test the function ##
def bird_eye_imgfile(filename):
    img = cv2.imread(filename)
    img_undistort=undistort(img)
    img_birds_eye = birds_eye(img_undistort)
    
    compare_display_imgs(img, img_birds_eye)

    return img_birds_eye

# bird_eye_imgfile('./test_images\\straight_lines1.jpg')


# In[10]:


# for imgFileName in glob.glob('test_images/test*.jpg'):
#     bird_eye_imgfile(imgFileName)


# ### Apply Binary Thresholds

# Visualize perspective transform on example image

# ### Visualize multiple colorspace channels

# In[11]:


def threshold(img):
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]   

    s_thresh_min = 175
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    
    l_thresh_min = 210
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    return combined_binary

## Test the function ##
def threshold_imgfile(filename):
    img = cv2.imread(filename)
    img_undistort=undistort(img)
    img_birds_eye = birds_eye(img_undistort)
    img_threshold = threshold(img_birds_eye)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    fig.subplots_adjust(hspace = .2, wspace=.05)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img)
    ax1.set_title('Input Img', fontsize=30)
    
    ax2.imshow(img_threshold,cmap='gray')
    ax2.set_title('Output img', fontsize=30)
    
    return img_threshold

# img_threshold=threshold_imgfile('./test_images\\test2.jpg')

# # Histogram in threshold img
# histogram = np.sum(img_threshold[img_threshold.shape[0]//2:,:], axis=0)
# plt.plot(histogram)


# # In[12]:


# # Histogram analysis
# histogram = np.sum(img_threshold[img_threshold.shape[0]//2:,:], axis=0)
# plt.plot(histogram)


# ### Threshold on all images

# In[13]:


def threshold_display_compare(imgFileName):
    img = cv2.imread(imgFileName)
    img_undistort = undistort(img)
    img_birds_eye = birds_eye(img_undistort)
    img_threshold = threshold(img_birds_eye)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img)
    ax1.set_title('Orig', fontsize=30)
    ax2.imshow(img_threshold,cmap='gray')
    ax2.set_title('Threshold img', fontsize=30)
    print('Here is your comparison.')
    return img_threshold

# for image in glob.glob('test_images/test*.jpg'):
#     threshold_display_compare(image)


# In[19]:


def get_NewaverageList(newValuesList,averageList):
    """ 
    Calculate a new average of a list with the given newValueList
    and the previously calculated averageList 
    """
    if(averageList==[]):
        averageList=np.zeros(len(newValuesList))
    for i, newValue in np.ndenumerate(newValuesList):
        diff=newValue-averageList[i[0]]
        averageList[i[0]]=averageList[i[0]]+diff/500
    return averageList

ym_per_pix = 30./720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

class Lane(object):
    """ 
    Class that contains all the lane related functionality to calculate the lanes/curvature/distance to middle 
    """
    def __init__(self):
        # Initialization
        self.lefty=np.array([])
        self.leftx=np.array([])
        self.righty=np.array([])
        self.rightx=np.array([])
        self.lane_counter=0
        
        self.average_dict={}
        self.average_dict['left_fit']=[]
        self.average_dict['right_fit']=[]
        self.average_dict['lefty']=[]
        self.average_dict['leftx']=[]
        self.average_dict['righty']=[]
        self.average_dict['rightx']=[]
        
        self.lane_counter_threshold=10
        return
            
    def find_lanes(self,binary_warped):
        """
        Call this function to find the lanes, others are private
        """
        # Init phase -> do blind search
        if(self.lane_counter<self.lane_counter_threshold):
            self.lane_counter+=1
            # Find lanes without any knowledge about history
            return self._find_lanes_blind(binary_warped)
        else:
            # Find lanes with previously calculated parameters
            out_img=self._find_lanes_bias(binary_warped)
            #self._filter_lanes()
            return out_img
        
    def _find_lanes_blind(self,binary_warped):
        """
        Find/calculate the coefficients for the lanes with prior knowledge (sliding window approach).
        """
        # Code mostly taken from class
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        max_minpix=1
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order (curve) polynomial to each
        # Check whether a lane is found
        if(lefty.size==0 or righty.size==0):
            # Not found -> Keep same parameters are the previous image (do nothing)
            
            # Check whether a lane is found in the previous image 
            if(self.lefty.size==0 or self.righty.size==0):
                # -> set the parameter Lanefound to False
                self.Lanefound=False
                return out_img
        else:
            # Normal path (lane is found)
            self.righty=righty
            self.rightx=rightx
            self.lefty=lefty
            self.leftx=leftx
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        
        self.Lanefound=True

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig("test.jpg")

        return out_img
    
    def _find_lanes_bias(self,binary_warped):
        """
        Find/calculate the coefficients for the lanes with prior knowledge.
        """
        # Init variables
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        
        # Get left lane related values
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 

        # Get right lane related values
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order (curve) polynomial to each
        # Check whether a lane is found
        if(lefty.size==0 or righty.size==0):
            # Not found -> Keep same parameters are the previous image (do nothing)
            
            # Check whether a lane is found in the previous image 
            if(self.lefty.size==0 or self.righty.size==0):
                # -> set the parameter Lanefound to False
                self.Lanefound=False
                return out_img
        else:
            # Normal path (lane is found)
            self.righty=righty
            self.rightx=rightx
            self.lefty=lefty
            self.leftx=leftx
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        
        self.Lanefound=True
        return out_img

    def _filter_lanes(self):
        """
        NOT WORKING
        This function is supposed to filter the calculated coefficients that they are displayed smoother
        Not working because polyfit returns different size arrays.
        Realized through a kinda-moving average algorithm
        """
        # Calculate new averages
        self.average_dict['left_fit']=get_NewaverageList(self.left_fit,self.average_dict['left_fit'])
        self.average_dict['right_fit']=get_NewaverageList(self.right_fit,self.average_dict['right_fit'])

        # Overwrite the values with the average
        self.left_fit=self.average_dict['left_fit']
        self.right_fit=self.average_dict['right_fit']
        return

    def _fill_lanes(self,orig_img, combined_binary):
        """
        This function draws in the orig_img the lanes.
        based on the provided information of the combined_binary image.
        """
        # Get the coefficients for the curvature
        left_fitx = self.left_fit[0]*self.lefty**2 + self.left_fit[1]*self.lefty + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.righty**2 + self.right_fit[1]*self.righty + self.right_fit[2]
        
        # Calculate the required points
        warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, self.lefty])))])
        pts_right = np.array([np.transpose(np.vstack([right_fitx, self.righty]))])
        pts = np.hstack((pts_left, pts_right))

        # Drawing the lane
        cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        out_img = cv2.warpPerspective(color_warp, get_camera_matrix_inv(), (combined_binary.shape[1], combined_binary.shape[0]))

        # Print curvate and distance to middle into the out_img
        curvature = 'Curvature: ' + ' {:0.6f}'.format(self._curvate_radius()) + 'm '
        cv2.putText(out_img, curvature, (40,70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

        distance = 'Distance to middle: ' + ' {:0.6f}'.format(self._distance_to_middle()) + 'm '
        cv2.putText(out_img, distance, (40,120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

        return cv2.addWeighted(orig_img, 1, out_img, 0.5, 0)
    
    def _curvate_radius(self):
        """
        This function returns the curvature of the lane.
        """
        left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)

        # Radian conversion
        left_curverad = ((1 + (2*left_fit_cr[0]*np.max(self.lefty) + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*np.max(self.lefty) + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
        
        # Half (right/left)
        return int((left_curverad + right_curverad)/2) 
    
    def _distance_to_middle(self):
        """
        This function returns the distance of the car to the middle.
        Positive values symbolize a right offset, a negative a left offset
        """
        # Car middle target is in the middle of the image
        car_position = 1280/2

        rightx_int = self.right_fit[0]*720**2 + self.right_fit[1]*720 + self.right_fit[2]
        leftx_int = self.left_fit[0]*720**2 + self.left_fit[1]*720 + self.left_fit[2]
        
        lane_center = (rightx_int+leftx_int)/2
        center_dist = (car_position - lane_center) * xm_per_pix
        return center_dist

# ## Video processing pipeline (core of the code)

# In[20]:

lane_obj=Lane()

def pipeline(img):  
    global lane_obj
    img_undistort=undistort(img)
    img_birds_eye = birds_eye(img_undistort)
    img_threshold = threshold(img_birds_eye)
    
    lane_obj.find_lanes(img_threshold)
    out_img=lane_obj._fill_lanes(img,img_threshold) 
    return out_img


# In[21]:


# Compare one image
run_pipeline_and_save('test_images\\test3.jpg')

# def compare_img_pipeline(imgFilename):
# imgFile = mpimg.imread('./project_img\\frame7.jpg')
# imgFile = cv2.cvtColor(imgFile, cv2.COLOR_BGR2RGB)
# cv2.imshow("img",imgFile)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# out_img=pipeline(imgFile)
# cv2.imwrite('test.jpg',out_img)



# # In[39]:


# # Compare all images
# for image in glob.glob('project_img/*.jpg'):
#     run_pipeline_and_save(image)


# ### Process Project Video

# In[25]:

def proc_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_img = pipeline(img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    return out_img


# In[26]:


def process_video():
    from moviepy.editor import VideoFileClip

    video_input = VideoFileClip('project_video.mp4', audio=False)

    processed_video = video_input.fl_image(proc_img)

    video_output = 'project_video_output.mp4'
    processed_video.write_videofile(video_output, audio=False)

    video_input.reader.close()
    return


process_video()