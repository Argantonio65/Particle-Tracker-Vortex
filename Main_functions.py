# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:17:57 2016

@author: Antonio Moreno
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
#from moviepy.video.io.bindings import mplfig_to_npimage
import pandas as pd
import argparse
import sys
import os
from collections import deque
import argparse
import time

def read_configuration_csv(path):
    global g,f
    with open(path) as f:
        lis=[line.split() for line in f]        # Read
        lis = [val for sublist in lis for val in sublist] # Flaten
        
        Exp_name = map(int, lis[0].split(',')[1:])[0]
        Exp_id = map(int, lis[1].split(',')[1:])[0]
        Cam_id = map(int, lis[2].split(',')[1:])[0]
        CAMname = lis[3].split(',')[1:][0]

        ROI_LED = map(int, lis[4].split(',')[1:5])
        ROI_tracker = map(int, lis[5].split(',')[1:5])

        Distortion_params_fromText = map(float, lis[6].split(',')[1:7])

        Distortion_params = Distortion_params_fromText[0:2] + Distortion_params_fromText[3:5] + [Distortion_params_fromText[2]]
        
        
        Color_boundaryHSVmin = map(int, lis[7].split(',')[1:5])
        Color_boundaryHSVmax = map(int, lis[8].split(',')[1:5])

        cx = float(lis[9].split(',')[1:][0])
        cy = float(lis[10].split(',')[1:][0])

        fx = float(lis[11].split(',')[1:][0])
        fy = float(lis[12].split(',')[1:][0])

        #create cam matrix.
        MatrixIntrinsic = np.array([[fx, Distortion_params_fromText[-1], cx], [0, fy, cy], [0, 0, 1]])
        
    return Exp_name, Exp_id, Cam_id, CAMname, ROI_LED, ROI_tracker, Distortion_params, Color_boundaryHSVmin, Color_boundaryHSVmax, MatrixIntrinsic

    
def writing_Output(Exp_id, Cam_id, B, u_coord, v_coord, name_output):
    '''
    Write file output. specify name_output: example 'SyncRecord_Cam1.txt'
    TO DEVELOP SECTION with x and y from tracker.
    '''
    ########## writing to csv
    CODIGO = pd.DataFrame(B, columns = ['LED'])

    ###########Make up numbers for the tracker coordinates
    f = np.vectorize(lambda x: '{0:.1f}'.format(x))
    CODIGO['u_coord_tracker'] = u_coord #f(100*np.random.random(len(B)))
    CODIGO['v_coord_tracker'] = v_coord #f(100*np.random.random(len(B)))
    ###########################################################

    # Apply a remap index for the synchronizer
    CODIGO['Time_id']= int(0)
    CODIGO.ix[CODIGO.LED!=CODIGO.LED.shift(1),'Time_id']= int(1)
    CODIGO.ix[0,'Time_id']= int(0)
    CODIGO['Time_id'] = CODIGO['Time_id'].cumsum()
    CODIGO['counter'] = CODIGO.groupby('Time_id').cumcount() + 1
    CODIGO['res_exp_rel'] = map(int, np.ones(len(B))*Exp_id)
    CODIGO['res_cam_rel'] = map(int, np.ones(len(B))*Cam_id)

    CODIGO = CODIGO[['res_exp_rel', 'res_cam_rel','Time_id', 'counter', 'LED', 'u_coord_tracker', 'v_coord_tracker']] #reorder columns
    CODIGO.index = np.arange(1, len(CODIGO)+1) #initiate index with 1


    CODIGO.to_csv(name_output, sep = ',', index_label = 'Frame_id')
    return;

def Undistort_Image(frame, Distortion_params, MatrixIntrinsic):
    '''
    Performs the undistortion of the image unless the Distortion_params vector is empty
    '''
    if np.array(Distortion_params).any():
        #Undistort only if Distortion_params is not all 0
        h, w = frame.shape[:2]
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(MatrixIntrinsic, np.array(Distortion_params), (w,h), 0, (w,h))
        frame = cv2.undistort(frame, MatrixIntrinsic, np.array(Distortion_params), None, None)


    return frame

class LED_sync:
    '''
    Class for LED synchronizer apparatus.
    When supplying a viceo_frame and a ROI, this class will:
    1- Initiate a mask on the frame.
    2- Cropp the area in which the LED is located. To be supplied as LED_ROI = [v1:v2,u1:u2] in camera pixel coordinates.
    3- Initiate a plot framework to be embedded in a opencv MAT frame
    4- When calling self.dectedLED(frame), will search for a green LED and in case of detection will supply a value of 255 to a timeseries self.B (else will provide 0).
    5- self.plot_show() will plot the vector B in the figure initiated and will integrate it in a possition in the self.LED_mask which is used for real time display as an opencv frame.

    NOTES: The integrated plot for the detection timeseries (self.B) is used as a debbuging factor to see if the LED is correctly detected.
    '''
    def __init__(self, frame, LED_ROI):
        #Prepare LED detector
        self.LED_mask = np.zeros(frame.shape,np.uint8)
        self.LED_mask[LED_ROI[0]:LED_ROI[1],LED_ROI[2]:LED_ROI[3]] = frame[LED_ROI[0]:LED_ROI[1],LED_ROI[2]:LED_ROI[3]]  #LED_ROI = [v1,v2,u1,u2]   -> [288:311,145:165]
        self.LED_mask_ROI = self.LED_mask[LED_ROI[0]:LED_ROI[1],LED_ROI[2]:LED_ROI[3]]

        self.LED_ROI = LED_ROI

        #Prepare plot for LED figure
        self.fig, self.ax = plt.subplots(figsize=(6,2), facecolor='w');
        self.ax.set_title('Green channel intensity at ROI', fontsize = 10)
        self.ax.set_yticks([0,255])
        self.ax.set_ylim([0,255])


        self.B = [] #vector for plotting the LED sync line
        line, = self.ax.plot(self.B, lw=3)
        plt.box('off');
        plt.tight_layout();
        #graphRGB = mplfig_to_npimage(self.fig);
        #self.gh, self.gw, _ = graphRGB.shape;

    def detectLED(self, frame, Mask):
        Mask[self.LED_ROI[0]:self.LED_ROI[1],self.LED_ROI[2]:self.LED_ROI[3]] = frame[self.LED_ROI[0]:self.LED_ROI[1],self.LED_ROI[2]:self.LED_ROI[3]]  #self.LED_ROI = [v1,v2,u1,u2]   -> [288:311,145:165]
        self.LED_mask_ROI = Mask[self.LED_ROI[0]:self.LED_ROI[1],self.LED_ROI[2]:self.LED_ROI[3]]

        gray_image_LED = cv2.cvtColor(self.LED_mask_ROI, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray_image_LED , 200, 255, cv2.THRESH_BINARY)[1] #threshold with value 200 (change for more sensitivity)
        contours, hierarchy = cv2.findContours(thresh_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0))[-2:]
        cv2.drawContours(self.LED_mask_ROI, contours, -1, (0,255,0), 3) # it returns LED_mask_ROI
        self.B.append(255 if np.max(thresh_img)>0 else 0)
        return Mask

    def plot_show(self, Mask):

        if len(self.B)%100==0:
            self.ax.plot(self.B, lw=1, color = 'g')

        #Mask[-self.gh:,-self.gw:,:] = mplfig_to_npimage(self.fig)
        return Mask

class Particle_tracker:
    '''
    Class for tracking particle:
    When instanciated it needs
    A frame
    A length for the particle trail plotter
    Track_ROI: Bounding box for the region where particles can be found
    HSVmin HSVmax: boundaries for the colour filter. This is done in the HSV color space. To be supplied as a list [0-180, 0-255, 0-255]
    '''
    def __init__(self, frame, buffer_length,Track_ROI, HSVmin, HSVmax):
        self.pts =  deque(maxlen=buffer_length) # buffer for the trail and past locations of the object
        self.TrackROI = Track_ROI
        self.shape = frame.shape
        self.HSVmin = HSVmin
        self.HSVmax = HSVmax

        self.u_coord = []
        self.v_coord = []

    def track(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #TO HSV
        except: # In case frame is corrupted, then it creates a black image and will render -99,-99 for the possition
            self.mask_track = np.zeros(self.shape,np.uint8)
            self.mask_track = cv2.inRange(self.mask_track,(self.HSVmin[0],self.HSVmin[1],self.HSVmin[2]),(self.HSVmax[0],self.HSVmax[1],self.HSVmax[2]))
            self.Masked = cv2.bitwise_and(frame, frame, mask=self.mask_track)
        else: # Frame is ok and can be tracked.
            self.mask_track = np.zeros(self.shape,np.uint8)
            self.mask_track[self.TrackROI[0]:self.TrackROI[1],self.TrackROI[2]:self.TrackROI[3]] = hsv[self.TrackROI[0]:self.TrackROI[1],self.TrackROI[2]:self.TrackROI[3]]
            self.mask_track = cv2.inRange(self.mask_track,(self.HSVmin[0],self.HSVmin[1],self.HSVmin[2]),(self.HSVmax[0],self.HSVmax[1],self.HSVmax[2]))
            



    def filter_mask(self, iter, erode = False, dilate = False):
        if erode == True:
            self.mask_track  = cv2.erode(self.mask_track , None, iterations = iter ) 
        if dilate == True:
            self.mask_track  = cv2.dilate(self.mask_track , None, iterations = iter )


    def plot_particle(self,frame):
        self.Masked = cv2.bitwise_and(frame, frame, mask=self.mask_track)
        

        ## FILTERING REFLECTIONS
        # #Alternative Hough Circles
        # circles_H = cv2.HoughCircles(cv2.cvtColor(self.Masked, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT,1, 2,
        #               param1=30,
        #               param2=15,
        #               minRadius=5,
        #               maxRadius=25)        

        # # ensure at least some circles were found
        # if circles_H is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles_H = np.round(circles_H[0, :]).astype("int")
         
        #     circles_H =  circles_H[circles_H[:,1] == np.max(circles_H[:,1])]
        #     #circles_H[:,1]

        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles_H:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(self.Masked, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(self.Masked, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        #         circle_img  = np.zeros(self.shape,np.uint8)

        #         height,width,depth = frame.shape

        #         # GENERATE A MASK BASED ON HOUGH CIRCLES.  quite noisy.
        #         #self.mask_track = np.zeros((height,width), np.uint8)
        #         #cv2.circle(self.mask_track,(x,y),2*r,1,thickness=-1)
        #         #self.Masked = cv2.bitwise_and(self.Masked, self.Masked, mask=self.mask_track)

        # #----------


        cnts = cv2.findContours(self.mask_track.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        center = None
        rad = []
        if len(cnts)>0:
            #get the largest contour:
            cbiggest = max(cnts, key= cv2.contourArea)
            center = []
            rad = []
            for c in cnts:
                (x,y),radius = cv2.minEnclosingCircle(c)
                center.append([int(x),int(y),radius])
            c = np.array(center)

            Extreme = False
            if len(self.pts) >=100 and Extreme == True:
                r =[]
                for i in range(-100,-1): 
                    r.append(self.pts[i])
                    
                r = np.array(r)
                u_m = np.mean(r[:,0])
                v_m = np.mean(r[:,1])
    
                #get c with extreme u
                c_max = abs(c[:,0]-u_m)
                c = c[np.where(c_max[c_max == max(c_max)])]

                self.u_coord.append(c[0,0])
                self.v_coord.append(c[0,1])

            else:
                
                (x,y),radius = cv2.minEnclosingCircle(cbiggest)

                self.u_coord.append(int(x))
                self.v_coord.append(int(y))

            #Draw circles centre and boundary
            cv2.circle(frame,(int(c[0,0]),int(c[0,1])),int(c[0,2]),(0,255,255),1)
            cv2.circle(frame,(int(c[0,0]),int(c[0,1])) ,2,(0,0,255),-1)

            self.pts.appendleft([c[0,0],c[0,1]])
        else:
            self.u_coord.append(-99)
            self.v_coord.append(-99)
    
    def plot_particle_trail(self, frame, buff):
        for i in xrange(1,len(self.pts)):
            if self.pts[i-1] is None or self.pts[i] is None:
                continue
            tickness = int(np.sqrt(buff/float(i+1))*0.5)
            cv2.line(frame,(int(self.pts[i-1][0]),int(self.pts[i-1][1])),(int(self.pts[i][0]),int(self.pts[i][1])),(0,0,255),tickness)



def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class Timer:
    def __init__(self):
        self.start = time.clock()
    def stop(self):
        print self.__class__.__name__ + ' timed in {} seconds'.format(time.clock() - self.start)

