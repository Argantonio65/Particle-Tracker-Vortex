# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:17:57 2016

@author: Antonio Moreno
"""
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt;
#from moviepy.video.io.bindings import mplfig_to_npimage;
import pandas as pd
import argparse
import imutils
import os
from collections import deque
from Main_functions import writing_Output, LED_sync, Particle_tracker, read_configuration_csv, printProgress, Timer, Undistort_Image
import time




ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default = '/Users/ammorenorodena/Desktop/OMNIA/Knowledge/Experiments/Vortex_particle_tracking/Development/RedParticle_right_trimmed.MOV',  help="Path to the video .MOV")
ap.add_argument("-B", "--buffer", type=int, default=500, help="Buffer length for plotting particle trail")
ap.add_argument("-c", "--config",  default='/Users/ammorenorodena/Desktop/OMNIA/Knowledge/Experiments/Vortex_particle_tracking/Development/Configfiles/Conf_file_CAM1_right.csv', help="Path for configuration file")
ap.add_argument("-o", "--outputpath",  default='/Users/ammorenorodena/Desktop/OMNIA/Knowledge/Experiments/Vortex_particle_tracking/Development/Configfiles/', help="Path for configuration file")
ap.add_argument("-p", "--plotting", default = True, help= 'Activate/deactivate video display')

args = vars(ap.parse_args())

#%% READ configuration file
configuration = args["config"]
Exp_name, Exp_id, Cam_id, CAMname, ROI_LED, ROI_tracker, Distortion_params, HSVmin, HSVmax, MatrixIntrinsic = read_configuration_csv(configuration)

#%%  VIDEO READER
cap = cv2.VideoCapture(args["video"])
totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #get length of video
fps    = cap.get(cv2.CAP_PROP_FPS)
#print "Total number of video-frames {}".format(totalframecount)
#print "fps {}".format(fps)
#print "Expected duration {} s".format(totalframecount/float(fps))
ret, frame = cap.read()
h,w,_ = frame.shape

def nothing(x):
    pass


#%% LED syncron
LEDsync = LED_sync(frame,ROI_LED); #initialize LED syncronizer. Supply frame and ROI_LED

#%% PARTICLE TRACKER INITIALIZATION
Particle_track = Particle_tracker(frame, args["buffer"],ROI_tracker,HSVmin,HSVmax)
t1 = Timer()

while(True):
    
    ret, frame = cap.read()


    if ret == True:
        #Undistort the frame only if Distortion_params list is not empty
        frame = Undistort_Image(frame, Distortion_params, MatrixIntrinsic)

        Particle_track.track(frame)
        Particle_track.filter_mask(erode = True, dilate = True, iter = 4)
        Particle_track.plot_particle(frame)
        Particle_track.plot_particle_trail(frame, args["buffer"])

        #LED synchronization.
        Masked = LEDsync.detectLED(frame,Particle_track.Masked) #detect LED and plot it on the masked for debugging
        
        activeplot = False
        if activeplot == True:
            Masked = LEDsync.plot_show(Particle_track.Masked) # Plot LED detection pattern Only for debugging (memory demanding)

        # Display the resulting
        if args["plotting"]== True: #Activate deactivate from arguments the video display.
            cv2.imshow(CAMname + '_display',imutils.resize(np.hstack([frame,Masked]), width = 700))

    else:
        break
        print "dropped frame"

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #PROGRESS
    #printProgress(len(LEDsync.B), totalframecount, barLength = 20)

#%% WRITING TO OUTPUT 
writing_Output(Exp_id, Cam_id, LEDsync.B, Particle_track.u_coord, Particle_track.v_coord, args["outputpath"] + '/SyncRecord_' + CAMname + '_' + str(Exp_name) + '.txt')
t1.stop()



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
