# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor
import frameextractor

def generate_middle_frames(video_path_dir, frame_save_dir):
    for count, video in enumerate(os.listdir(video_path_dir)):
        print(video)
        video_path = os.path.join(video_path_dir, video)
        frameextractor.frameExtractor(video_path, frame_save_dir, count)

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

generate_middle_frames('traindata', 'trainframes')



    # Store or use the extracted features as needed
# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video




# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================



