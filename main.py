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

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
def extract_features(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read the frame from video.")
        cap.release()
        
        # Preprocess the frame and extract features using the model
        extractor = HandShapeFeatureExtractor.get_instance()
        preprocessed_frame = extractor._HandShapeFeatureExtractor__pre_process_input_image(frame)
        features = extractor.model.predict(preprocessed_frame)
        return features
    except Exception as e:
        print(str(e))
        raise

for video_file in os.listdir(os.path.join('traindata')):
    video_path = os.path.join('traindata', video_file)
    print(extract_features(video_path))
    # Store or use the extracted features as needed
# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video




# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================



