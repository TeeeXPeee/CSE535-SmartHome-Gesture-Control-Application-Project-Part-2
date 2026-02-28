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

mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'DecreaseFanSpeed': 10,
    'FanOff': 11,
    'FanOn': 12,
    'IncreaseFanSpeed': 13,
    'LightOff': 14,
    'LightOn': 15,
    'SetThermo': 16,
}

def generate_middle_frames(video_path_dir, frame_save_dir):
    results = {}
    for count, video in enumerate(os.listdir(video_path_dir)):
        video_path = os.path.join(video_path_dir, video)
        frame_path = frameextractor.frameExtractor(video_path, frame_save_dir, count)
        results[frame_path] = video
    
    return results

def extract_features_from_frames(frame_dir, feature_extrator):
    features = {}
    for filename in os.listdir(frame_dir):
        path = os.path.join(frame_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        feature = feature_extrator.extract_feature(img)
        feature_vec = feature.squeeze()
        feature_vec = feature_vec / np.linalg.norm(feature_vec) if np.linalg.norm(feature_vec) > 0 else feature_vec
        features[filename] = feature_vec
    return features



feature_extractor = HandShapeFeatureExtractor.get_instance()
# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
train_frames = generate_middle_frames('traindata', 'trainframes')
train_features = extract_features_from_frames('trainframes', feature_extractor)

    # Store or use the extracted features as needed
# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
generate_middle_frames('test', 'testframes')
test_features = extract_features_from_frames('testframes', feature_extractor)




# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

def find_closest_gesture(test_dict, train_dict, train_frames):

    result = []

    for _, test in test_dict.items():

        test_norm = np.linalg.norm(test)
        if test_norm > 0:
            test = test / test_norm
        max_similarity = -1
        best_feature = None

        for filename, train in train_dict.items():
            train = train / np.linalg.norm(train) if np.linalg.norm(train) > 0 else train

            score = cosine_similarity(test, train)

            if score > max_similarity:
                max_similarity = score
                best_feature = filename
        path = train_frames[best_feature].split('-')[-1][:-4]
        result.append(mapping[path])
    
    return result

recognized_gestures = find_closest_gesture(test_features, train_features, train_frames)
with open('results.csv', 'w') as f:
    for gesture in recognized_gestures:
        f.write(f"{gesture}\n")