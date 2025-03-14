### utils.py
### utility functions like JSON encoding (saving results), filename sanitization (removing special characters and emojis for window file/folder naming), and score calculations


import cv2
import ffmpeg
import json
import os
import re
import torch
import numpy as np
import streamlit as st
import torch.nn.functional as F
import torchvision.transforms as transforms

# Utility class for handling numpy types in JSON
class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def sanitize_filename(filename):
    # Remove special characters, emojis, and spaces
    sanitized = re.sub(r'[^\w\-. ]', '', filename)  # Keep letters, numbers, dots, dashes, spaces
    return sanitized.strip()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def save_results(output_dir, video_name, results):
    history_file = "./saves/processed_videos.json"

    # Ensure saves directory exists
    os.makedirs("./saves", exist_ok=True)

    # Load existing history if available
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = {}

    # Save results for this video
    history[video_name] = results

    with open(history_file, "w") as f:
        json.dump(history, f, cls=NumpyTypeEncoder, indent=4)

def weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5):
    safe_score = bert_weight * bert_scores['safe'] + resnet_weight * resnet_scores['safe']
    harmful_score = bert_weight * bert_scores['harmful'] + resnet_weight * resnet_scores['harmful']

    if harmful_score > safe_score:
        return "Harmful", harmful_score
    else:
        return "Safe", safe_score

def calculate_average_scores(confidence_scores_by_class):
    return {class_name: (sum(scores) / len(scores) if scores else 0.0) for class_name, scores in confidence_scores_by_class.items()}


### end
