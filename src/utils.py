### src/utils.py


### IMPORTS
### ________________________________________________________________
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
from fpdf import FPDF


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

# function to select diverse frames to avoid showing similar ones
def select_diverse_frames(nsfw_frames, max_frames=5):
  if not nsfw_frames:
    return []

  if len(nsfw_frames) <= max_frames:
    return nsfw_frames

  # Sort by confidence first
  sorted_frames = sorted(nsfw_frames, key=lambda x: x['confidence'], reverse=True)

  # Take top frame and then select frames that are spaced out
  selected = [sorted_frames[0]]

  # Try to select frames that are spaced out
  spacing = max(1, len(nsfw_frames) // max_frames)
  remaining_slots = max_frames - 1

  for i in range(spacing, len(sorted_frames), spacing):
    if remaining_slots <= 0:
      break
    selected.append(sorted_frames[i])
    remaining_slots -= 1

  return selected

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

def save_to_pdf(video_name, history_file):  # Changed parameter name from result_path to history_file
    output_dir = os.path.join("saves", "reports", video_name)
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, f"{video_name}_report.pdf")

    # Load results from the provided history file
    if not os.path.exists(history_file):
        raise FileNotFoundError("Processed videos history file not found!")

    with open(history_file, "r") as f:
        history = json.load(f)

    if video_name not in history:
        raise ValueError(f"Results for '{video_name}' not found in the history file.")

    results = history[video_name]

    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt=f"Analysis Report: {video_name}", ln=True, align="C")
    pdf.ln(10)

    # Add final prediction and confidence
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Final Prediction: {results['final_prediction']} ({results['final_confidence']*100:.2f}%)", ln=True)
    pdf.ln(5)

    # Add Text Classification Results
    pdf.cell(200, 10, txt="Text Classification Results:", ln=True)
    pdf.cell(200, 10, txt=f"- Harmful: {results['harmful_conf_text']*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"- Safe: {results['safe_conf_text']*100:.2f}%", ln=True)
    pdf.ln(5)

    # Add Visual Classification Results
    pdf.cell(200, 10, txt="Video Classification Results:", ln=True)
    pdf.cell(200, 10, txt=f"- Harmful: {results['harmful_score_resnet']*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"- Safe: {results['safe_score_resnet']*100:.2f}%", ln=True)
    pdf.ln(5)

    # Add Transcription
    pdf.cell(200, 10, txt="Video Transcription:", ln=True)
    pdf.ln(5)
    for segment in results["transcription"]:
        pdf.cell(200, 10, txt=f"{segment['start_time']}s: {segment['text']}", ln=True)

    # Save PDF
    pdf.output(pdf_path)
    return pdf_path


### END
### ________________________________________________________________
