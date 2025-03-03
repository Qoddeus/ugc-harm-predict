import cv2
import ffmpeg
import os
import re
import shutil
import torch
import numpy as np
import streamlit as st
import tensorflow as tf
import torch.nn as nn

from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from transformers import BertModel, BertTokenizer, pipeline


# PREP________________________________________________________________________________________________________________________________________

# Define the custom model
class CustomBertClassifier(nn.Module):
  def __init__(self, bert_model, num_labels):
    super(CustomBertClassifier, self).__init__()
    self.bert = bert_model  # Base BERT model
    self.fc = nn.Linear(bert_model.config.hidden_size, num_labels)  # Fully connected layer
    self.num_labels = num_labels # Save num_labels as an attribute

  def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pooled_output = outputs.pooler_output  # Use the pooled output from BERT
    logits = self.fc(pooled_output)  # Pass through the fully connected layer

    loss = None
    if labels is not None: # Calculate loss only during training
      loss_fct = nn.CrossEntropyLoss() # Or any other suitable loss function
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    return loss, logits # Return loss and logits

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = pipeline("automatic-speech-recognition", "openai/whisper-tiny.en", torch_dtype=torch.float16, device=device)

tokenizer = BertTokenizer.from_pretrained("models/bert")
bert_model = BertModel.from_pretrained("models/bert")
bert_model = CustomBertClassifier(bert_model, num_labels=2).to(device)
bert_model.load_state_dict(torch.load("models/bert/pytorch_model.bin", map_location=device))

resnet_model = load_model("models/resnet50")
class_names = ['nsfw', 'safe', 'violence']


# FUNCTIONS___________________________________________________________________________________________________________________________________

def extract_audio(video_path, audio_path):
  stream = ffmpeg.input(video_path)
  stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16000')
  ffmpeg.run(stream)

def transcribe_audio(audio_path, whisper_model):
  transcription = whisper_model(audio_path, return_timestamps=True)
  return transcription["text"]

def classify_text(text, bert_model, tokenizer, device):
  bert_model.eval()
  inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
  with torch.no_grad():
    _, logits = bert_model(
      input_ids=inputs['input_ids'],
      attention_mask=inputs['attention_mask']
    )
  probs = torch.nn.functional.softmax(logits, dim=-1)
  harmful_confidence = probs[0][1].item()
  safe_confidence = probs[0][0].item()
  label = "Harmful" if harmful_confidence > safe_confidence else "Safe"
  return label, harmful_confidence, safe_confidence

def process_frames(video_path, output_dir, resnet_model, class_names, batch_size=32):
  os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

  processed_frames_dir = os.path.join(output_dir, "processed_frames")  
  frame_example = cv2.imread(os.path.join(processed_frames_dir, "frame_0001.jpg"))  # Fix path
  os.makedirs(processed_frames_dir, exist_ok=True)  # Ensure processed frames directory exists

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError(f"Error: Could not open video at {video_path}")

  frame_count = 0
  batch_frames = []
  predictions_per_frame = []
  confidence_scores_by_class = {class_name: [] for class_name in class_names}

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = img_to_array(resized_frame)
    img_array = preprocess_input(img_array)
    batch_frames.append((frame, img_array))

    if len(batch_frames) == batch_size or not ret:
      batch_input = np.array([frame[1] for frame in batch_frames])
      predictions = resnet_model.predict(batch_input)

      for i, (original_frame, prediction) in enumerate(zip(batch_frames, predictions)):
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction[predicted_class_index]

        original_frame = original_frame[0]
        cv2.putText(
          original_frame,
          f"Predicted: {predicted_class_name} ({confidence:.4f})",
          (10, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0, 255, 0),
          2,
          cv2.LINE_AA,
        )

        output_path = os.path.join(processed_frames_dir, f"frame_{frame_count - len(batch_frames) + i + 1:04d}.jpg")
        cv2.imwrite(output_path, original_frame)
        predictions_per_frame.append((frame_count - len(batch_frames) + i + 1, predicted_class_name, confidence))
        confidence_scores_by_class[predicted_class_name].append(confidence)

      batch_frames = []

  cap.release()
  return frame_count, predictions_per_frame, confidence_scores_by_class

def calculate_average_scores(confidence_scores_by_class):
  average_confidence_by_class = {}
  for class_name, scores in confidence_scores_by_class.items():
    average_confidence_by_class[class_name] = sum(scores) / len(scores) if scores else 0.0
  return average_confidence_by_class

def combine_frames_to_video(output_dir, output_video_path, frame_count, audio_path, frame_rate=30):
  """
  Combine frames into a video and add the original audio.

  Args:
    output_dir (str): Directory containing the processed frames.
    output_video_path (str): Path to save the final video.
    frame_count (int): Number of frames to include.
    audio_path (str): Path to the original audio file.
    frame_rate (int): Frame rate of the video.
  """
  # Step 1: Combine frames into a video (without audio)
  temp_video_path = "temp_video.mp4"
  frame_example = cv2.imread(os.path.join(output_dir, "frame_0001.jpg"))
  height, width, _ = frame_example.shape
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  video_writer = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (width, height))

  for i in range(1, frame_count + 1):
    frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

  video_writer.release()

  # Step 2: Add audio to the video using FFmpeg
  input_video = ffmpeg.input(temp_video_path)
  input_audio = ffmpeg.input(audio_path)
  ffmpeg.output(input_video, input_audio, output_video_path, vcodec="h264", acodec="aac", strict="experimental").run()


def weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5):
  """
  Perform weighted fusion of confidence scores from BERT and ResNet models.

  Args:
    bert_scores (dict): Confidence scores from BERT {'safe': value, 'harmful': value}.
    resnet_scores (dict): Confidence scores from ResNet {'safe': value, 'harmful': value}.
    bert_weight (float): Weight for BERT scores (default 0.5).
    resnet_weight (float): Weight for ResNet scores (default 0.5).

  Returns:
    tuple: Final prediction ('Safe' or 'Harmful') and the corresponding confidence score.
  """
  safe_score = bert_weight * bert_scores['safe'] + resnet_weight * resnet_scores['safe']
  harmful_score = bert_weight * bert_scores['harmful'] + resnet_weight * resnet_scores['harmful']

  if harmful_score > safe_score:
    return "Harmful", harmful_score
  else:
    return "Safe", safe_score

def get_unique_folder_name(base_name, output_root="output"):
  folder_name = base_name
  folder_path = os.path.join(output_root, folder_name)
  count = 2

  while os.path.exists(folder_path):
    folder_name = f"{base_name}_{count}"
    folder_path = os.path.join(output_root, folder_name)
    count += 1

  os.makedirs(folder_path, exist_ok=True)
  return folder_path

def get_video_folders(output_root="output"):
  return [f for f in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, f))]

def get_metadata(video_folder):
  metadata_path = os.path.join("output", video_folder, "saves", "metadata.txt")
  metadata = {}
  if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
      for line in f:
        key, value = line.strip().split(": ")
        metadata[key] = value
  return metadata

def save_metadata(video_folder, metadata):
  saves_path = os.path.join("output", video_folder, "saves")
  os.makedirs(saves_path, exist_ok=True)
  metadata_path = os.path.join(saves_path, "metadata.txt")
  with open(metadata_path, "w") as f:
    for key, value in metadata.items():
      f.write(f"{key}: {value}\n")

def delete_video(video_folder):
  shutil.rmtree(os.path.join("output", video_folder))

def generate_thumbnail(video_path, thumbnail_path):
  cap = cv2.VideoCapture(video_path)
  success, frame = cap.read()
  if success:
    cv2.imwrite(thumbnail_path, frame)
  cap.release()

# STREAMLIT___________________________________________________________________________________________________________________________________

def main():
  st.title("Harmful Content Detection in Short Videos")
  tab1, tab2 = st.tabs(["📤 Upload & Process", "📂 View Processed Videos"])

  with tab1:
    uploaded_file = st.file_uploader("Upload a short video (MP4 format)", type=["mp4"])
    if uploaded_file is not None:
      base_name = re.sub(r'\W+', '_', os.path.splitext(uploaded_file.name)[0])
      temp_dir = "temp"
      os.makedirs(temp_dir, exist_ok=True)
      temp_video_path = os.path.join(temp_dir, uploaded_file.name)
      with open(temp_video_path, "wb") as f:
          f.write(uploaded_file.read())

      st.video(temp_video_path)
      if st.button("Process video"):
        st.write(f"Processing video: **{uploaded_file.name}**")
        progress_bar = st.progress(0)
        output_folder = os.path.join("output", base_name)
        os.makedirs(output_folder, exist_ok=True)
        video_path = os.path.join(output_folder, uploaded_file.name)
        os.rename(temp_video_path, video_path)
        saves_folder = os.path.join(output_folder, "saves")
        os.makedirs(saves_folder, exist_ok=True)

        processed_frames_folder = os.path.join(output_folder, "processed_frames")
        frame_count, predictions, confidence_scores = process_frames(video_path, processed_frames_folder, resnet_model, class_names)
        progress_bar.progress(40)

        if frame_count == 0:
          st.error("No frames were processed. There might be an issue with reading the video.")
        
        output_video_path = os.path.join(output_folder, f"processed_{uploaded_file.name}")
        audio_path = os.path.join(output_folder, "output_audio.wav")
        thumbnail_path = os.path.join(saves_folder, "thumbnail.jpg")

        extract_audio(video_path, audio_path)
        transcription = transcribe_audio(audio_path, whisper_model)
        label, harm_conf, safe_conf = classify_text(transcription, bert_model, tokenizer, device)
        progress_bar.progress(70)

        generate_thumbnail(video_path, thumbnail_path)
        progress_bar.progress(100)


        # os.remove(audio_path)
        st.success("Processing complete!")
        st.video(output_video_path)
        save_metadata(base_name, {"Final Prediction": label, "Confidence": max(harm_conf, safe_conf)})

      with tab2:
        st.subheader("Previously Processed Videos")
        video_folders = get_video_folders()
        for video_folder in video_folders:
          metadata = get_metadata(video_folder)
          processed_video_path = os.path.join("output", video_folder, f"processed_{video_folder}.mp4")
          thumbnail_path = os.path.join("output", video_folder, "saves", "thumbnail.jpg")
          col1, col2 = st.columns([1, 3])
          with col1:
            st.image(thumbnail_path, width=150)
          with col2:
            st.write(f"**{video_folder}**")
            st.write(f"Final Prediction: {metadata.get('Final Prediction', 'N/A')} ({metadata.get('Confidence', 'N/A')})")
            if st.button(f"View {video_folder}"):
              st.video(processed_video_path)
            if st.button(f"Delete {video_folder}"):
              delete_video(video_folder)
              st.experimental_rerun()


# RUN_________________________________________________________________________________________________________________________________________

if __name__ == '__main__':
  st.set_page_config(
    page_title = 'Harmful Content Detection',
    page_icon = '🕵️‍♂️',
    layout = 'wide',
    initial_sidebar_state = 'collapsed'
  )
  
  main()

