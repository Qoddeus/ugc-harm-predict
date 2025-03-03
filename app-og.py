import cv2
import ffmpeg
import json
import os
import re
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

def extract_frames(video_path, output_dir, resnet_model, class_names, batch_size=32):
  os.makedirs(output_dir, exist_ok=True)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError("Error: Could not open video.")

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

        text_color = (0, 255, 0) if predicted_class_name == "safe" else (0, 0, 255)  # Green for safe, Red for harmful
        bg_color = (255, 255, 255)  # White background
        text = f"Predicted: {predicted_class_name} ({confidence:.4f})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = 10, 40  # Top-left position

        cv2.rectangle(original_frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline), bg_color, -1)
        cv2.putText(original_frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        output_path = os.path.join(output_dir, f"frame_{frame_count - len(batch_frames) + i + 1:04d}.jpg")
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

  input_video = ffmpeg.input(temp_video_path)
  input_audio = ffmpeg.input(audio_path)
  ffmpeg.output(input_video, input_audio, output_video_path, vcodec="h264", acodec="aac", strict="experimental").run()

  os.remove(temp_video_path)

def weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5):
  safe_score = bert_weight * bert_scores['safe'] + resnet_weight * resnet_scores['safe']
  harmful_score = bert_weight * bert_scores['harmful'] + resnet_weight * resnet_scores['harmful']

  if harmful_score > safe_score:
    return "Harmful", harmful_score
  else:
    return "Safe", safe_score

def save_results(output_dir, video_name, results):
  history_file = "saves/processed_videos.json"
  
  # Load existing history if available
  if os.path.exists(history_file):
    with open(history_file, "r") as f:
        history = json.load(f)
  else:
    history = {}

  # Save results for this video
  history[video_name] = results

  with open(history_file, "w") as f:
    json.dump(history, f, indent=4)


# STREAMLIT___________________________________________________________________________________________________________________________________

def main():
  st.title("Harmful Content Detection in Short Videos")  
  tab1, tab2 = st.tabs(["📤 Upload & Process", "📂 View Processed Videos"])
  
  with tab1:
    uploaded_file = st.file_uploader("Upload a short video (MP4 format)", type=["mp4"])
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
      st.subheader("Uploaded Video")
      if uploaded_file is not None:
        video_name = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join("output", video_name)
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"{video_name}.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(video_path)

        if st.button("Process video"):
          st.write(f"Processing video: **{uploaded_file.name}**")
          progress_bar = st.progress(0)
          with st.spinner("Analyzing video, please wait..."):

            audio_path = os.path.join(output_dir, "output_audio.wav")
            extract_audio(video_path, audio_path)
            transcription = transcribe_audio(audio_path, whisper_model)
            text_label, harmful_conf_text, safe_conf_text = classify_text(transcription, bert_model, tokenizer, device)
            progress_bar.progress(20)

            frames_path = os.path.join(output_dir, "processed_frames")
            frame_count, predictions_per_frame, confidence_scores_by_class = extract_frames(video_path, frames_path, resnet_model, class_names)
            average_confidence_by_class = calculate_average_scores(confidence_scores_by_class)
            progress_bar.progress(50)

            harmful_classes = ['nsfw', 'violence']
            safe_classes = ['safe']
            harmful_score_resnet = sum(average_confidence_by_class[class_name] for class_name in harmful_classes) / len(harmful_classes)
            safe_score_resnet = sum(average_confidence_by_class[class_name] for class_name in safe_classes) / len(safe_classes)

            bert_scores = {'safe': safe_conf_text, 'harmful': harmful_conf_text}
            resnet_scores = {'safe': safe_score_resnet, 'harmful': harmful_score_resnet}
            final_prediction, final_confidence = weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5)
            progress_bar.progress(70)

            processed_video_path = os.path.join(output_dir, f"processed_{video_name}.mp4")
            combine_frames_to_video(frames_path, processed_video_path, frame_count, audio_path)
            progress_bar.progress(100)

          with col2:
            st.subheader("Processed Video")
            st.video(processed_video_path)
            col2p1, col2p2 = st.columns(2)

            with col2p1:
              st.write(f"Text Classification: {text_label}")
              st.markdown(f"- Harmful: {harmful_conf_text*100:.2f}%")
              st.markdown(f"- Safe: {safe_conf_text*100:.2f}%")

            with col2p2:
              st.write(f"Video Classification: Harmful")
              st.markdown(f"- Harmful: {harmful_score_resnet*100:.2f}%")
              st.markdown(f"- Safe: {safe_score_resnet*100:.2f}%")

            st.write(f"Final Prediction (Weighted Fusion):")
            st.markdown(f"- Based on the combined analysis, the content is classified as {final_prediction} with a confidence level of {final_confidence*100:.2f}%.")

            with st.expander("Show transcription"):
              st.write(transcription)

            results = {
              "harmful_score_resnet": harmful_score_resnet,
              "safe_score_resnet": safe_score_resnet,
              "bert_scores": bert_scores,
              "safe_conf_text": safe_conf_text,
              "harmful_conf_text": harmful_conf_text,
              "resnet_scores": resnet_scores,
              "final_prediction": final_prediction,
              "final_confidence": final_confidence,
              "transcription": transcription
            }

            save_results(output_dir, video_name, results)

      with tab2:
        st.subheader("Previously Processed Videos")

        history_file = "saves/processed_videos.json"
        if os.path.exists(history_file):
          with open(history_file, "r") as f:
            history = json.load(f)

          selected_video = st.selectbox("Select a processed video:", list(history.keys()))
          if selected_video:
            st.write(f"### Results for {selected_video}")
            results = history[selected_video]

            st.write(f"**Final Prediction:** {results['final_prediction']} ({results['final_confidence']*100:.2f}%)")
            st.write("### Text Classification Results:")
            st.markdown(f"- **Harmful:** {results['harmful_conf_text']*100:.2f}%")
            st.markdown(f"- **Safe:** {results['safe_conf_text']*100:.2f}%")

            st.write("### Video Classification Results:")
            st.markdown(f"- **Harmful:** {results['harmful_score_resnet']*100:.2f}%")
            st.markdown(f"- **Safe:** {results['safe_score_resnet']*100:.2f}%")

            with st.expander("Show transcription"):
              st.write(results["transcription"])
        else:
          st.write("No processed videos found.")


# RUN_________________________________________________________________________________________________________________________________________

if __name__ == '__main__':
  st.set_page_config(
    page_title = 'Harmful Content Detection',
    page_icon = '🕵️‍♂️',
    layout = 'wide',
    initial_sidebar_state = 'collapsed'
  )
  
  main()

