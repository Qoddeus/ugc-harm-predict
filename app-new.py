import cv2
import ffmpeg
import json
import os
import re
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from datetime import datetime
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, pipeline


# PREP________________________________________________________________________________________________________________________________________

# Define the new attention-based classifier
class BertClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        attentions = outputs.attentions  # Extract attention scores
        return logits, attentions

class ResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model_path = "models/bert.pth"
resnet_model_path = "models/resnet50.pth"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Use base tokenizer
bert_model = BertClassifier().to(device) # Load BERT model with classification head
bert_model.load_state_dict(torch.load(bert_model_path, map_location=device)) # Load the trained weights
bert_model.eval()

whisper_model = pipeline("automatic-speech-recognition", "openai/whisper-tiny.en", torch_dtype=torch.float16, device=device)

resnet_model = ResNetModel()
state_dict = torch.load(resnet_model_path, map_location=device)
resnet_model.load_state_dict(state_dict, strict=False)
resnet_model.eval()
class_names = ['nsfw', 'safe', 'violence']


# FUNCTIONS___________________________________________________________________________________________________________________________________

def extract_audio(video_path, audio_path):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16000')
    ffmpeg.run(stream)

def transcribe_audio(audio_path, whisper_model):
    result = whisper_model(audio_path, return_timestamps=True)

    transcribed_segments = []
    for segment in result["chunks"]:
        start_time = segment["timestamp"][0]  # Start timestamp in seconds
        text = segment["text"]

        transcribed_segments.append({
            "start_time": start_time,
            "text": text
        })

    return transcribed_segments

def classify_text(transcription, bert_model, tokenizer, device):
    bert_model.eval()
    text = " ".join(segment["text"] for segment in transcription)
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits, attentions = bert_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    probs = F.softmax(logits, dim=-1)
    harmful_confidence = probs[0][1].item()
    safe_confidence = probs[0][0].item()
    label = "Harmful" if harmful_confidence > safe_confidence else "Safe"

    highlighted_text = highlight_toxic_words(text, inputs, attentions)

    return label, harmful_confidence, safe_confidence, highlighted_text

def highlight_toxic_words(text, inputs, attentions):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert token IDs to words
    attention_scores = attentions[-1].mean(dim=1).mean(dim=0).cpu().numpy()
    attention_scores = attention_scores.reshape(-1)[:len(tokens)]  # Ensure correct token length

    highlighted_text = []
    merged_tokens = []
    merged_attention = []

    for token, score in zip(tokens, attention_scores):
        if token.startswith("##"):
            merged_tokens[-1] += token[2:]  # Merge subwords
            merged_attention[-1] = np.maximum(merged_attention[-1], score)  # Keep max importance

        else:
            merged_tokens.append(token)
            merged_attention.append(score)

    # Determine dynamic thresholds
    threshold_high = np.percentile(merged_attention, 80)
    threshold_mid = np.percentile(merged_attention, 60)

    for token, score in zip(merged_tokens, merged_attention):
        if token in {".", ",", "!", "?", "'", '"', "’"}:
            highlighted_text.append(token)
            continue

        # Force highlight known toxic words
        if token.lower() in {"fuck", "bitch", "idiot", "stupid"}:
            highlighted_text.append(f"<span style='background-color:rgba(255, 0, 0, 1)'>{token}</span>")
            continue

        if np.any(score > threshold_high):
            score_scalar = np.max(score)  # Get the max score
            color = f"rgba(255, 0, 0, {max(score_scalar, 1):.2f})"  # High toxicity (red)
        elif score > threshold_mid:
            score_scalar = np.max(score)  # Get the max score
            color = f"rgba(255, 255, 0, {max(score_scalar, 1):.2f})"  # Mild toxicity (yellow)
        else:
            color = "white"

        highlighted_text.append(f"<span style='color:{color}'>{token}</span>")

    return " ".join(highlighted_text)

def display_transcription_with_timestamps(transcription, video_id):
    formatted_transcription = ""

    for segment in transcription:
        start_time = segment["start_time"]
        text = segment["text"]

        # Convert seconds to MM:SS format
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        formatted_time = f"{minutes:02}:{seconds:02}"

        # Add a clickable span for seeking
        formatted_transcription += (
            f"<span style='cursor:pointer; color:cyan; text-decoration:underline;' "
            f"onclick='seekVideo(\"{video_id}\", {start_time})'>{formatted_time}</span> {text}<br>"
        )

    # Inject JavaScript for seeking
    st.markdown("""
        <script>
        function seekVideo(video_id, time) {
            var vid = document.getElementById(video_id);
            if (vid) {
                vid.currentTime = time;
                vid.play();
            }
        }
        </script>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:18px;'>{formatted_transcription}</div>", unsafe_allow_html=True)

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

def extract_frames(video_path, output_dir, resnet_model, class_names, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    batch_frames = [] # for batch processing, comment if not needed
    batch_tensors = [] # for batch processing, comment if not needed
    predictions_per_frame = []
    confidence_scores_by_class = {class_name: [] for class_name in class_names}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        input_tensor = preprocess_image(frame)
        # input_tensor = input_tensor.unsqueeze(0).to(device) # for single frame processing, currently using batch processing
        batch_frames.append(frame)  # for batch processing, comment if not needed | Store original frame
        batch_tensors.append(input_tensor)  # for batch processing, comment if not needed | Store tensor

        # If batch is full or video ends, process batch
        if len(batch_frames) == batch_size or not ret:
            batch_input = torch.stack(batch_tensors).to(device)  # Stack batch tensors

            # Run batch inference
            with torch.no_grad():
                outputs = resnet_model(batch_input)  # Forward pass
                predictions = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Apply softmax

            # Process results for each frame in the batch
            for i, (frame, prediction) in enumerate(zip(batch_frames, predictions)):
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = prediction[predicted_class_index]

                # Color coding: Green for "safe", Red for "harmful"
                text_color = (0, 255, 0) if predicted_class_name == "safe" else (0, 0, 255)
                bg_color = (255, 255, 255)  # White background
                text = f"Predicted: {predicted_class_name} ({confidence:.4f})"

                # Draw text on frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = 10, 40  # Top-left position

                cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline), bg_color, -1)
                cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                # Save the frame with overlay
                output_path = os.path.join(output_dir, f"frame_{frame_count - len(batch_frames) + i + 1:04d}.jpg")
                cv2.imwrite(output_path, frame)

                # Store predictions
                predictions_per_frame.append((frame_count - len(batch_frames) + i + 1, predicted_class_name, confidence))
                confidence_scores_by_class[predicted_class_name].append(confidence)

            # Reset batch
            batch_frames = []
            batch_tensors = []

        # for single frame processing
        # with torch.no_grad():
        #     output = resnet_model(input_tensor)
        #     prediction = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        #
        # predicted_class_index = np.argmax(prediction)
        # predicted_class_name = class_names[predicted_class_index]
        # confidence = prediction[predicted_class_index]
        #
        # # Color coding: Green for "safe", Red for "harmful"
        # text_color = (0, 255, 0) if predicted_class_name == "safe" else (0, 0, 255)
        # bg_color = (255, 255, 255)  # White background
        # text = f"Predicted: {predicted_class_name} ({confidence:.4f})"
        #
        # # Draw text on frame
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # thickness = 2
        # (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # x, y = 10, 40  # Top-left position
        #
        # cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline), bg_color, -1)
        # cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        #
        # output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        # cv2.imwrite(output_path, frame)
        # predictions_per_frame.append((frame_count, predicted_class_name, confidence))
        # confidence_scores_by_class[predicted_class_name].append(confidence)

    cap.release()
    return frame_count, predictions_per_frame, confidence_scores_by_class

def calculate_average_scores(confidence_scores_by_class):
    return {class_name: (sum(scores) / len(scores) if scores else 0.0) for class_name, scores in confidence_scores_by_class.items()}

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

class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
        json.dump(history, f, cls=NumpyTypeEncoder, indent=4)

def weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5):
    safe_score = bert_weight * bert_scores['safe'] + resnet_weight * resnet_scores['safe']
    harmful_score = bert_weight * bert_scores['harmful'] + resnet_weight * resnet_scores['harmful']

    if harmful_score > safe_score:
        return "Harmful", harmful_score
    else:
        return "Safe", safe_score

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
                sanitized_name = sanitize_filename(uploaded_file.name)
                video_name = os.path.splitext(sanitized_name)[0]
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
                        text_label, harmful_conf_text, safe_conf_text, highlighted_text = classify_text(transcription, bert_model, tokenizer, device)
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

                        with st.expander("Show transcription"):
                            display_transcription_with_timestamps(transcription, "video_player")

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

                        with st.expander("Toxic Text Highlighting"):
                            st.markdown(f"<div style='font-size:18px;'>{highlighted_text}</div>", unsafe_allow_html=True)

                    results = {
                        "harmful_score_resnet": harmful_score_resnet,
                        "safe_score_resnet": safe_score_resnet,
                        "bert_scores": bert_scores,
                        "safe_conf_text": safe_conf_text,
                        "harmful_conf_text": harmful_conf_text,
                        "resnet_scores": resnet_scores,
                        "final_prediction": final_prediction,
                        "final_confidence": final_confidence,
                        "transcription": transcription,
                        "highlighted_text": highlighted_text
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
                        results = history[selected_video]

                        col3, col4 = st.columns(2)

                        with col3:
                            st.write(f"### Results for {selected_video}")
                            video_name = os.path.splitext(selected_video)[0]  # Remove file extension
                            video_path = os.path.join("output", video_name, f"processed_{video_name}.mp4")
                            if os.path.exists(video_path):
                                st.video(video_path)
                            else:
                                st.warning("Processed video not found!")

                        with col4:
                            st.write("### **Classification Results**")
                            col4p1, col4p2 = st.columns(2)  # Two sub-columns
                            with col4p1:
                                st.write("#### Text:")
                                st.markdown(f"- **Harmful:** {results['harmful_conf_text']*100:.2f}%")
                                st.markdown(f"- **Safe:** {results['safe_conf_text']*100:.2f}%")

                            with col4p2:
                                st.write("#### Visual:")
                                st.markdown(f"- **Harmful:** {results['harmful_score_resnet']*100:.2f}%")
                                st.markdown(f"- **Safe:** {results['safe_score_resnet']*100:.2f}%")

                            st.write("#### Final Prediction:")
                            st.markdown(f"- **{results['final_prediction']}:** {results['final_confidence']*100:.2f}%")

                            st.write("---")
                            st.write("### **Transcription**")
                            display_transcription_with_timestamps(results['transcription'], "video_player")

                            st.write("---")
                            st.write("#### **Highlighted Toxic Texts**")
                            st.markdown(f"<div style='font-size:18px;'>{results['highlighted_text']}</div>", unsafe_allow_html=True)


# RUN_________________________________________________________________________________________________________________________________________

if __name__ == '__main__':
    st.set_page_config(
        page_title = 'Harmful Content Detection',
        page_icon = '🕵️‍♂️',
        layout = 'wide',
        initial_sidebar_state = 'collapsed'
    )

    main()
