### src/proc_video.py
### video frame extraction and processing


import cv2
import ffmpeg
import os
import numpy as np
import torch
from src.utils import preprocess_image

def extract_frames(video_path, output_dir, resnet_model, class_names, batch_size=32, progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    batch_frames = [] # for batch processing
    batch_tensors = [] # for batch processing
    predictions_per_frame = []
    confidence_scores_by_class = {class_name: [] for class_name in class_names}
    device = next(resnet_model.parameters()).device

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Update progress if callback is provided
        if progress_callback:
            progress_callback()

        input_tensor = preprocess_image(frame)
        batch_frames.append(frame)  # Store original frame
        batch_tensors.append(input_tensor)  # Store tensor

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
                bg_color = (0, 0, 0)  # Black background
                text = f"Predicted: {predicted_class_name} ({confidence:.4f})"

                # Draw text on frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
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

    cap.release()
    return frame_count, predictions_per_frame, confidence_scores_by_class

def combine_frames_to_video(output_dir, output_video_path, frame_count, audio_path, frame_rate=30):
    temp_video_path = "./temp_video.mp4"
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


### end
