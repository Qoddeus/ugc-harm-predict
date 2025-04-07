### src/proc_video.py


### IMPORTS
### ________________________________________________________________
import cv2
import ffmpeg
import os
import numpy as np
import torch
from src.utils import preprocess_image, select_diverse_frames

def extract_frames(video_path, output_dir, resnet_model, class_names, batch_size=32, progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    batch_frames = [] # for batch processing
    batch_tensors = [] # for batch processing
    predictions_per_frame = []
    confidence_scores_by_class = {class_name: [] for class_name in class_names}
    nsfw_frames = []  # Store frame numbers with NSFW content
    violence_frames = []  # Store frame numbers with violence content
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
                frame_index = frame_count - len(batch_frames) + i + 1
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = prediction[predicted_class_index]

                # Store NSFW frames
                if predicted_class_name == "nsfw" and confidence > 0.3:  # Threshold to avoid false positives
                    nsfw_frames.append({
                        "frame_number": frame_index,
                        "confidence": float(confidence),
                        "path": f"frame_{frame_index:04d}.jpg",
                        "type": "nsfw"
                    })

                # Store violence frames
                if predicted_class_name == "violence" and confidence > 0.3:  # Same threshold for consistency
                    violence_frames.append({
                        "frame_number": frame_index,
                        "confidence": float(confidence),
                        "path": f"frame_{frame_index:04d}.jpg",
                        "type": "violence"
                    })

                # Color coding: Green for "safe", Red for "harmful" (NSFW), Orange for "violence"
                if predicted_class_name == "safe":
                    text_color = (0, 255, 0)  # Green
                elif predicted_class_name == "nsfw":
                    text_color = (0, 0, 255)  # Red
                else:  # violence
                    text_color = (0, 165, 255)  # Orange in BGR

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

    # Select a subset of diverse NSFW frames if there are many
    selected_nsfw_frames = select_diverse_frames(nsfw_frames, max_frames=5)

    # Select a subset of diverse violence frames if there are many
    selected_violence_frames = select_diverse_frames(violence_frames, max_frames=5)

    # Combine both types of harmful frames
    harmful_frames = selected_nsfw_frames + selected_violence_frames

    # Sort by frame number for chronological display
    harmful_frames.sort(key=lambda x: x["frame_number"])

    return frame_count, predictions_per_frame, confidence_scores_by_class, harmful_frames

def combine_frames_to_video(output_dir, output_video_path, frame_count, audio_path, frame_rate=30):
    temp_video_path = "./temp_video.mp4"

    # 1. Find the first valid frame to get dimensions
    frame_example = None
    for i in range(1, frame_count + 1):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        if os.path.exists(frame_path):
            frame_example = cv2.imread(frame_path)
            if frame_example is not None:
                break

    # 2. Error if no frames found
    if frame_example is None:
        available_frames = [f for f in os.listdir(output_dir) if f.startswith('frame_')]
        raise ValueError(
            f"No valid frames found in {output_dir}. "
            f"Expected {frame_count} frames. Found: {len(available_frames)}"
        )

    # 3. Get dimensions from first valid frame
    height, width, _ = frame_example.shape

    # 4. Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (width, height))

    # 5. Write all valid frames
    for i in range(1, frame_count + 1):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Warning: Could not read frame {i} (corrupted file)")
        else:
            print(f"Warning: Missing frame {i}")

    video_writer.release()

    # 6. Combine with audio (existing code)
    input_video = ffmpeg.input(temp_video_path)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.output(input_video, input_audio, output_video_path,
                  vcodec='h264', acodec='aac').run()

    os.remove(temp_video_path)

### END
### ________________________________________________________________
