# proc_video_sequence.py

import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from src.utils import preprocess_image


def extract_frame_sequences(video_path, output_dir, model, class_names, sequence_length=10, batch_size=1,
                          progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    sequence_buffer = []
    predictions_per_frame = []
    confidence_scores_by_class = {class_name: [] for class_name in class_names}  # This is correct
    violence_sequences = []
    device = next(model.parameters()).device

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def preprocess_frame(frame):
        """Convert OpenCV frame to preprocessed tensor"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return transform(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if progress_callback:
            progress_callback()

        processed_frame = preprocess_frame(frame)
        sequence_buffer.append(processed_frame)

        if len(sequence_buffer) >= sequence_length:
            input_batch = torch.stack(sequence_buffer[-sequence_length:]).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_batch)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                pred = np.argmax(probs, axis=1)[0]
                confidence = float(probs[0][pred])  # Convert to Python float immediately

            predicted_class_name = class_names[pred]
            predictions_per_frame.append((frame_count, predicted_class_name, confidence))

            # This is safe because we initialized it as a list
            confidence_scores_by_class[predicted_class_name].append(confidence)

            if predicted_class_name == "Violence" and confidence > 0.5:
                violence_sequences.append({
                    "start_frame": frame_count - sequence_length + 1,
                    "end_frame": frame_count,
                    "confidence": confidence,
                    "frames": sequence_buffer[-sequence_length:],
                    "type": "violence"
                })

            # Save the current frame with annotation
            output_frame = frame.copy()
            text = f"{predicted_class_name} ({confidence:.2f})"
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)  # Green/Red
            cv2.putText(output_frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Save frame
            output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, output_frame)

    cap.release()

    # Calculate average confidence scores
    avg_confidence = {
        class_name: np.mean(scores) if scores else 0.0
        for class_name, scores in confidence_scores_by_class.items()
    }

    return frame_count, predictions_per_frame, avg_confidence, violence_sequences