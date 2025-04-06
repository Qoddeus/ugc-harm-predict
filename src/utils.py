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

def is_portrait_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        height, width, _ = frame.shape
        cap.release()
        return height > width
    cap.release()
    return False

def create_clickable_blog_post_with_image(title, url, summary, image_url, fixed_width="500px", fixed_height="400px"):
    # Creates a clickable blog post element with an image preview and fixed size.
    st.markdown(
        f"""
        <div style="width: {fixed_width}; height: {fixed_height}; border: 1px solid #e0e0e0; padding: 10px; margin-bottom: 10px; border-radius: 5px; display: flex; flex-direction: column;">
            <a href="{url}" target="_blank" style="text-decoration: none; display: block; flex-grow: 1;">
                <img src="{image_url}" alt="{title}" style="width: 100%; max-height: 200px; object-fit: cover; border-radius: 5px; margin-bottom: 10px;">
                <h3 style="margin-top: 0;">{title}</h3>
            </a>
            <p style="flex-grow: 1; overflow: hidden;">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
     )

blog_posts = [
        {
            "title": "Effects of Inappropriate Content to Minors",
            "url": "https://thedigitalparents.com/online-safety/effects-of-inappropriate-content-to-minors/",
            "summary": "Here we are going to discuss the effects inappropriate content can have on your child and the consequences. Also, why they shouldn’t watch inappropriate content, and how to establish some guidelines for your child.",
            "image_url": "https://thedigitalparents.b-cdn.net/wp-content/uploads/2023/12/pexels-pavel-danilyuk-8763024.jpg",
        },

        {
            "title": "What Is Content Moderation? | Types of Content Moderation, Tools, and more",
            "url": "https://imagga.com/blog/what-is-content-moderation/",
            "summary": "The volume of content generated online every second is staggering. Platforms built around user-generated content face constant challenges in managing inappropriate or illegal text, images, videos, and live streams.",
            "image_url": "https://imagga.com/blog/wp-content/uploads/2021/09/Art6_featured_image-1024x682.jpg",
        },

        {
            "title": "What are the Dangers of Inappropriate Content for Kids?",
            "url": "https://ogymogy.com/blog/dangers-of-inappropriate-content/",
            "summary": "The internet is not just a place, it’s a potentially dangerous territory for everyone, especially children. The threat of encountering inappropriate content is real and immediate, with excessive screen time leading to study distraction, anxiety, depression, and more. Understanding these risks and the potential harm of adult content for kids is not just necessary, it’s vital.",
            "image_url": "https://ogymogy.com/blog/wp-content/uploads/2024/06/what-are-the-danger-of-content-.jpg",
        },

        {
            "title": "Creating a Safe and Respectful Online Community by Understanding the Importance of Content Moderation in Social Media",
            "url": "https://newmediaservices.com.au/the-importance-of-content-moderation-in-social-media/",
            "summary": "Social media is a crucial part of our lives. It’s the first thing we check when we wake up and the last thing we visit before sleeping at night. We use it to engage with friends, share updates, and discover new content.",
            "image_url": "https://newmediaservices.com.au/wp-content/uploads/2024/07/The-Importance-of-Content-Moderation-in-Social-Media.webp",
        },

        {
            "title": "Online harms: protecting children and young people",
            "url": "https://learning.nspcc.org.uk/news/2024/january/online-harms-protecting-children-and-young-people#:~:text=Accessing%20and%20engaging%20with%20harmful%20content%20online%20can,to%20help%20keep%20children%20safe%20from%20online%20harm%3F",
            "summary": "Accessing and engaging with harmful content online can be damaging to children’s wellbeing, leaving them scared and confused. It can also influence their behaviour or what they believe. But what is harmful online content? And what can we do to help keep children safe from online harm?",
            "image_url": "https://learning.nspcc.org.uk/media/qttbeugx/online-harms-blog.jpg",
        },
        {
            "title": "The Vital Role of Content Moderation: A Deep Dive into Online Safety",
            "url": "https://blog.emb.global/vital-role-of-content-moderation/",
            "summary": "Content moderation is crucial and evolving. It involves careful scrutiny, assessment, and possible removal of user-created content. This is to foster a secure and positive online space. This practice is key. It’s vital for our journey through the complex networks of online interaction.",
            "image_url": "https://blog.emb.global/wp-content/uploads/2023/11/Try-Magic-Design-2023-11-28T130131.812-1024x576.webp",
        },
        # Add more articles as needed
    ]
### END
### ________________________________________________________________