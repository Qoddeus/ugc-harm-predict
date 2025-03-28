### main.py


### IMPORTS
### ________________________________________________________________
import cv2
import os
import json
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import webbrowser
from pytubefix import YouTube
from src.models_load import load_models
from src.proc_audio import extract_audio, transcribe_audio, display_transcription_with_timestamps
from src.proc_text import classify_text
from src.proc_video import extract_frames, combine_frames_to_video
from src.utils import sanitize_filename, save_results, weighted_fusion, calculate_average_scores, get_total_frames, save_to_pdf


### FUNCTIONS
### ________________________________________________________________
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

def home_page():
    # Center the image
    st.markdown(
        """
        <style>
        .centered-image { display: block; margin-left: auto; margin-right: auto; width: 200px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.columns(3)[1]:
        st.image("saves/bg.webp")

    st.markdown("""
    ## BuddyGuard: Keeping Online Adventures Safe and Fun.

    This Web application utilizes Machine Learning to detect harmful content in user generated videos by analyzing both the visual elements and spoken words.

    *Disclaimer: This application is designed for educational and research purposes.*
    """)

    # Button that redirects to the Upload & Process page
    if st.button("Get Started", type="primary"):
        st.session_state.page = "Upload & Process"
        st.rerun()

    # Blog posts section added to home_page
    st.subheader("Related Articles")

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
    # Display blog posts in rows of 2
    for i in range(0, len(blog_posts), 2):
        row = st.columns(2)  # Create 2 columns for each row
        for j in range(2):
            if i + j < len(blog_posts):  # Check if there's a post for the column
                with row[j]:
                    post = blog_posts[i + j]
                    create_clickable_blog_post_with_image(post["title"], post["url"], post["summary"],
                                                          post["image_url"])

def is_portrait_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        height, width, _ = frame.shape
        cap.release()
        return height > width
    cap.release()
    return False

def upload_process_page():
    st.title("Upload & Process Video")

    # Add CSS to control video size
    st.markdown("""
    <style>
        .portrait-video video { max-height: 200px !important; margin: 0 auto; display: block; }
        .stImage { margin-bottom: 10px; }
        .stImage img { border-radius: 5px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

    # Load all models
    models = load_models()
    tokenizer = models['tokenizer']
    bert_model = models['bert_model']
    whisper_model = models['whisper_model']
    resnet_model = models['resnet_model']
    class_names = models['class_names']
    device = models['device']

    # Initialize a key in session state to track if a video is uploaded
    if 'uploaded_video' not in st.session_state:
        st.session_state.uploaded_video = None

    # Initialize a key for cancel processing
    if 'cancel_processing' not in st.session_state:
        st.session_state.cancel_processing = False

    # # Add a new input field for YouTube URL
    # youtube_url = st.text_input("Enter YouTube URL:")
    #
    # # Add a button to download the video from the YouTube URL
    # if youtube_url:
    #     if st.button("Download and Process YouTube Video"):
    #         try:
    #             yt = YouTube(youtube_url)
    #             video_stream = yt.streams.filter(file_extension='mp4').first()
    #             if video_stream:
    #                 sanitized_name = sanitize_filename(yt.title)
    #                 video_name = os.path.splitext(sanitized_name)[0]
    #                 output_dir = os.path.join("output", video_name)
    #                 os.makedirs(output_dir, exist_ok=True)
    #                 video_path = os.path.join(output_dir, f"{video_name}.mp4")
    #
    #                 # Download the video
    #                 video_stream.download(output_path=output_dir, filename=f"{video_name}.mp4")
    #
    #                 # Store the downloaded video in session state
    #                 st.session_state.uploaded_video = video_path
    #                 st.session_state.output_dir = output_dir  # Store output_dir in session state
    #                 st.success(f"Video '{yt.title}' downloaded successfully!")
    #             else:
    #                 st.error("No suitable video stream found.")
    #         except Exception as e:
    #             st.error(f"Error downloading YouTube video: {e}")

    # File uploader for local files

    # Create two columns with proportions
    col1, col2 = st.columns([3, 1])

    # Input field for YouTube URL
    youtube_url = col1.text_input("Enter YouTube URL:")

    # Add vertical spacing to align the button with the input field
    col2.markdown("\n")  # Creates vertical space
    if col2.button("Upload YouTube Video"):
        if youtube_url:
            try:
                yt = YouTube(youtube_url)
                video_stream = yt.streams.filter(file_extension='mp4').first()
                if video_stream:
                    sanitized_name = yt.title.replace(" ", "_").replace("/", "_")  # Simple sanitization
                    video_name = os.path.splitext(sanitized_name)[0]
                    output_dir = os.path.join("output", video_name)
                    os.makedirs(output_dir, exist_ok=True)
                    video_path = os.path.join(output_dir, f"{video_name}.mp4")

                    # Download the video
                    video_stream.download(output_path=output_dir, filename=f"{video_name}.mp4")

                    # Store the downloaded video in session state
                    st.session_state.uploaded_video = video_path
                    st.session_state.output_dir = output_dir
                    st.success(f"Video '{yt.title}' downloaded successfully!")
                else:
                    st.error("No suitable video stream found.")
            except Exception as e:
                st.error(f"Error downloading YouTube video: {e}")

    uploaded_file = st.file_uploader("Or upload a video file", type=["mp4", "avi", "mov", "webm", "mpg"])

    if uploaded_file is not None:
        # Store the uploaded file in session state
        st.session_state.uploaded_video = uploaded_file

    # Process the uploaded video if it exists in session state
    if st.session_state.uploaded_video is not None:
        uploaded_file = st.session_state.uploaded_video
        if isinstance(uploaded_file, str):  # If it's a path from YouTube download
            video_path = uploaded_file
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = st.session_state.output_dir  # Retrieve output_dir from session state
        else:  # If it's an uploaded file
            sanitized_name = sanitize_filename(uploaded_file.name)
            video_name = os.path.splitext(sanitized_name)[0]
            output_dir = os.path.join("output", video_name)
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, f"{video_name}.mp4")

            # Only write the file if it doesn't exist already
            if not os.path.exists(video_path):
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

        col1, col2, col3 = st.columns(3)

        with col2:
            if os.path.exists(video_path):
                st.subheader("Original Video")
                if is_portrait_video(video_path):
                    st.markdown('<div class="portrait-video">', unsafe_allow_html=True)
                    st.video(video_path)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.video(video_path)

        process_button = st.button("Analyze Video", type="primary", use_container_width=True)

        if process_button:
            # Reset cancel flag when starting a new process
            st.session_state.cancel_processing = False

            progress_bar = st.progress(0)
            processing_status = st.empty()  # Placeholder for status messages

            with processing_status.container():
                st.spinner(f"Analyzing video **{os.path.basename(video_path)}**, please wait...")

                # Cancel button is only useful during processing
                if st.button("Cancel Process", type="secondary", use_container_width=True):
                    st.session_state.cancel_processing = True
                    st.warning("Cancelling the process. Please wait...")

                # Calculate total work units
                total_frames = get_total_frames(video_path)
                total_work = total_frames + 2  # Frames + audio processing + final processing
                current_work = [0]

                def update_progress(increment=1):
                    current_work[0] += increment
                    progress_percentage = min(current_work[0] / total_work, 1.0)
                    progress_bar.progress(progress_percentage)

                    # Check for cancel request
                    if st.session_state.cancel_processing:
                        st.warning("Process cancelled by user")
                        st.stop()  # This stops the current execution

                # Extract audio and analyze text
                audio_path = os.path.join(output_dir, "output_audio.wav")
                extract_audio(video_path, audio_path)

                # Check for cancel request before continuing
                if st.session_state.cancel_processing:
                    st.warning("Process cancelled by user")
                    st.stop()

                transcription = transcribe_audio(audio_path, whisper_model)

                # Check for cancel request before continuing
                if st.session_state.cancel_processing:
                    st.warning("Process cancelled by user")
                    st.stop()

                text_label, harmful_conf_text, safe_conf_text, highlighted_text = classify_text(transcription, bert_model, tokenizer, device)
                update_progress()

                # Process video frames with progress callback and cancel check
                frames_path = os.path.join(output_dir, "processed_frames")

                def progress_with_cancel_check():
                    if st.session_state.cancel_processing:
                        st.warning("Process cancelled by user")
                        st.stop()
                    update_progress()

                frame_count, predictions_per_frame, confidence_scores_by_class, harmful_frames = extract_frames(
                    video_path, frames_path, resnet_model, class_names,
                    progress_callback=progress_with_cancel_check
                )

                # Check for cancel request before continuing
                if st.session_state.cancel_processing:
                    st.warning("Process cancelled by user")
                    st.stop()

                # Calculate final scores
                harmful_classes = ['nsfw', 'violence']
                safe_classes = ['safe']
                average_confidence_by_class = calculate_average_scores(confidence_scores_by_class)
                harmful_score_resnet = sum(average_confidence_by_class[class_name] for class_name in harmful_classes) / len(harmful_classes)
                safe_score_resnet = sum(average_confidence_by_class[class_name] for class_name in safe_classes) / len(safe_classes)

                bert_scores = {'safe': safe_conf_text, 'harmful': harmful_conf_text}
                resnet_scores = {'safe': safe_score_resnet, 'harmful': harmful_score_resnet}
                final_prediction, final_confidence = weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5)

                # Check for cancel request before continuing
                if st.session_state.cancel_processing:
                    st.warning("Process cancelled by user")
                    st.stop()

                # Create processed video
                video_basename = os.path.basename(video_path)
                video_name_no_ext = os.path.splitext(video_basename)[0]
                processed_video_path = os.path.join(output_dir, f"processed_{video_name_no_ext}.mp4")
                combine_frames_to_video(frames_path, processed_video_path, frame_count, audio_path)
                update_progress()

                # Save results
                results = {
                    "harmful_score_resnet": harmful_score_resnet,
                    "safe_score_resnet": safe_score_resnet,
                    "resnet_scores": resnet_scores,
                    "harmful_conf_text": harmful_conf_text,
                    "safe_conf_text": safe_conf_text,
                    "bert_scores": bert_scores,
                    "final_prediction": final_prediction,
                    "final_confidence": final_confidence,
                    "transcription": transcription,
                    "highlighted_text": highlighted_text,
                    "harmful_frames": harmful_frames,  # Updated from nsfw_frames to harmful_frames
                    "nsfw_frames": [frame for frame in harmful_frames if frame.get('type') == 'nsfw' or 'type' not in frame],  # For backward compatibility
                    "frames_path": frames_path
                }
                save_results(output_dir, video_name, results)

                st.success("Processing complete!")

            col1, col2, col3 = st.columns(3)

            with col2:
                st.subheader("Analyzed Video")
                if os.path.exists(processed_video_path):
                    if is_portrait_video(processed_video_path):
                        st.markdown('<div class="portrait-video">', unsafe_allow_html=True)
                        st.video(processed_video_path)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.video(processed_video_path)

        if os.path.exists("saves/processed_videos.json"):
            with open("saves/processed_videos.json", "r") as f:
                history = json.load(f)
                if video_name in history:
                    st.subheader("Analysis Results")
                    results = history[video_name]

                    st.markdown("#### Content Analysis")

                    # Create metrics in a row
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Text Harmful", f"{(1-results['safe_conf_text'])*100:.2f}%",
                                 f"{(0.5-results['safe_conf_text'])*200:.2f}%" if results['safe_conf_text'] < 0.5 else f"{(0.5-results['safe_conf_text'])*200:.2f}%")
                    with metric_col2:
                        st.metric("Visual Harmful", f"{(1-results['safe_score_resnet'])*100:.2f}%",
                                 f"{(0.5-results['safe_score_resnet'])*200:.2f}%" if results['safe_score_resnet'] < 0.5 else f"{(0.5-results['safe_score_resnet'])*200:.2f}%")
                    with metric_col3:
                        st.metric("Overall Harmful", f"{results['final_confidence']*100:.2f}%" if results['final_prediction'] == "Harmful" else f"{(1-results['final_confidence'])*100:.2f}%",
                                 "Harmful" if results['final_prediction'] == "Harmful" else "Safe")

                    # Detailed results in expanders
                    with st.expander("📝 Text Analysis"):
                        st.write("#### Text Classification")
                        st.progress(results['safe_conf_text'], text=f"Safe Content: {results['safe_conf_text']*100:.2f}%")
                        st.progress(results['harmful_conf_text'], text=f"Harmful Content: {results['harmful_conf_text']*100:.2f}%")

                        st.write("#### Highlighted Toxic Content")
                        st.markdown(f"<div style='font-size:16px;'>{results['highlighted_text']}</div>", unsafe_allow_html=True)

                    with st.expander("🎬 Visual Analysis"):
                        st.write("#### Visual Classification")
                        st.progress(results['safe_score_resnet'], text=f"Safe Content: {results['safe_score_resnet']*100:.2f}%")
                        st.progress(results['harmful_score_resnet'], text=f"Harmful Content: {results['harmful_score_resnet']*100:.2f}%")

                        # Display harmful frames if available
                        if ('harmful_frames' in results and results['harmful_frames']) or ('nsfw_frames' in results and results['nsfw_frames']):
                            st.write("#### Detected Harmful Frames")

                            # Get the frames from either harmful_frames or nsfw_frames (for backward compatibility)
                            frames_to_display = results.get('harmful_frames', results.get('nsfw_frames', []))

                            # Check if frames have type attribute
                            has_type_attribute = any('type' in frame for frame in frames_to_display)

                            if has_type_attribute:
                                # Group frames by type
                                nsfw_frames = [frame for frame in frames_to_display if frame.get('type') == 'nsfw']
                                violence_frames = [frame for frame in frames_to_display if frame.get('type') == 'violence']

                                # Handle frames without type (legacy data) - assume they are NSFW
                                untyped_frames = [frame for frame in frames_to_display if 'type' not in frame]
                                if untyped_frames:
                                    nsfw_frames.extend(untyped_frames)

                                # Display NSFW frames if any
                                if nsfw_frames:
                                    st.write("##### NSFW Content")
                                    cols_per_row = 3

                                    # Process frames in groups of 3
                                    for i in range(0, len(nsfw_frames), cols_per_row):
                                        # Create a row with 3 columns
                                        cols = st.columns(cols_per_row)

                                        # Fill each column with a frame
                                        for col_idx in range(cols_per_row):
                                            frame_idx = i + col_idx

                                            # Check if we still have frames to display
                                            if frame_idx < len(nsfw_frames):
                                                frame = nsfw_frames[frame_idx]
                                                frame_path = os.path.join(results.get('frames_path', ''), frame['path'])

                                                # Display frame in the respective column
                                                with cols[col_idx]:
                                                    if os.path.exists(frame_path):
                                                        st.image(
                                                            frame_path,
                                                            caption=f"Frame {frame['frame_number']} - Confidence: {frame['confidence']:.2f}",
                                                            use_container_width=True
                                                        )
                                                    else:
                                                        st.warning(f"Frame {frame['frame_number']} not found")

                                # Display violence frames if any
                                if violence_frames:
                                    st.write("##### Violence Content")
                                    cols_per_row = 3

                                    # Process frames in groups of 3
                                    for i in range(0, len(violence_frames), cols_per_row):
                                        # Create a row with 3 columns
                                        cols = st.columns(cols_per_row)

                                        # Fill each column with a frame
                                        for col_idx in range(cols_per_row):
                                            frame_idx = i + col_idx

                                            # Check if we still have frames to display
                                            if frame_idx < len(violence_frames):
                                                frame = violence_frames[frame_idx]
                                                frame_path = os.path.join(results.get('frames_path', ''), frame['path'])

                                                # Display frame in the respective column
                                                with cols[col_idx]:
                                                    if os.path.exists(frame_path):
                                                        st.image(
                                                            frame_path,
                                                            caption=f"Frame {frame['frame_number']} - Confidence: {frame['confidence']:.2f}",
                                                            use_container_width=True
                                                        )
                                                    else:
                                                        st.warning(f"Frame {frame['frame_number']} not found")
                            else:
                                # Display all frames without categorization (legacy approach)
                                st.write("##### Harmful Content")
                                cols_per_row = 3

                                # Process frames in groups of 3
                                for i in range(0, len(frames_to_display), cols_per_row):
                                    # Create a row with 3 columns
                                    cols = st.columns(cols_per_row)

                                    # Fill each column with a frame
                                    for col_idx in range(cols_per_row):
                                        frame_idx = i + col_idx

                                        # Check if we still have frames to display
                                        if frame_idx < len(frames_to_display):
                                            frame = frames_to_display[frame_idx]
                                            frame_path = os.path.join(results.get('frames_path', ''), frame['path'])

                                            # Display frame in the respective column
                                            with cols[col_idx]:
                                                if os.path.exists(frame_path):
                                                    st.image(
                                                        frame_path,
                                                        caption=f"Frame {frame['frame_number']} - Confidence: {frame['confidence']:.2f}",
                                                        use_container_width=True
                                                    )
                                                else:
                                                    st.warning(f"Frame {frame['frame_number']} not found")

                            if not frames_to_display:
                                st.info("No harmful frames detected in this video")
                        else:
                            st.info("No harmful frames detected in this video")

                    with st.expander("🔊 Transcription"):
                        display_transcription_with_timestamps(results['transcription'], "video_player")

                    # Clear button to reset the UI
                    if st.button("Clear", type="secondary"):
                        st.session_state.uploaded_video = None
                        st.session_state.cancel_processing = False

def history_page():
    st.title("View Processed Videos")

    # Add CSS to control video size
    st.markdown("""
    <style>
    .portrait-video video {
        max-height: 200px !important;
        margin: 0 auto;
        display: block;
    }
    .stImage {
    margin-bottom: 10px;
    }
    .stImage img {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

    history_file = "saves/processed_videos.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

        if not history:
            st.info("No processed videos found. Process some videos in the Upload tab first.")
            return

        selected_video = st.selectbox("Select a processed video:", list(history.keys()))
        if selected_video:
            results = history[selected_video]
            video_name = os.path.splitext(selected_video)[0]

            col1, col2, col3 = st.columns(3)

            with col2:
                video_path = os.path.join("output", video_name, f"processed_{video_name}.mp4")
                if os.path.exists(video_path):
                    if is_portrait_video(video_path):
                        st.markdown('<div class="portrait-video">', unsafe_allow_html=True)
                        st.video(video_path)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.video(video_path)
                else:
                    st.warning("Processed video not found!")

            # Final verdict with color coding
            verdict_color = "red" if results['final_prediction'] == "Harmful" else "green"
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: {verdict_color}; color: white; text-align: center;'>
            <h3>VERDICT: {results['final_prediction'].upper()}</h3>
            <p>Confidence: {results['final_confidence']*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Create tabs for different analysis sections
            tab1, tab2, tab3 = st.tabs(["Text", "Visual", "Transcription"])

            with tab1:
                st.write("### Text Analysis")
                st.progress(results['safe_conf_text'], text=f"Safe Content: {results['safe_conf_text']*100:.2f}%")
                st.progress(results['harmful_conf_text'], text=f"Harmful Content: {results['harmful_conf_text']*100:.2f}%")

                st.write("### Highlighted Toxic Content")
                st.markdown(f"<div style='font-size:16px;'>{results['highlighted_text']}</div>", unsafe_allow_html=True)

            with tab2:
                st.write("### Visual Analysis")
                st.progress(results['safe_score_resnet'], text=f"Safe Content: {results['safe_score_resnet']*100:.2f}%")
                st.progress(results['harmful_score_resnet'], text=f"Harmful Content: {results['harmful_score_resnet']*100:.2f}%")

                # Display harmful frames if available
                if ('harmful_frames' in results and results['harmful_frames']) or ('nsfw_frames' in results and results['nsfw_frames']):
                    st.write("#### Detected Harmful Frames")

                    # Get the frames from either harmful_frames or nsfw_frames (for backward compatibility)
                    frames_to_display = results.get('harmful_frames', results.get('nsfw_frames', []))

                    # Check if frames have type attribute
                    has_type_attribute = any('type' in frame for frame in frames_to_display)

                    if has_type_attribute:
                        # Group frames by type
                        nsfw_frames = [frame for frame in frames_to_display if frame.get('type') == 'nsfw']
                        violence_frames = [frame for frame in frames_to_display if frame.get('type') == 'violence']

                        # Handle frames without type (legacy data) - assume they are NSFW
                        untyped_frames = [frame for frame in frames_to_display if 'type' not in frame]
                        if untyped_frames:
                            nsfw_frames.extend(untyped_frames)

                        # Display NSFW frames if any
                        if nsfw_frames:
                            st.write("##### NSFW Content")
                            cols_per_row = 3

                            # Process frames in groups of 3
                            for i in range(0, len(nsfw_frames), cols_per_row):
                                # Create a row with 3 columns
                                cols = st.columns(cols_per_row)

                                # Fill each column with a frame
                                for col_idx in range(cols_per_row):
                                    frame_idx = i + col_idx

                                    # Check if we still have frames to display
                                    if frame_idx < len(nsfw_frames):
                                        frame = nsfw_frames[frame_idx]
                                        frame_path = os.path.join(results.get('frames_path', ''), frame['path'])

                                        # Display frame in the respective column
                                        with cols[col_idx]:
                                            if os.path.exists(frame_path):
                                                st.image(
                                                    frame_path,
                                                    caption=f"Frame {frame['frame_number']} - Confidence: {frame['confidence']:.2f}",
                                                    use_container_width=True
                                                )
                                            else:
                                                st.warning(f"Frame {frame['frame_number']} not found")

                        # Display violence frames if any
                        if violence_frames:
                            st.write("##### Violence Content")
                            cols_per_row = 3

                            # Process frames in groups of 3
                            for i in range(0, len(violence_frames), cols_per_row):
                                # Create a row with 3 columns
                                cols = st.columns(cols_per_row)

                                # Fill each column with a frame
                                for col_idx in range(cols_per_row):
                                    frame_idx = i + col_idx

                                    # Check if we still have frames to display
                                    if frame_idx < len(violence_frames):
                                        frame = violence_frames[frame_idx]
                                        frame_path = os.path.join(results.get('frames_path', ''), frame['path'])

                                        # Display frame in the respective column
                                        with cols[col_idx]:
                                            if os.path.exists(frame_path):
                                                st.image(
                                                    frame_path,
                                                    caption=f"Frame {frame['frame_number']} - Confidence: {frame['confidence']:.2f}",
                                                    use_container_width=True
                                                )
                                            else:
                                                st.warning(f"Frame {frame['frame_number']} not found")
                    else:
                        # Display all frames without categorization (legacy approach)
                        st.write("##### Harmful Content")
                        cols_per_row = 3

                        # Process frames in groups of 3
                        for i in range(0, len(frames_to_display), cols_per_row):
                            # Create a row with 3 columns
                            cols = st.columns(cols_per_row)

                            # Fill each column with a frame
                            for col_idx in range(cols_per_row):
                                frame_idx = i + col_idx

                                # Check if we still have frames to display
                                if frame_idx < len(frames_to_display):
                                    frame = frames_to_display[frame_idx]
                                    frame_path = os.path.join(results.get('frames_path', ''), frame['path'])

                                    # Display frame in the respective column
                                    with cols[col_idx]:
                                        if os.path.exists(frame_path):
                                            st.image(
                                                frame_path,
                                                caption=f"Frame {frame['frame_number']} - Confidence: {frame['confidence']:.2f}",
                                                use_container_width=True
                                            )
                                        else:
                                            st.warning(f"Frame {frame['frame_number']} not found")

                    if not frames_to_display:
                        st.info("No harmful frames detected in this video")
                else:
                    st.info("No harmful frames detected in this video")

            with tab3:
                st.write("### Transcription")
                display_transcription_with_timestamps(results['transcription'], "video_player")

            st.markdown('---')

            if st.button("Generate PDF Report", type="primary"):
                try:
                    pdf_path = save_to_pdf(video_name, history_file)
                    rel_path = os.path.relpath(pdf_path, start=os.getcwd())
                    # Create a downloadable link
                    st.success(f"PDF report generated successfully!")
                    with st.expander("View PDF"):
                        pdf_viewer(pdf_path)
                except (FileNotFoundError, ValueError) as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("No processed videos found. Process some videos in the Upload tab first.")


def main():
    st.set_page_config(
        page_title='Harmful Content Detection',
        page_icon='🕵️‍♂️',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )

    # Initialize session state for page navigation if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        selected_page = st.radio("Go to", ["Home", "Upload & Process", "View Processed Videos"])

        # Update the session state when user selects from radio buttons
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()

        st.markdown('''
            ---

            ### Key Features
            - **Audio Analysis**: Transcribes speech and identifies harmful language
            - **Visual Content Analysis**: Detects NSFW content and violence
            - **Combined Analysis**: Provides an overall harmfulness rating
            - **Highlighted Results**: View detailed analysis with visual indicators

            ---

            ### How to use
            1. Navigate to the **Upload & Process** page
            2. Upload a video file
            3. Click "Analyze Video" to analyze the content
            4. View the results and detailed breakdown
            5. Access previous analyses in the **View Processed Videos** page

            ---

            Version 0.1.1.0

            © 2025 BuddyGuard
        ''')

    # Display the selected page based on session state
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Upload & Process":
        upload_process_page()
    elif st.session_state.page == "View Processed Videos":
        history_page()


if __name__ == '__main__':
    main()


### END
### ________________________________________________________________
