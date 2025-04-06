import os
import json
import streamlit as st
from pytubefix import YouTube
from src.models_load import load_models
from src.proc_audio import extract_audio, transcribe_audio, display_transcription_with_timestamps
from src.proc_text import classify_text
from src.proc_video import extract_frames, combine_frames_to_video
from src.utils import sanitize_filename, is_portrait_video, get_total_frames, calculate_average_scores, weighted_fusion, \
    save_results


# Upload and Process Page


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

# File uploader for local files

# Create two columns with proportions
col1, col2 = st.columns([3, 1])

# Input field with placeholder text inside
youtube_url = col1.text_input("Enter youtube video", placeholder="Paste your YouTube link here", label_visibility="collapsed")

# Add vertical spacing to align the button with the input field
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