### main.py
### user interface app


import os
import json
import cv2
import streamlit as st
from src.models_load import load_models
from src.proc_audio import extract_audio, transcribe_audio, display_transcription_with_timestamps
from src.proc_text import classify_text
from src.proc_video import extract_frames, combine_frames_to_video
from src.utils import sanitize_filename, save_results, weighted_fusion, calculate_average_scores, get_total_frames

def home_page():
    st.title("Harmful Content Detection")
    # st.image("https://cdn.pixabay.com/photo/2018/04/07/08/28/cyber-security-3297280_1280.jpg", width=600)

    st.markdown("""
    ## About this application

    This tool uses AI to detect harmful content in videos by analyzing both the visual elements and spoken words.

    ### Key Features
    - **Audio Analysis**: Transcribes speech and identifies harmful language
    - **Visual Content Analysis**: Detects NSFW content and violence
    - **Combined Analysis**: Provides an overall harmfulness rating
    - **Highlighted Results**: View detailed analysis with visual indicators

    ### How to use
    1. Navigate to the **Upload & Process** page
    2. Upload a video file (MP4 format)
    3. Click "Analyze Video" to analyze the content
    4. View the results and detailed breakdown
    5. Access previous analyses in the **View Processed Videos** page
    """)
    # *Disclaimer: This application is designed for educational and research purposes.*

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
    .portrait-video video {
        max-height: 400px !important;
        margin: 0 auto;
        display: block;
    }
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

    uploaded_file = st.file_uploader("Upload a short video (MP4 format)", type=["mp4"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Video Analysis")
            sanitized_name = sanitize_filename(uploaded_file.name)
            video_name = os.path.splitext(sanitized_name)[0]
            output_dir = os.path.join("output", video_name)
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, f"{video_name}.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            # Don't show original video, will show processed video after completion
            # st.video(video_path)

            process_button = st.button("Analyze Video", type="primary")

            if process_button:
                st.write(f"Processing video: **{uploaded_file.name}**")
                progress_bar = st.progress(0)
                with st.spinner("Analyzing video, please wait..."):
                    # Calculate total work units
                    total_frames = get_total_frames(video_path)
                    total_work = total_frames + 2  # Frames + audio processing + final processing
                    current_work = [0]

                    def update_progress(increment=1):
                        current_work[0] += increment
                        progress_bar.progress(min(current_work[0] / total_work, 1.0))

                    # Extract audio and analyze text
                    audio_path = os.path.join(output_dir, "output_audio.wav")
                    extract_audio(video_path, audio_path)
                    transcription = transcribe_audio(audio_path, whisper_model)
                    text_label, harmful_conf_text, safe_conf_text, highlighted_text = classify_text(transcription, bert_model, tokenizer, device)
                    update_progress()

                    # Process video frames with progress callback
                    frames_path = os.path.join(output_dir, "processed_frames")
                    frame_count, predictions_per_frame, confidence_scores_by_class = extract_frames(
                        video_path, frames_path, resnet_model, class_names,
                        progress_callback=lambda _=None: update_progress()
                    )

                    # Calculate final scores
                    harmful_classes = ['nsfw', 'violence']
                    safe_classes = ['safe']
                    average_confidence_by_class = calculate_average_scores(confidence_scores_by_class)
                    harmful_score_resnet = sum(average_confidence_by_class[class_name] for class_name in harmful_classes) / len(harmful_classes)
                    safe_score_resnet = sum(average_confidence_by_class[class_name] for class_name in safe_classes) / len(safe_classes)

                    bert_scores = {'safe': safe_conf_text, 'harmful': harmful_conf_text}
                    resnet_scores = {'safe': safe_score_resnet, 'harmful': harmful_score_resnet}
                    final_prediction, final_confidence = weighted_fusion(bert_scores, resnet_scores, bert_weight=0.5, resnet_weight=0.5)

                    # Create processed video
                    processed_video_path = os.path.join(output_dir, f"processed_{video_name}.mp4")
                    combine_frames_to_video(frames_path, processed_video_path, frame_count, audio_path)
                    update_progress()

                    # Save results
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

                    st.success("Processing complete!")

                    # Show the processed video after completion
                    if os.path.exists(processed_video_path):
                        if is_portrait_video(processed_video_path):
                            st.markdown('<div class="portrait-video">', unsafe_allow_html=True)
                            st.video(processed_video_path)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.video(processed_video_path)

        # Only show the right column after processing if results exist
        if os.path.exists("saves/processed_videos.json"):
            with open("saves/processed_videos.json", "r") as f:
                history = json.load(f)
                if video_name in history:
                    with col2:
                        st.subheader("Analysis Results")
                        results = history[video_name]

                        st.markdown("### Content Analysis")

                        # Create metrics in a row
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Text Safety", f"{results['safe_conf_text']*100:.1f}%",
                                     f"{(results['safe_conf_text']-0.5)*200:.1f}%" if results['safe_conf_text'] > 0.5 else f"{(results['safe_conf_text']-0.5)*200:.1f}%")
                        with metric_col2:
                            st.metric("Visual Safety", f"{results['safe_score_resnet']*100:.1f}%",
                                     f"{(results['safe_score_resnet']-0.5)*200:.1f}%" if results['safe_score_resnet'] > 0.5 else f"{(results['safe_score_resnet']-0.5)*200:.1f}%")
                        with metric_col3:
                            st.metric("Overall Safety", f"{(1-results['final_confidence'])*100:.1f}%" if results['final_prediction'] == "Harmful" else f"{results['final_confidence']*100:.1f}%",
                                     "Safe" if results['final_prediction'] == "Safe" else "Harmful")

                        # Detailed results in expanders
                        with st.expander("📝 Text Analysis"):
                            st.write("#### Text Classification")
                            st.progress(results['safe_conf_text'], text=f"Safe Content: {results['safe_conf_text']*100:.1f}%")
                            st.progress(results['harmful_conf_text'], text=f"Harmful Content: {results['harmful_conf_text']*100:.1f}%")

                            st.write("#### Highlighted Toxic Content")
                            st.markdown(f"<div style='font-size:16px;'>{results['highlighted_text']}</div>", unsafe_allow_html=True)

                        with st.expander("🎬 Visual Analysis"):
                            st.write("#### Visual Classification")
                            st.progress(results['safe_score_resnet'], text=f"Safe Content: {results['safe_score_resnet']*100:.1f}%")
                            st.progress(results['harmful_score_resnet'], text=f"Harmful Content: {results['harmful_score_resnet']*100:.1f}%")

                        with st.expander("🔊 Transcription"):
                            display_transcription_with_timestamps(results['transcription'], "video_player")

def history_page():
    st.title("View Processed Videos")

    # Add CSS to control video size
    st.markdown("""
    <style>
    .portrait-video video {
        max-height: 400px !important;
        margin: 0 auto;
        display: block;
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

            col1, col2 = st.columns(2)

            with col1:
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

            with col2:
                # Create tabs for different analysis sections
                tab1, tab2, tab3 = st.tabs(["Text", "Visual", "Transcription"])

                with tab1:
                    st.write("### Text Analysis")
                    st.progress(results['safe_conf_text'], text=f"Safe Content: {results['safe_conf_text']*100:.1f}%")
                    st.progress(results['harmful_conf_text'], text=f"Harmful Content: {results['harmful_conf_text']*100:.1f}%")

                    st.write("### Highlighted Toxic Content")
                    st.markdown(f"<div style='font-size:16px;'>{results['highlighted_text']}</div>", unsafe_allow_html=True)

                with tab2:
                    st.write("### Visual Analysis")
                    st.progress(results['safe_score_resnet'], text=f"Safe Content: {results['safe_score_resnet']*100:.1f}%")
                    st.progress(results['harmful_score_resnet'], text=f"Harmful Content: {results['harmful_score_resnet']*100:.1f}%")

                with tab3:
                    st.write("### Transcription")
                    display_transcription_with_timestamps(results['transcription'], "video_player")
    else:
        st.info("No processed videos found. Process some videos in the Upload tab first.")


def main():
    st.set_page_config(
        page_title='Harmful Content Detection',
        page_icon='🕵️‍♂️',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")
        page = st.radio("Go to", ["Home", "Upload & Process", "View Processed Videos"])

        st.markdown("---")
        st.markdown("### About")
        st.markdown("This application helps detect harmful content in videos using AI.")
        st.markdown("Version 1.0")
        st.markdown("---")
        st.markdown("© 2025 Harmful Content Detection")

    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Upload & Process":
        upload_process_page()
    elif page == "View Processed Videos":
        history_page()


if __name__ == '__main__':
    main()


### end
