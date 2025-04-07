# pages/2__History.py

import os
import json
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from src.proc_audio import display_transcription_with_timestamps
from src.utils import save_to_pdf, is_portrait_video

# History page

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