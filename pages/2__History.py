# pages/2__History.py

import os
import json
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from src.proc_audio import display_transcription_with_timestamps
from src.utils import save_to_pdf, is_portrait_video, get_detected_sequences

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
            st.progress(results['safe_score_resnet'], text=f"Safe: {results['safe_score_resnet'] * 100:.2f}%")
            st.progress(results['harmful_score_resnet'], text=f"Violent: {results['harmful_score_resnet'] * 100:.2f}%")

            # Get sequences from the original output folder
            video_name = os.path.splitext(selected_video)[0]
            output_dir = os.path.join("output", video_name)
            sequences = get_detected_sequences(output_dir)
            if sequences:
                st.write(f"**Detected {len(sequences)} violent sequences**")

                # Display GIFs in rows of 2
                for i in range(0, len(sequences), 2):
                    cols = st.columns(2)
                    for col_idx in range(2):
                        if i + col_idx < len(sequences):
                            with cols[col_idx]:
                                st.markdown(f"**Sequence {i + col_idx + 1}**")
                                st.image(
                                    sequences[i + col_idx]['gif_path'],
                                    use_container_width=True
                                )
            else:
                st.info("No violent sequences detected")

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