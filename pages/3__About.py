# pages/3__About.py


import streamlit as st
from styles.styles import spacer

# Configure the page
st.set_page_config(
    page_title="About BuddyGuard",
    page_icon="ℹ️",
    layout="wide"
)


def about_page():
    st.title("About BuddyGuard 🛡️")

    # ---- RED DISCLAIMER BANNER ----
    st.markdown("""
    <div style="
        background-color: #ffdddd;
        border-left: 6px solid #f44336;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 4px;
        color: #000000;
    ">
        <h4 style="color: #d32f2f; margin:0;">Important Disclaimer</h4>
        <p style="margin:5px 0;">
        BuddyGuard's AI models may produce <strong>false positives/negatives</strong>. 
        Always use human judgment and never rely solely on automated detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # App Description (Now with inline disclaimer)
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header("Our Mission")
            st.markdown("""
            BuddyGuard is an AI-powered content moderation system designed to:
            - 🔍 Detect violent content in videos using computer vision
            - 🎙️ Analyze spoken words for harmful language
            - 🛡️ Protect users from exposure to disturbing content
            - 📊 Provide transparent scoring of potential risks

            <div style="
                background-color: #fff4e6;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-size: small;
                color: #000000;
            ">
            <strong>Note:</strong> Performance varies by video quality, language, and context.
            </div>
            """, unsafe_allow_html=True)

            spacer(20)

            st.header("How It Works")
            st.markdown("""
            1. **Video Analysis**: Our deep learning model examines each frame for violent content
            2. **Audio Processing**: Speech-to-text conversion with toxic language detection
            3. **Fusion Scoring**: Combined visual and audio analysis for comprehensive protection
            4. **Human Review**: Flagged content can be reviewed by moderators
            """)

        # with col2:
        #     st.image("saves/Buddyguard_4_3.png", use_container_width=True)
        #     st.markdown("""
        #     <div style="
        #         text-align: center;
        #         font-size: small;
        #         color: #666;
        #     ">
        #     BuddyGuard - Your AI content safety companion
        #     </div>
        #     """, unsafe_allow_html=True)

        with col2:
            st.image("saves/Buddyguard_4_3.png", use_container_width=True)
            st.markdown("""
            <div style="text-align: center; font-size: small;">
            BuddyGuard - Your AI content safety companion
            </div>
            """, unsafe_allow_html=True)

        # Team/Technology Section
        st.markdown("---")
        st.header("Technology Stack")

        tech_cols = st.columns(4)
        tech = [
            ("PyTorch", "Deep Learning Framework", "https://pytorch.org"),
            ("ResNet-LSTM", "Hybrid Vision Model", ""),
            ("Whisper", "Speech Recognition", "https://openai.com/research/whisper"),
            ("BERT", "Text Classification", "https://arxiv.org/abs/1810.04805")
        ]

        for col, (name, desc, link) in zip(tech_cols, tech):
            with col:
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 15px;
                    height: 150px;
                    margin-bottom: 20px;
                ">
                    <h4>{name}</h4>
                    <p style="font-size: small;">{desc}</p>
                    {f'<a href="{link}" target="_blank">🔗 Learn More</a>' if link else ''}
                </div>
                """, unsafe_allow_html=True)

    # Final Footer Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="
        text-align: center;
        padding: 15px;
        background-color: #f5f5f5;
        border-radius: 5px;
        color: #000000;
    ">
        <p><strong>⚠️ Use At Your Own Risk:</strong> BuddyGuard is for <em>assistive purposes only</em>.</p>
        <p style="font-size: small;">
        By using this tool, you acknowledge that AI systems have inherent limitations 
        and may produce inaccurate results.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    about_page()