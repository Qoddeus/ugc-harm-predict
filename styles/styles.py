import streamlit as st


def spacer(padding_height):
    """Adds a vertical spacer with a specified height.

    Args:
        padding_height (int): The desired height of the spacer in pixels.
    """
    spacer1 = st.empty()
    spacer1.markdown(f"<div style='padding-top: {padding_height}px;'></div>", unsafe_allow_html=True)