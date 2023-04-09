import streamlit as st

# Function to align text to column width. Allowed alignments: "justify", "center", "right", "left"
def align_text(text, alignment):
    return st.markdown(f'<div style="text-align: {alignment};">{text}</div>', unsafe_allow_html=True)

# Insert Horizontal Line
def horizontal_line():
    return st.markdown("""---""")