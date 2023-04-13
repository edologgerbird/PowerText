import streamlit as st
import base64

deployed_on_st = True
if deployed_on_st:
    path = "analysis_system/"
else:
    path = ""


def align_text(text, alignment):
    '''Function to align text to column width. Allowed alignments: "justify", "center", "right", "left"
    Args:
        text (str): Text to be aligned
        alignment (str): Alignment of text

    Returns:
        str: Aligned text
    '''

    return st.markdown(f'<div style="text-align: {alignment};">{text}</div>', unsafe_allow_html=True)


def horizontal_line():
    '''Insert Horizontal Line
    Returns:
        str: Horizontal Line
    '''

    return st.markdown("""---""")


@st.cache_data
def get_base64_of_bin_file(bin_file):
    '''Function to get base64 of binary file

    Args:
        bin_file (str): Path to binary file

    Returns:
        str: Base64 of binary file
    '''

    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.cache_resource
def add_logo():
    '''Function to add logo to sidebar

    Returns:
        None
    '''

    bin_str = get_base64_of_bin_file(f"{path}assets/logo-colour.png")
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url("data:image/png;base64,%s");
                background-repeat: no-repeat;
                margin-top: 1.5em;
                padding-top: 150px;
                background-position: 40px 40px;
                background-size: 15em;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Powertext Analysis System";
                font-weight: 600;
                padding-left: 20px;
                margin-top: 10px;
                margin-right: 12px;
                margin-bottom: 2px;
                font-size: 20px;
                position: relative;
                flex-wrap: wrap;
                display: flex;
                top: 60px;
            }
        </style>
        """ % bin_str,
        unsafe_allow_html=True,
    )
