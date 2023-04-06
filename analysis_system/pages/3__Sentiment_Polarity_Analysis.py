# Importing relevant modules

import streamlit as st
import utils.utility as util
import utils.design_format as format

st.title("Sentiment Polarity Analysis")
format.horizontal_line()


if __name__ == "__main__":
    try:
        run()
    except:
        util.page_under_construction("Sentiment Polarity Analysis")
