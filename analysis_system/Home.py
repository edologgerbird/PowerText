import streamlit as st
import utils.design_format as format
import utils.utility as util
import pandas as pd

st.title("🔥Powertext Analysis System")
format.horizontal_line()
st.subheader("Welcome to Powertext!")
format.align_text("This is a demo of the Powertext Analysis System. The system is designed to help you analyse content retrieved from social media and automatically classify these posts into broad Terms-of-service violations.", "justify")
format.align_text("For the purpose of this demo, will we upload a CSV of scraped data from Reddit.", "justify")
format.horizontal_line()

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    st.write("You selected `%s`" % uploaded_file.name)
    if st.button("Upload CSV"):
        # Storing CSV data to ST cache
        util.cache_object(pd.read_csv(uploaded_file), "csv_file")
        util.customDisppearingMsg(
            "PDF file uploaded successfully!", wait=3, type_='success', icon=None)
        util.customDisppearingMsg(
            "You may now navigate to the other tabs for analysis of the Reddit content!", wait=-1, type_='info', icon=None)
