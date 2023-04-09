import streamlit as st
import time


def page_under_construction(page_name, error=None):
    col1, col2, col3 = st.columns((1, 3, 1))
    if error:
        col2.image('./assets/capy.png', use_column_width='auto',
                   caption="Something Broke! Cappy the Capybara is working hard to fix it!")
        col1, col2, col3 = st.columns((1, 3, 1))
        col2.exception(error)
    else:
        col2.image('./assets/capy.png', use_column_width='auto',
                   caption="Cappy the Capybara is taking a break at the moment! Chill!")
        col1, col2, col3 = st.columns((1, 3, 1))
        col2.error(
            f'{page_name} is currently under construction! Do come back soon!')
    st.stop()

def cache_object(object, key):
    if key not in st.session_state:
        st.session_state[key] = object
    else:
        st.session_state[key] = object
    return st.session_state[key]

def customDisppearingMsg(msg, wait=3, type_='success', icon=None):
    placeholder = st.empty()
    if type_ == 'success':
        placeholder.success(msg, icon=icon)
    elif type_ == 'warning':
        placeholder.warning(msg, icon=icon)
    elif type_ == 'info':
        placeholder.info(msg, icon=icon)
    if wait > 0:
        time.sleep(wait)
        placeholder.empty()
    return placeholder

def check_session_state_key(key):
    if key not in st.session_state:
        return False
    else:
        return True
    
def no_file_uploaded():
    customDisppearingMsg("No file uploaded yet! Please upload your CSV file in the 'Home' page!", wait=-1, type_='warning', icon='⚠️')
