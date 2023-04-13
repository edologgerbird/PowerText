import streamlit as st
import time


def page_under_construction(page_name, error=None):
    '''Function to display a page under construction message
    Args:
        page_name (str): Name of the page
        error (Exception, optional): Error message. Defaults to None.

    Returns:
        None
    '''

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
    '''Function to cache objects in Streamlit Session State
    Args:
        object (object): Object to be cached
        key (str): Key to be used to cache the object

    Returns:
        object: Cached object
    '''

    if key not in st.session_state:
        st.session_state[key] = object
    else:
        st.session_state[key] = object
    return st.session_state[key]


def customDisppearingMsg(msg, wait=3, type_='success', icon=None):
    '''Function to display a custom disappearing message
    Args:
        msg (str): Message to be displayed
        wait (int, optional): Time to wait before disappearing. Defaults to 3.
        type_ (str, optional): Type of message. Defaults to 'success'.
        icon (str, optional): Icon to be displayed. Defaults to None.

    Returns:
        object: Placeholder object
    '''

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
    '''Function to check if a key exists in Streamlit Session State
    Args:
        key (str): Key to be checked

    Returns:
        bool: True if key exists, False otherwise
    '''

    if key not in st.session_state:
        return False
    else:
        return True


def no_file_uploaded():
    '''Function to display a message when no file is uploaded
    Args:
        None

    Returns:
        None
    '''

    customDisppearingMsg(
        "No file uploaded yet! Please upload your CSV file in the 'Home' page!", wait=-1, type_='warning', icon='⚠️')
