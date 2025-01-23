import streamlit as st
from sidebar import display_sidebar
from interface import display_chat_interface, display_header_interface 


st.set_page_config(
        page_title="Document Q&A Assistant",
        page_icon="🤖",
        layout="wide"
    )

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display page
display_header_interface()
display_sidebar()
display_chat_interface()