import streamlit as st
import sys
sys.path.append("helpers")
from api_utils import get_streaming_response


def display_chat_interface():
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Stream the response
            for response in get_streaming_response(
                prompt, st.session_state.session_id, st.session_state.model
            ):
                if response:
                    full_response += response["token"]
                    message_placeholder.markdown(full_response + "▌")
                    st.session_state.session_id = response["session_id"]
            message_placeholder.markdown(full_response)

        # Update session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def display_header_interface():
    st.markdown(
        """
    <style>
    /* Main title styling */
    .main-title-container {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .main-title {
        font-size: 1.5em;
        font-weight: 700;
        color: white;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        font-size: 1.2em;
        color: #e0e0e0;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        margin-top: 10px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, _ = st.columns([2, 7, 2])
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=["mistral", "llama3.3"],
            key="model",
            format_func=lambda x: x.upper(),
        )
    with col2:
        st.markdown(
            """
            <div class="main-title-container">
                <h2 class="main-title">🤖 Document Q&A Assistant</h2>
                <p class="subtitle">Your AI-powered document analysis companion</p>
            </div>
        """,
            unsafe_allow_html=True,
        )