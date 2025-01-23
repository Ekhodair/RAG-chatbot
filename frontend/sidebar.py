import sys
import streamlit as st
import sys
# sys.path.append("helpers")
from frontend.api_utils import upload_document, list_documents, delete_document
from helpers.db_utils import get_all_chat_sessions, get_chat_history, delete_chat_session



def create_new_chat():
    """Create a new chat session"""
    st.session_state.session_id = None
    st.session_state.messages = []
    st.rerun()

def display_chat_history():
    st.markdown("""
        <style>
        .chat-container {
            position: relative;
        }
        .chat-actions {
            visibility: hidden;
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
        }
        .chat-container:hover .chat-actions {
            visibility: visible;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("🗨️ New Chat", key="new_chat", use_container_width=True):
        create_new_chat()

    st.divider()
    chat_sessions = get_all_chat_sessions()
    if not chat_sessions:
        st.info("No previous chats found", icon="ℹ️")
        return
    for session in chat_sessions:
        chat_title = session['first_message'][:50] + '...' if len(session['first_message']) > 50 else session['first_message']
        chat_container = st.container()
        with chat_container:
            cols = st.columns([15, 1])
            
            with cols[0]:
                if st.button(
                    f"📝 {chat_title}",
                    key=f"chat_{session['session_id']}",
                    use_container_width=True
                ):
                    # Load this chat session
                    messages = get_chat_history(session['session_id'])
                    st.session_state.session_id = session['session_id']
                    st.session_state.messages = []
                    for msg in messages:
                        st.session_state.messages.append({
                            "role": "user" if msg["role"] == "user" else "assistant",
                            "content": msg["content"]
                        })
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", 
                    key=f"delete_{session['session_id']}", 
                    help="Delete this chat",
                    type="secondary",
                ):
                    if delete_chat_session(session['session_id']):
                        # If currently viewing this chat, create a new session
                        if st.session_state.session_id == session['session_id']:
                            create_new_chat()
                        else:
                            st.rerun()

def display_document_management():
    """Display document management section"""
    # Document Management Section
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file is not None:
        if st.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.success(f"File '{uploaded_file.name}' uploaded successfully.")
                    st.session_state.documents = list_documents()
    
    # Document List Section
    st.header("Uploaded Documents")
    if st.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()
    
    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()
    
    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.text(f"{doc['filename']}\n(ID: {doc['id']})")
        
        selected_file_id = st.selectbox(
            "Select a document to delete",
            options=[doc['id'] for doc in documents],
            format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x)
        )
        
        if st.button("Delete Selected Document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.success(f"Document deleted successfully.")
                    st.session_state.documents = list_documents()

def display_sidebar():
    """Display the complete sidebar with tabs"""
    tab1, tab2 = st.sidebar.tabs(["💬 Chat History", "📚 Documents"])
    with tab1:
        display_chat_history()
    
    with tab2:
        display_document_management()