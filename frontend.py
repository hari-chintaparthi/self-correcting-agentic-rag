import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the LangGraph workflow and ingestion engine directly
from rag_pipeline import app as rag_graph, ingest_document

# Page configuration
st.set_page_config(
    page_title="Self-Correcting Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        color: white;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 Agentic RAG Control Panel")
st.sidebar.markdown("""
This production-grade system combines **LangGraph**, **Groq (Llama 3.3)**, and **local FAISS vector storage** to perform self-correcting RAG.

### Pipeline Steps:
1. **Semantic Search** (Hugging Face API)
2. **Relevance Judge Node** (Grader)
3. **Query Expansion** (Self-Correction Loop)
4. **Context-Grounded Generation**
5. **Hallucination Evaluation**
""")
st.sidebar.info("Upload documents in the dashboard to populate the vector store.")

# Title
st.title("🧠 Self-Correcting Agentic RAG Platform")
st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs(["💬 Ask Questions", "📤 Upload Documents"])

with tab1:
    st.header("💬 Query Your Document Library")
    
    # Initialize message list
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "trace" in message:
                with st.expander("🔍 RAG Execution Trace"):
                    st.write(message["trace"])

    # Input query
    if user_query := st.chat_input("Ask a question about your uploaded documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Trigger LangGraph directly
        with st.spinner("Agent thinking & retrieving..."):
            try:
                # Define initial state
                initial_state = {
                    "query": user_query,
                    "rewritten_query": "",
                    "retrieved_documents": [],
                    "generation": "",
                    "relevance_score": 0.0,
                    "hallucination_grade": "NO",
                    "loop_count": 0
                }
                
                # Run the LangGraph state machine
                final_state = rag_graph.invoke(initial_state)
                
                answer = final_state.get("generation", "No answer could be generated.")
                
                # Construct trace data
                trace = {
                    "Original Query": final_state.get("query"),
                    "Rewritten Query": final_state.get("rewritten_query") or "No rewrite needed",
                    "Document Relevance Score": final_state.get("relevance_score"),
                    "Hallucination Grade (Passed?)": final_state.get("hallucination_grade"),
                    "Correction Graph Loops Executed": final_state.get("loop_count"),
                    "Retrieved Chunks Used": final_state.get("retrieved_documents", [])
                }
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "trace": trace
                })
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("🔍 RAG Execution Trace"):
                        # Render trace dashboard
                        col1, col2, col3 = st.columns(3)
                        col1.markdown(f"**Relevance Judge Score**\n<div class='metric-value'>{trace['Document Relevance Score']}</div>", unsafe_allow_html=True)
                        col2.markdown(f"**Loops Executed**\n<div class='metric-value'>{trace['Correction Graph Loops Executed']}/3</div>", unsafe_allow_html=True)
                        col3.markdown(f"**Hallucination Check**\n<div class='metric-value'>{trace['Hallucination Grade (Passed?)']}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown(f"**Final Rewritten Query:** `{trace['Rewritten Query']}`")
                        st.markdown("**Context Chunks Used:**")
                        for idx, chunk in enumerate(trace["Retrieved Chunks Used"]):
                            st.info(f"Chunk {idx+1}: {chunk}")
                            
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {str(e)}")

with tab2:
    st.header("📤 Add New Documents to Vector Store")
    st.markdown("Upload PDF, Word, or TXT documents. They will be parsed, split into 500-character semantic chunks, embedded via Hugging Face, and saved into your local FAISS database.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Ingest Files"):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                    # Save uploaded file to a temporary file locally to process it
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name
                    
                    try:
                        # Call local ingestion function
                        message = ingest_document(temp_path, uploaded_file.name)
                        st.success(message)
                    except Exception as e:
                        st.error(f"Error uploading '{uploaded_file.name}': {str(e)}")
                    finally:
                        # Remove the temp file
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
