import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Private RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title("🤖 Private RAG Document Assistant")

st.write(
    "Upload a PDF document and ask questions using "
    "Retrieval-Augmented Generation."
)


# ==========================================
# SESSION STATE
# ==========================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


embeddings = load_embeddings()


# ==========================================
# SIDEBAR - DOCUMENT UPLOAD
# ==========================================

with st.sidebar:

    st.header("📄 Upload Your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("Process Document"):

            with st.spinner("Reading and processing your PDF..."):

                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load PDF
                loader = PyPDFLoader("temp.pdf")
                documents = loader.load()

                # Split text into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )

                chunks = text_splitter.split_documents(documents)

                # Create Chroma vector database
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                # Remove temporary file
                os.remove("temp.pdf")

                st.success(
                    f"Successfully processed {len(chunks)} text chunks!"
                )


# ==========================================
# QUESTION INPUT
# ==========================================

st.subheader("💬 Ask Questions From Your Document")

user_query = st.text_input(
    "Enter your question:"
)


# ==========================================
# RAG PIPELINE
# ==========================================

if user_query:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process a PDF document first."
        )

    else:

        with st.spinner("Searching relevant information..."):

            # Retrieve relevant chunks
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

            relevant_docs = retriever.invoke(user_query)

            # Combine retrieved documents
            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

        # Display retrieved context
        st.markdown("### 📚 Retrieved Context")

        with st.expander("View Document Context"):
            st.write(context)

        # Generate Answer
        st.markdown("### 🤖 AI Answer")

        try:

            # Get Hugging Face API Token
            hf_token = st.secrets["HF_TOKEN"]

            # Connect to Hugging Face
            client = InferenceClient(
                api_key=hf_token
            )

            # Prompt for the LLM
            prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:

"I could not find this information in the uploaded document."

Context:
{context}

Question:
{user_query}

Answer:
"""

            # Generate response
            response = client.text_generation(
                prompt,
                model="google/gemma-2-2b-it",
                max_new_tokens=200,
                temperature=0.2
            )

            # Show final answer
            st.success(response)

        except Exception as e:

            st.error(f"Something went wrong: {str(e)}")
