import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Private RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🤖 Private RAG Document Assistant")

st.caption(
    "Upload a PDF and retrieve relevant information using "
    "Retrieval-Augmented Generation."
)


# ==========================================
# SESSION STATE
# ==========================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# ==========================================
# SIDEBAR - PDF UPLOAD
# ==========================================

with st.sidebar:

    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("Process Document"):

            with st.spinner("Processing PDF..."):

                # Save uploaded PDF temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load PDF
                loader = PyPDFLoader("temp.pdf")
                documents = loader.load()

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )

                chunks = text_splitter.split_documents(documents)

                # Create vector database
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                # Delete temporary file
                os.remove("temp.pdf")

                st.success(
                    f"Successfully processed {len(chunks)} chunks!"
                )


# ==========================================
# MAIN QUESTION SECTION
# ==========================================

st.subheader("💬 Ask Questions From Your Document")

user_query = st.text_input(
    "Enter your question:"
)


# ==========================================
# RETRIEVAL PIPELINE
# ==========================================

if user_query:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process a PDF document first."
        )

    else:

        with st.spinner("Searching your document..."):

            # Create retriever
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

            # Retrieve relevant documents
            relevant_docs = retriever.invoke(user_query)

            # Combine retrieved text
            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

        # Display result
        st.markdown("### 📚 Retrieved Information")

        st.info(
            "The following information was retrieved from your uploaded document."
        )

        st.write(context)

        # Display source pages
        st.markdown("### 📄 Source Pages")

        for i, doc in enumerate(relevant_docs):

            page_number = doc.metadata.get("page", "Unknown")

            st.write(
                f"Result {i + 1} — Page {page_number + 1 if isinstance(page_number, int) else page_number}"
            )
