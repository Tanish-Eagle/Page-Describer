# ---------------------------------------------
# Load environment variables from .env file
# (kept to match instructor template; not required for Ollama)
# ---------------------------------------------
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------------------------
# Standard libraries for web requests and parsing
# ---------------------------------------------
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------
# Streamlit for web UI
# ---------------------------------------------
import streamlit as st

# ---------------------------------------------
# LangChain components for LLM interaction
# ---------------------------------------------
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------
# LangChain components for embeddings and vector DB
# ---------------------------------------------
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =================================================
# MODEL INITIALIZATION
# =================================================

# Initialize the local LLM (Gemma 2B) using Ollama
# This model is used ONLY for generating answers
llm = Ollama(model="gemma:2b")

# Initialize the embedding model
# This model converts text into numeric vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# =================================================
# PROMPT SETUP
# =================================================

# Prompt template that forces the LLM to answer
# ONLY using retrieved webpage content (RAG)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers using ONLY the provided webpage content."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# Converts the LLM output into a plain string
output_parser = StrOutputParser()

# Chain = Prompt → LLM → Output parser
chain = prompt | llm | output_parser


# =================================================
# WEB SCRAPING FUNCTION
# =================================================

def extract_text_from_url(url):
    """
    Downloads a webpage and extracts readable text.
    HTML tags, scripts, and styles are removed.
    """

    # Fetch webpage
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

    # Parse HTML
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text("\n")

    # Clean short and empty lines
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]

    # Limit content size to keep embedding fast
    return "\n".join(lines[:300])


# =================================================
# STREAMLIT UI – PAGE LOADING
# =================================================

st.title("Webpage Q&A using Gemma + Ollama (RAG)")

# Input field for URL
url = st.text_input("Enter a webpage URL")

# Store vector database in session state
# so it persists across user interactions
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# Button to load and index webpage
if st.button("Load & Index Page"):
    if not url:
        st.warning("Please enter a URL")
    else:
        # Step 1: Download and clean webpage text
        with st.spinner("Downloading webpage..."):
            text = extract_text_from_url(url)

        # Wrap raw text in a LangChain Document object
        docs = [Document(page_content=text)]

        # Step 2: Split text into overlapping chunks
        # Chunking helps with embeddings and retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        # Step 3 & 4: Create embeddings and store in ChromaDB
        with st.spinner("Creating embeddings & storing in ChromaDB..."):
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

        # Save vector DB for later queries
        st.session_state.vectorstore = vectordb
        st.success("Webpage indexed successfully!")


# =================================================
# STREAMLIT UI – QUESTION ANSWERING
# =================================================

# Input field for user question
question = st.text_input("Ask a question about the webpage")

if st.button("Ask"):
    if not st.session_state.vectorstore:
        st.error("Please load a webpage first.")
    elif not question:
        st.warning("Enter a question.")
    else:
        # Step 5 & 6: Retrieve relevant chunks
        with st.spinner("Searching relevant content..."):
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            # Modern LangChain retriever call
            docs = retriever.invoke(question)

        # Combine retrieved chunks into a single context
        context = "\n\n".join([d.page_content for d in docs])

        # Step 7: Send context + question to LLM
        with st.spinner("Gemma is thinking..."):
            answer = chain.invoke({
                "context": context,
                "question": question
            })

        # Display answer
        st.subheader("Answer")
        st.write(answer)
