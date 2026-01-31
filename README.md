## Webpage Q&A using Ollama + PostgreSQL (RAG)

## Overview

In this project, I built a Retrieval-Augmented Generation (RAG) application that allows a user to input a webpage URL, index its content, and then ask questions about that page. The system retrieves relevant sections from the webpage and uses a local language model to generate answers strictly based on the retrieved content.

This project demonstrates my understanding of web scraping, embeddings, vector databases, local LLM usage, and full-stack integration using Streamlit.

## Architecture Overview

The system works in the following stages:


1. Web Scraping

    I download the webpage using requests and parse it with BeautifulSoup. I remove scripts, styles, and non-content tags to extract only meaningful visible text.

2. Text Chunking

    I split the extracted text into overlapping chunks using RecursiveCharacterTextSplitter. This improves retrieval accuracy because smaller semantic segments are easier to embed and search.

3. Embedding Generation

    I use the nomic-embed-text model through OllamaEmbeddings to convert each chunk into a vector representation.


4. Persistent Vector Storage

    I store embeddings in PostgreSQL using the pgvector extension via LangChain’s PGVector integration. This allows semantic search using vector similarity.


5. Retrieval

    When a user asks a question, I retrieve the top-k most relevant chunks from PostgreSQL based on vector similarity.


6. Answer Generation

    I send the retrieved context along with the user’s question to a local LLM (gemma:2b via Ollama). The prompt forces the model to answer strictly using the provided context.

7. User Interface

    I built a simple Streamlit interface where users can:

        * Enter a webpage URL

        * Index the page

        * Ask questions about it


## Technology Stack

Frontend:

* Streamlit

Backend:


* Python

* Requests

* BeautifulSoup


LLM and Embeddings:

* Ollama

* gemma:2b (for answer generation)

* nomic-embed-text (for embeddings)


Vector Database:


* PostgreSQL

* pgvector extension

* langchain\_postgres


## Why I Used PostgreSQL Instead of FAISS or Chroma


I chose PostgreSQL with pgvector to build a more production-oriented system. While FAISS and Chroma are excellent for experimentation, PostgreSQL provides durable, persistent storage, transactional safety, and better integration with full-stack applications.


This approach allows embeddings to remain stored even after the application stops, making the system more realistic and scalable.


## Why Persistent Storage Matters


Persistent storage ensures that once a webpage is indexed, I do not need to re-scrape and re-embed it every time the application restarts. This reduces computation time, improves efficiency, and enables multi-session or multi-user scenarios in the future.


## Model Separation

I deliberately separated the embedding model and the language model.

* The embedding model (nomic-embed-text) is optimized for semantic similarity.

* The language model (gemma:2b) is optimized for reasoning and answer generation.


This separation follows good system design principles and allows each component to be replaced or upgraded independently.

## Environment Configuration


I use a .env file to store sensitive configuration such as the PostgreSQL connection string. The .env file is excluded from version control using .gitignore to avoid exposing credentials.


Example environment variable:


PGVECTOR\_URL=postgresql+psycopg://postgres:postgres@localhost:5433/rag\_db


## Database Requirements


PostgreSQL must be running with the pgvector extension enabled. The database must exist before running the application.


The application stores embeddings inside a collection named "webpage\_chunks".


## How to Run the Project

1. Start PostgreSQL (with pgvector enabled).

2. Ensure Ollama is running locally.

3. Pull required models in Ollama:


    * gemma:2b

    * nomic-embed-text

4. Set the PGVECTOR\_URL in the .env file.

5. Install dependencies inside a virtual environment.

6. Run:

    streamlit run app.py


## How the System Could Scale


This architecture is modular.


* The database can be deployed on a dedicated server.

* The LLM could be replaced with a larger local model or a cloud-based API.

* Connection pooling and containerization (Docker) could improve reliability.

* Multiple users could index and query different documents.

Since the embedding layer, database, and LLM are separate components, each can scale independently.


## Limitations


* The scraper is simple and does not handle complex dynamic sites.

* There is no authentication or multi-user isolation.

* There is no deduplication logic for repeated indexing.

* It currently indexes one webpage at a time.


## Future Improvements

* Add support for indexing multiple URLs.

* Add metadata filtering.

* Add authentication.

* Deploy using Docker Compose.

* Add caching to prevent re-indexing the same URL.


## Conclusion


This project demonstrates my ability to:


* Build a complete RAG pipeline

* Integrate local LLMs with Ollama

* Use PostgreSQL with pgvector for persistent vector storage

* Design modular systems

* Build interactive interfaces using Streamlit


The goal of this assignment was to demonstrate a working RAG application using local models and persistent storage, and this implementation fulfills that requirement while keeping the architecture clean and extensible.