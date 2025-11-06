# src/tools/rag_tool.py

import os
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define paths to the data and the persistent vector store
VECTOR_STORE_PATH = "./chroma_db"
DATA_PATH_PDF = "./data/Master_Plan_for_Delhi_2021.pdf" 
DATA_PATH_JSON = "./data/compliance_rules.json"

def get_retriever():
    """
    Initializes a Chroma vector store. If the store doesn't exist, it creates it
    by loading, splitting, and embedding the documents. Otherwise, it loads the
    existing store from disk.
    Returns:
        A LangChain retriever object.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Creating new vector store...")
        # Load documents from PDF and JSON sources
        pdf_loader = PyPDFLoader(DATA_PATH_PDF)
        pdf_docs = pdf_loader.load()
        json_loader = JSONLoader(file_path=DATA_PATH_JSON, jq_schema='.rules[].description', text_content=False)
        json_docs = json_loader.load()
        
        # Combine all documents
        all_docs = pdf_docs + json_docs
        
        # Split documents into smaller, manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        # Create and persist the vector store using OpenAI embeddings
        vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=VECTOR_STORE_PATH)
        print("Vector store created successfully.")
    else:
        # Load the existing vector store from disk
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=OpenAIEmbeddings())
    
    return vector_store.as_retriever()

# Initialize the retriever when the module is loaded
retriever = get_retriever()

@tool
def rag_compliance_lookup(query: str) -> str:
    """
    Looks up relevant urban planning compliance rules from the vector store
    based on a user's query.
    """
    docs = retriever.invoke(query)
    # Join the content of the retrieved documents into a single string
    return "\n---\n".join([doc.page_content for doc in docs])