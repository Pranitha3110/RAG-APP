import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def process_single_pdf(filename="dmv.pdf"): # Change this to your PDF name
    # 1. Load the specific PDF 
    loader = PyPDFLoader(filename)
    documents = loader.load()
    
    # 2. Split into chunks: 500 chars, 50 overlap 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create embeddings and save [cite: 188]
    # Note: Use 'text-embedding-3-small' as requested in assignment [cite: 188]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save folder for GitHub [cite: 206]
    vectorstore.save_local("faiss_index")
    print(f"Processed {filename} into {len(chunks)} chunks and saved index.")

if __name__ == "__main__":
    process_single_pdf()