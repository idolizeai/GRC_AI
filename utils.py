from summarizer import Summarizer
from transformers import pipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.docstore.document import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings                    
import torch
import shutil
import textwrap
import re
import os
import torch
import shutil
import gc
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


## manipulate this code as per requirements to preprocess data
def preprocess(text):
    text = text.replace("\n", "") #removes /n
    text = re.sub(r'-\d+-', '', text) #removes headers
    return text
    
def chunker(text,file_path):
    text_document = Document(page_content=text, metadata={"source": file_path})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=100
        )
    chunks = text_splitter.split_documents([text_document])
    return chunks

def pdf_path():
    pdf_name = os.listdir('uploads')[0]
    pdf_path = os.path.join('uploads', pdf_name)
    return(pdf_path)

#   sentences_list = [sentence.page_content for sentence in sentences] #we neet to convert it into list for model input

def create_model():

    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  model_kwargs={"device": "cuda"})
    return model

def delete_collection(db_folder="vector_db"):
    """Properly closes and deletes the existing vector database."""
    if os.path.exists(db_folder):
        try:
            vector_store = Chroma(persist_directory=db_folder, embedding_function=create_model())
            vector_store.delete_collection()            
            del vector_store
            gc.collect()
            time.sleep(2)  

        except Exception as e:
            print(f"Warning: Could not properly close the existing vector database: {e}")

        try:
            shutil.rmtree(db_folder)
            print("Deleted old vector_db successfully!")
        except PermissionError:
            time.sleep(2)  
            shutil.rmtree(db_folder, ignore_errors=True)
    os.makedirs(db_folder, exist_ok=True)

def create_vector_db(chunks_var):
    db_folder = 'vector_db'
    if os.path.exists(db_folder):
        try:
            vector_store = Chroma(persist_directory=db_folder, embedding_function=create_model())
            vector_store.delete_collection()
            del vector_store
            gc.collect()
            time.sleep(2)  

        except Exception as e:
            print(f"Warning: Could not properly close the existing vector database: {e}")
        try:
            shutil.rmtree(db_folder)
        except PermissionError:
            time.sleep(2) 
            shutil.rmtree(db_folder, ignore_errors=True)

    os.makedirs(db_folder, exist_ok=True)
    model = create_model()
    vector_store = Chroma.from_documents(documents=chunks_var, embedding=model, persist_directory=db_folder)
    vector_store.persist()
    
    return vector_store


# def create_vector_db(chunks_var):
#     db_folder = 'vector_db'
#     if os.path.exists(db_folder):
#         shutil.rmtree(db_folder)
#     os.makedirs(db_folder, exist_ok=True)
#     model = create_model()
#     vector_store = Chroma.from_documents(documents=chunks_var,
#                                           embedding=model,
#                                             persist_directory=db_folder)   
#     vector_store.persist()
#     return vector_store

def create_complete_retriever(vector_store, llm):
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 6,                     # Number of chunks to retrieve
            "score_threshold": 0.05,     # Minimum similarity score1
        }
    )
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=compression_retriever,
        llm=llm
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[compression_retriever, multi_query_retriever],
        weights=[0.7, 0.3]
    )

    return ensemble_retriever


