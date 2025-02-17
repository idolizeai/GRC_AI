from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from flask_sqlalchemy import SQLAlchemy
from langchain.docstore.document import Document
from sqlalchemy.exc import IntegrityError
from transformers import pipeline
from torch import cuda
import json
from utils import create_vector_db, preprocess, chunker, pdf_path, create_model, create_complete_retriever, delete_collection
from classifier_utils import read_pdf, preprocess_text, split_into_chunks, summarize, classification
import spacy
from flask_cors import CORS
import datetime
import warnings
import shutil
import urllib
import os
import random


grc_params = urllib.parse.quote_plus(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=216.48.191.98;'
    'DATABASE=GRC_AI;'
    'UID=ibsadmin;'
    'PWD=Viking@@ibs2023'
)


llm = OllamaLLM(model="llama3.1:8b", temperature=0, streaming=True)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Use necessary amount of sentences but keep the answer concise. "
    "\n\n"
    "{context}"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

db_folder = 'uploads'
if os.path.exists(db_folder):
    shutil.rmtree(db_folder)

warnings.filterwarnings("ignore")  
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = f"mssql+pyodbc:///?odbc_connect={grc_params}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
nlp = spacy.load("en_core_web_sm")
db = SQLAlchemy(app)
CORS(app)

class LLM(db.Model):
    __tablename__ = 'LLM'
    user_query = db.Column(db.Text, nullable=True) 
    ai_answer = db.Column(db.Text, nullable=True)   
    backend_unique_id = db.Column(db.String(255),nullable=True)
    call_id = db.Column(db.String(255), primary_key=True)
    user_query_datetime = db.Column(db.DateTime, nullable=True)
    ai_answer_datetime = db.Column(db.DateTime, nullable=True)

class Circular_Index(db.Model):
    __tablename__ = 'Circular_Index'
    frontend_unique_id = db.Column(db.String(255), nullable=True)
    # Restoring the primary key here:
    backend_unique_id = db.Column(db.String(255), primary_key=True)
    description_of_document = db.Column(db.Text, nullable=True)
 
class Circular_Processing(db.Model):
    __tablename__ = 'Circular_Processing'
    summary = db.Column((db.Text), nullable=True)
    classification = db.Column(db.Text, nullable=True)
    # Restoring the primary key:
    document_id = db.Column(db.String(255), primary_key=True)
    backend_id = db.Column(db.String(255), nullable=True)
    revised_classification = db.Column(db.Text, nullable=True)

class Paraphrase(db.Model):
    __tablename__ = 'paraphrase'  
    backend_unique_id = db.Column(db.String(255), nullable=True, primary_key=True) 
    summary = db.Column(db.Text, nullable=True)  
    classification = db.Column(db.Text, nullable=True)

with app.app_context():
    from sqlalchemy import inspect
    inspector = inspect(db.engine)

    if 'LLM' not in inspector.get_table_names():
        db.create_all()
        print("✅ LLM Table Created Successfully!")
    else:
        print("✅ LLM Table Already Exists!")


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API"})

@app.route('/api/upload_documents', methods=["POST"])
def upload_documents():
    file1 = request.files.get('document')
    delete_collection("vector_db")
    shutil.rmtree("uploads", ignore_errors=True)
    os.makedirs("uploads", exist_ok=True)
    filename1 = secure_filename(file1.filename)
    save_path1 = os.path.join("uploads", filename1)
    file1.save(save_path1)
    reader = PdfReader(save_path1)
    data = "".join([page.extract_text() for page in reader.pages])
    preprocessed_data = preprocess(data)
    saved_path = pdf_path()
    chunks = chunker(preprocessed_data, saved_path)
    vector_store = create_vector_db(chunks)

    type = request.form.get('type')
    frontend_unique_id = "frontend-unique-id"
    description = request.form.get('description')
    description = description.strip() if description and description.strip() else None
    prefix = f'Circular_{type}_'
    max_record = db.session.query(Circular_Index.backend_unique_id) \
        .filter(Circular_Index.backend_unique_id.like(f'{prefix}%')) \
        .order_by(Circular_Index.backend_unique_id.desc()).first()
    if max_record:
        try:
            current_num = int(max_record[0].split('_')[-1])
        except Exception:
            current_num = 0
        new_number = current_num + 1
    else:
        new_number = 1
    backend_unique_id = prefix + f'{new_number:03d}'
   
    new_index = Circular_Index(
        frontend_unique_id=frontend_unique_id,
        backend_unique_id=backend_unique_id,
        description_of_document=description
    )   
    try:
        db.session.add(new_index)
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({'error': 'Database error during insertion.', 'details': str(e)}), 500
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500
     
    return jsonify({'Result': 'Vector Database Created',
                    'number_of_chunks': len(chunks),
                    'backend_unique_id': backend_unique_id,
                    }), 200


@app.route('/api/query_llm', methods=['POST'])
def query_llm():
    try:
        data = request.json
        query = data.get("query")
        user_query_datetime=datetime.datetime.utcnow()
        backend_unique_id = data.get("backend_unique_id")   
        model = create_model()
        vector_store = Chroma(persist_directory='vector_db', embedding_function=model)
        retriever = create_complete_retriever(vector_store, llm)
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        document_chain = create_stuff_documents_chain(llm, chat_prompt)  
        chain = create_retrieval_chain(retriever_from_llm, document_chain) 
        result = chain.invoke({"input": query})
        response_answer = result["answer"]
        ai_answer_datetime = datetime.datetime.utcnow()
        call_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S") + str(random.randint(100, 999))
        
        new_record = LLM(
            user_query=query,
            ai_answer=response_answer,
            backend_unique_id=str(backend_unique_id),
            call_id=call_id,
            user_query_datetime=user_query_datetime,
            ai_answer_datetime=ai_answer_datetime
        )


        
        try:
            db.session.add(new_record)
            db.session.commit()
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            db.session.rollback()
            
        def generate():
            for chunk in chain.stream({"input": query}):  # Using `.stream()` for LLM Streaming
                yield json.dumps({"chunk": chunk}) + "\n"

        return jsonify({
            'Response': response_answer
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in query_llm: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/selected_text_summary', methods=['POST'])
def selected_text_summary():
    data = request.json
    backend_unique_id = data.get("backend_unique_id")
    selected_text = data.get("selected_text")
    print(selected_text)
    doc = [Document(page_content=selected_text, metadata={"source":backend_unique_id})]
    summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly skilled assistant specializing in making complex 
               information accessible and clear. Using the provided context, your task is to:
               1. First, provide a concise summary that captures the main points in 2-3 sentences.
               2. Then, paraphrase the summary in simple, everyday language that anyone can understand.    
               Context: {context}""")
        ])
    chain = create_stuff_documents_chain(llm,summary_prompt)
    result = chain.invoke({"context":doc})
    print(result)
    

    return jsonify({
        'summary': result
    })

@app.route('/api/classifier', methods = ['POST'])
def classifier():
    data = request.json
    backend_unique_id = data.get("backend_unique_id")
    summary = data.get("summary")
    classification = data.get("classification")

    record  = Paraphrase(
        backend_unique_id= backend_unique_id,
        summary = summary,
        classification = classification
    )

    db.session.add(record)
    db.session.commit()   
    return jsonify({
     "message" : "Classified Successfully"
    })


@app.route('/api/overall_summary', methods=['POST'])
def summarize_classify_update():
    backend_unique_id = request.form.get('backend_unique_id')
    
    try:
        pdf_file_path = pdf_path()
        raw_text = read_pdf(pdf_file_path)
        
        # Preprocess and summarize extracted text.
        cleaned_text = preprocess_text(raw_text)
        device_id = 0 if cuda.is_available() else -1
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device_id,
            batch_size=8
        )
        chunks = split_into_chunks(cleaned_text, summarizer.tokenizer, max_tokens=512)
        final_summary = summarize(chunks, summarizer)
        
        # Split summary into sentences.
        doc = nlp(final_summary)
        summary_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip().split()) > 7]
        
        # Classify the summary sentences using utility function.
        classification_results = classification(summary_sentences)
        
        # Update processing in the database.
        inserted_ids = []
        for idx, item in enumerate(classification_results, start=1):
            predicted_category = item.get("predicted_category")
            sentence = summary_sentences[idx-1]
            document_id = f"{backend_unique_id}_summary{idx}"
            new_entry = Circular_Processing(
                document_id=document_id,
                summary=sentence,
                backend_id=backend_unique_id,
                classification=predicted_category,
                revised_classification=None
            )
            db.session.add(new_entry)
            inserted_ids.append(document_id)
        db.session.commit()
        
        return jsonify({
            'message': 'Processing complete!',
            'summary': summary_sentences,
            'classifications': classification_results,
            'inserted_document_ids': inserted_ids
        }), 201
        
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

    
if __name__ == '__main__':
    app.run(debug=True, port=8003)





#-------------------------------------------------------------------------------------------------------------------------------------------------
# @app.route('/api/upload_documents', methods=["POST"])
# def upload_documents():
#     """
#     Endpoint that receives two files and stores that into uploads directory.
#     """
#     file1 = request.files.get('document')   
#     shutil.rmtree("uploads") if os.path.exists("uploads") and os.path.isdir("uploads") else print("uploads exists")
#     if not os.path.exists("uploads"):
#         os.makedirs("uploads")
#     UPLOAD_FOLDER = "uploads"
#     shutil.rmtree("uploads")
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     filename1 = secure_filename(file1.filename)
#     save_path1 = os.path.join(UPLOAD_FOLDER, filename1)
#     file1.save(save_path1)
#     data=""
#     reader = PdfReader(save_path1)
#     for i in range(len(reader.pages)):
#         page = reader.pages[i]
#         text = page.extract_text()
#         data += text
#     preprocessed_data = preprocess(data)
#     saved_path = pdf_path()
#     chunks = chunker(preprocessed_data,saved_path) 
#     vector_store = create_vector_db(chunks)    
#     if vector_store is not None:
#         del vector_store        
#     import gc
#     gc.collect()
    
#     return jsonify({
#         'Result': 'Vector Database Created',
#         'number_of_chunks': len(chunks)
#     }), 200