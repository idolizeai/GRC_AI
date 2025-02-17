import re
import time
import spacy
import torch
from transformers import pipeline
from torch import cuda
from PyPDF2 import PdfReader

nlp = spacy.load("en_core_web_sm")

def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return " ".join(text)

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def split_into_chunks(text, tokenizer, max_tokens=512):
    # Use spaCy to split into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        tokens = tokenizer.encode(current_chunk + " " + sentence)
        if len(tokens) > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_chunk(chunk, summarizer):
    try:
        summary = summarizer(
            chunk,
            min_length=90,
            max_length=100,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return ""

def summarize(chunks, summarizer):
    print("Summarizing text chunks in batch...")
    # Process all chunks at once instead of using ThreadPoolExecutor
    summaries = summarizer(
        chunks,
        min_length=90,
        max_length=100,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )
    return " ".join([s['summary_text'] for s in summaries])
    
def classify_sentence_custom(sentence):
    sentence_lower = sentence.lower()
    doc = nlp(sentence)
    
    # Extract named entities
    entity_labels = [ent.label_ for ent in doc.ents]
    
    # Rule 1: Calendar — if a DATE entity or month name is present
    if "DATE" in entity_labels \
       or re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', sentence_lower) \
       or re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', sentence_lower) \
       or re.search(r'\b(\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?)\b', sentence_lower) \
       or any(keyword in sentence_lower for keyword in ["tomorrow", "today", "yesterday", "deadline", "due date"]):
        return "calendar"
    # Rule 2: ToDo — if the sentence contains imperative language or action verbs
    todo_keywords = [
        "shall", "must", "to do", "action required", "update", "submit", "complete", 
        "execute", "perform", "ensure", "implement", "review", "approve", "prepare"
    ]
    if any(keyword in sentence_lower for keyword in todo_keywords):
        return "ToDo"
    
    # Rule 3: Checklist — if the sentence explicitly mentions checklist-related words or a list-like pattern
    checklist_keywords = [
        "checklist", "list", "item", "step", "points to be considered", "include", "register",
        "ensure", "verify", "confirm", "follow", "adhere to", "comply with"
    ]
    if any(keyword in sentence_lower for keyword in checklist_keywords):
        return "checklist"
    
    # Rule 4: Information — fallback category
    return "information"

def classification(sentences):
    candidate_labels = [
        "To-Do: A specific task or action that requires completion, often with a deadline or priority.",
        "Checklist: An organized list of items or steps to verify, complete, or review in order to ensure nothing is missed.",
        "Calendar: Events, dates, or scheduled activities that are tied to specific times or deadlines.",
        "Information: General statements, facts, or details provided for awareness or reference without immediate action required."
    ]
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
    )
    
    classifications = []
    for sentence in sentences:
        try:
            rule_category = classify_sentence_custom(sentence)
            
            if rule_category == "information":
                result = classifier(sentence, candidate_labels=candidate_labels)
                classifier_category = result["labels"][0]
                confidence = result["scores"][0]                
                if confidence > 0.7:
                    predicted_category = classifier_category
                else:
                    predicted_category = rule_category
            else:
                predicted_category = rule_category
            
            
            classifications.append({
                "sentence": sentence,
                "predicted_category": predicted_category
            })
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {e}")
            classifications.append({
                "sentence": sentence,
                "predicted_category": "error"
            })
    
    return classifications
    
    
def main(pdf_path):
    start_time = time.time()
    
    # Step 1: Extract and clean text
    print("Extracting text from PDF...")
    raw_text = read_pdf(pdf_path)
    text = preprocess_text(raw_text)
    
    # Step 2: Load a more powerful summarization model with increased batch_size
    print("Loading summarization model...")
    device_id = 0 if cuda.is_available() else -1
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",  # You can try other models like T5 or Pegasus here
        device=device_id,
        batch_size=16  # Increased batch_size value for faster throughput
    )
    
    # Step 3: Chunk the text using sentence boundaries
    print("Splitting text into chunks...")
    chunks = split_into_chunks(text, summarizer.tokenizer, max_tokens=512)
    print(f"Total chunks: {len(chunks)}")
    
    # First pass: summarize each chunk in batch mode
    print("Summarizing text chunks...")
    intermediate_summary = summarize(chunks, summarizer)
    
    # Optional: Perform a second summarization pass if the intermediate summary is long
    if len(intermediate_summary.split()) > 500:
        print("Performing second pass summarization...")
        final_summary = summarizer(
            intermediate_summary,
            min_length=150,
            max_length=250,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )[0]['summary_text']
    else:
        final_summary = intermediate_summary
    
    print("Final Summary: \n", final_summary)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")