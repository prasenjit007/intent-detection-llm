# app.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import os
import re
import tempfile

# 1. Initialize Groq LLM
llm = ChatGroq(
    api_key="YOUR_GROQ_API_KEY",
    model="llama3-8b-8192"  
)

# 2. Intent detection Prompt
intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an intent classification system.
Analyze the following user message and output only the INTENT name, nothing else.

Possible INTENTS:
- AddNumbers
- SubtractNumbers
- MultiplyNumbers
- BookFlight
- WeatherQuery
- PlayMusic
- TellJoke
- SetReminder
- RAGQuery

User Message: {user_input}

INTENT:
"""
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# 3. Extract numbers for math tasks
def extract_numbers(text):
    return list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", text)))

# 4. Perform tasks based on intent
def perform_task(intent, user_input, rag_chain=None):
    numbers = extract_numbers(user_input)

    if intent == "AddNumbers":
        if len(numbers) >= 2:
            return f"➕ The sum is {sum(numbers)}."
        else:
            return "❗ Provide at least two numbers to add."

    elif intent == "SubtractNumbers":
        if len(numbers) >= 2:
            return f"➖ The difference is {numbers[0] - numbers[1]}."
        else:
            return "❗ Provide two numbers to subtract."

    elif intent == "MultiplyNumbers":
        if len(numbers) >= 2:
            return f"✖️ The product is {numbers[0] * numbers[1]}."
        else:
            return "❗ Provide two numbers to multiply."

    elif intent == "BookFlight":
        return f"📅 Booking a flight with details: '{user_input}'!"

    elif intent == "WeatherQuery":
        return "☁️ Fetching latest weather..."

    elif intent == "PlayMusic":
        return "🎵 Playing some music..."

    elif intent == "TellJoke":
        return "😂 Why did the math book look sad? It had too many problems."

    elif intent == "SetReminder":
        return "⏰ Reminder set!"

    elif intent == "RAGQuery":
        if rag_chain:
            return rag_chain.run(user_input)
        else:
            return "📄 No document uploaded yet. Please upload first."
    else:
        return "🤔 Could not recognize the task."

# 5. Function to create Vectorstore and QA Chain
def create_vectorstore_and_qa(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if suffix.lower() == '.pdf':
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path)

        docs.extend(loader.load())

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)

    # Embedding model (small HF model for local use)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create ChromaDB
    vectordb = Chroma.from_documents(documents, embedding)

    # Retrieval-based QA Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff"
    )

    return rag_chain

# --- Streamlit UI Starts Here ---
st.title("🚀 Groq Agentic App: Intent Detection + RAG Reasoning")
st.write("Enter your message or upload documents for analysis!")

# Document upload
uploaded_files = st.file_uploader(
    "Upload Document(s) (PDF or TXT)", 
    type=['pdf', 'txt'], 
    accept_multiple_files=True
)

rag_chain = None
if uploaded_files:
    with st.spinner("Processing uploaded document(s)..."):
        rag_chain = create_vectorstore_and_qa(uploaded_files)
    st.success("✅ Document(s) processed and ready for reasoning!")

# User input
user_input = st.text_input("Enter your query/message:")

if st.button("Detect & Execute"):
    if not user_input:
        st.warning("⚠️ Please type something.")
    else:
        with st.spinner("Detecting intent..."):
            detected_intent = intent_chain.run(user_input).strip()

        st.success(f"✅ Detected Intent: `{detected_intent}`")

        task_output = perform_task(detected_intent, user_input, rag_chain)
        st.info(task_output)
