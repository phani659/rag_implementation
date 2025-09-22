import os
import logging
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file (for local testing only)
load_dotenv()

app = Flask(__name__)

# --- Global RAG Components ---
# These variables will hold the initialized components
retrieval_chain = None
llm = None
db = None
embeddings = None

# --- Configuration Constants ---
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-pro"

# The prompt template from the notebook
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}"""

def initialize_rag():
    """Initializes the LLM, Embeddings, and loads the pre-built FAISS index."""
    global llm, db, embeddings, retrieval_chain

    # 1. Initialize Gemini LLM
    try:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            logging.warning("GEMINI_API_KEY environment variable not set.")
            # In App Engine, you might rely on Application Default Credentials (ADC)
            # if using Google-native libraries, but since this uses a specific API key
            # and model, we check the environment variable.
            # If deploying to App Engine, ensure GEMINI_API_KEY is set in app.yaml.

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=gemini_api_key,
            # Set temperature low for factual RAG tasks
            temperature=0.1
        )
        logging.info(f"Initialized LLM: {GEMINI_MODEL}")
    except Exception as e:
        logging.error(f"Error initializing Gemini LLM: {e}")
        llm = None
        return # Cannot proceed without LLM

    # 2. Initialize Embeddings and load FAISS DB
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True # Required for loading from disk
        )
        logging.info(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        db = None
        return # Cannot proceed without RAG index

    # 3. Create Retrieval Chain
    try:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        logging.info("Successfully created LangChain Retrieval Chain.")
    except Exception as e:
        logging.error(f"Error creating retrieval chain: {e}")
        retrieval_chain = None

@app.before_first_request
def startup_init():
    """Runs initialization only once when the server starts."""
    initialize_rag()

@app.route('/')
def index():
    """Renders the main web page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles the user's question and invokes the RAG pipeline."""
    if not retrieval_chain:
        return jsonify({"error": "RAG system not fully initialized. Check server logs."}), 500

    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "No question provided."}), 400

    try:
        logging.info(f"Received query: '{query}'")
        # Invoke the chain
        response = retrieval_chain.invoke({"input": query})
        answer = response['answer']
        
        # Optionally, extract sources from the context documents
        source_docs = response.get('context', [])
        sources = [doc.metadata.get('source', 'N/A') for doc in source_docs]

        return jsonify({
            "answer": answer,
            "sources": list(set(sources)) # Unique sources
        })
    except Exception as e:
        logging.error(f"Error during RAG invocation: {e}")
        return jsonify({"error": f"An error occurred while processing the request: {str(e)}"}), 500

if __name__ == '__main__':
    # When running locally, Flask defaults to port 5000
    # In App Engine, it automatically uses the port defined by the environment
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
