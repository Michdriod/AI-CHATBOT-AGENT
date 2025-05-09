from flask import Flask, render_template, jsonify, request  # Flask is used to create the web app
from src.helper import download_hugging_face_embeddings  # Custom helper function to download embeddings
from langchain_pinecone import PineconeVectorStore  # Pinecone vector storage integration with LangChain
from langchain_groq import ChatGroq  # Groq language model integration
from langchain.chains import create_retrieval_chain  # For creating a RAG (Retrieval-Augmented Generation) chain
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines documents into answers
from langchain_core.prompts import ChatPromptTemplate  # Used to create prompt templates
from dotenv import load_dotenv  # Loads environment variables from a .env file
from src.prompt import *  # Imports the custom system prompt
import os  # To access environment variables

# Initialize a Flask web app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Pinecone and Groq API keys from environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Set the environment variables manually (optional redundancy)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Download HuggingFace embeddings model to be used for vectorization
embeddings = download_hugging_face_embeddings()

# Name of the Pinecone index previously created
index_name = "medicalbot"

# Load the existing Pinecone vector store using the given index and embeddings
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Convert the vector store to a retriever for similarity-based document search
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the Groq LLM with specific settings
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Choose Groq's model (could also use other versions)
    temperature=0.4,  # Controls randomness in output
    max_retries=2,  # How many times to retry in case of failure
    n=1,  # Number of completions to return
    max_tokens=500  # Max length of the response
)

# Create a prompt template with system and human message structure
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),  # System-level instructions (from src/prompt.py)
        ("human", "{input}"),  # Placeholder for user input
    ]
)

# Chain that combines retrieved documents and passes them through the LLM for an answer
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# RAG chain that handles both retrieval and answering
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Define the homepage route
@app.route("/")
def index():
    return render_template("chat.html")  # Renders the front-end chat interface

# Define the chat route (called when user sends a message)
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]  # Get the user message from the HTML form
    input = msg  # Assign it to 'input' (same as msg)
    print(input)  # Log input
    response = rag_chain.invoke({"input": msg})  # Call the RAG chain with the input
    print("Response: ", response["answer"])  # Log the answer
    return str(response["answer"])  # Return answer to front-end as string

# Run the app on 0.0.0.0 so it can be accessed from other devices, port 8080, with debug on
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
