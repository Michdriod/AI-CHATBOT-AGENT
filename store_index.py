from src.helper import extract_text_from_pdf, split_text, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables from a .env file into the environment
load_dotenv()


# Get the Pinecone API key from the environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = extract_text_from_pdf(data="Data/")    # Extract text from all PDF files located in the "Data/" directory
text_chunks = split_text(extracted_data)                # Split the extracted text into smaller chunks to prepare for embedding
embeddings = download_hugging_face_embeddings()         # Download HuggingFace transformer model for generating embeddings

# Initialize the Pinecone client with the API key
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"  # Define the name of the index to be created in Pinecone

# Create a new Pinecone index
pc.create_index(
    name=index_name,
    dimension=384,  # Dimension of the embeddings
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Generate embeddings for each text chunk and upload (upsert) them into the Pinecone index 
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)