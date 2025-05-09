from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Extract text from a PDF file in a directory
def extract_text_from_pdf(data):
    # Initialize a loader to read all PDF files in the specified directory
    loader = DirectoryLoader(
        path=data,              # Path to the directory containing PDF files
        glob="*.pdf",           # Match all files with .pdf extension
        loader_cls=PyPDFLoader  # Use PyPDFLoader to parse PDF files
    )
    
    # Load and return the parse documents from the PDF files
    documents = loader.load()
    return documents

# Split extracted documents into smaller text chunks
def split_text(extracted_data):
    # Use RecursiveCharacterTextSplitter to split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Maximum number of characters per chunk
        chunk_overlap=20,     # Number of overlapping characters between chunks
        length_function=len   # Function used to measure the length of each chunk
    )
    
    # Split the documents into chunks and return them
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings