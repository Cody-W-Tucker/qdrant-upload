import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ObsidianLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize the text splitter
text_splitter = SemanticChunker(
    OpenAIEmbeddings(model="text-embedding-3-large"),
    min_chunk_size=512,
    sentence_split_regex=(
        r'(?<=[.?!])\s+(?![^[]*\])|'  # Sentence endings not in brackets (for Obsidian [[links]])
        r'(?<=\n#)\s|(?<=\n##)\s|(?<=\n###)\s|(?<=\n####)\s|(?<=\n#####)\s|(?<=\n######)\s|'  # Headers
        r'(?<=\n-)\s|'  # Unordered lists
        r'(?<=\n\d\.)\s|'  # Ordered lists
        r'(?<=``````)\n'  # Code blocks
    )
)

# Function to load, split, and return documents from a folder
def collect_documents(data_folder):
    documents = []
    if not os.path.exists(data_folder):
        print(f"Warning: Data folder '{data_folder}' does not exist. No documents will be loaded.")
        return documents

    # Walk through the data folder (which contains symlinked directories)
    for root, dirs, files in os.walk(data_folder):
        for dir_name in dirs:
            path = os.path.join(root, dir_name)
            try:
                loader = ObsidianLoader(path)
                docs = loader.load()  # Load documents from the symlinked folder
                split_docs = text_splitter.split_documents(docs)  # Split documents
                documents.extend(split_docs)
                print(f"Loaded {len(split_docs)} documents from {path}")
            except Exception as e:
                print(f"Error loading documents from {path}: {e}")
    return documents

# Define the data folder based on the Nix flake's symlink location
data_folder = "./data/upload"

# Collect documents from the data folder
docs = collect_documents(data_folder)

# Calculate total words and print stats
total_words = sum(len(doc.page_content.split()) for doc in docs)
print(f"{len(docs)} documents loaded with a total of {total_words:,} words.")

# Exit early if no documents were loaded
if not docs:
    print("No documents to upload. Exiting.")
    exit(1)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set up Qdrant client and collection
qdrant_url = "http://qdrant.homehub.tv"
client = QdrantClient(url=qdrant_url, prefer_grpc=False)

# Create collection if it doesn’t exist (3072 is the dimension for text-embedding-3-large)
collection_name = "obsidian_docs"
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists. Adding documents to it.")
except Exception:
    print(f"Creating new collection '{collection_name}'.")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

# Upload documents to Qdrant
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=qdrant_url,
    prefer_grpc=False,
    collection_name=collection_name,
)

print(f"Successfully uploaded {len(docs)} documents to Qdrant at {qdrant_url}.")