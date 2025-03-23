import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
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
        r'(?<=[.?!])\s+(?![^[]*\])|'  # Sentence endings not in brackets (e.g., for Obsidian [[links]])
        r'(?<=\n#)\s|(?<=\n##)\s|(?<=\n###)\s|(?<=\n####)\s|(?<=\n#####)\s|(?<=\n######)\s|'  # Headers
        r'(?<=\n-)\s|'  # Unordered lists
        r'(?<=\n\d\.)\s|'  # Ordered lists
        r'(?<=``````)\n'  # Code blocks
    )
)

# Function to load, split, and return documents from symlinked folders
def collect_documents(data_folder):
    documents = []
    if not os.path.exists(data_folder):
        print(f"Warning: Data folder '{data_folder}' does not exist. No documents will be loaded.")
        return documents

    # Get list of symlinked directories in data_folder
    symlinked_dirs = [os.path.join(data_folder, d) for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    if not symlinked_dirs:
        print(f"No symlinked directories found in '{data_folder}'.")
        return documents

    # Process each symlinked directory individually
    for symlink_path in symlinked_dirs:
        try:
            # Use DirectoryLoader on the symlinked folder
            loader = DirectoryLoader(
                symlink_path,
                glob="**/*",  # Load all files
                recursive=True,        # Recursively search within each symlinked folder
                show_progress=True,
                silent_errors=True     # Skip files that fail to load
            )
            docs = loader.load()  # Load documents from this symlink
            if not docs:
                print(f"No supported files found in '{symlink_path}'.")
                continue

            split_docs = text_splitter.split_documents(docs)  # Split documents
            documents.extend(split_docs)
            print(f"Loaded {len(split_docs)} documents from {symlink_path}")
        except Exception as e:
            print(f"Error loading documents from {symlink_path}: {e}")
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
# Get collection name from command-line argument or default
collection_name = sys.argv[1] if len(sys.argv) > 1 else "general_docs"
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