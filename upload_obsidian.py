# Let's set the env vars
import os
from dotenv import load_dotenv
load_dotenv()
# Load the API key from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import necessary libraries
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize text splitter
#TODO: Change the separators to not get so many small chunks with headers, etc.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=75, separators=["\n\n", "\n", "(?<=\. )", " ", ""])

import os

# Function to combine multiple collection directories
def collect_documents(data_folder):
    documents = []
    for root, dirs, files in os.walk(data_folder):
        for dir_name in dirs:
            path = os.path.join(root, dir_name)
            loader = ObsidianLoader(path)
            docs = loader.load()  # Load documents
            split_docs = text_splitter.split_documents(docs)  # Split documents using the text_splitter
            documents.extend(split_docs)  # Add these documents to the list
    return documents

# Define your data folder
data_folder = "./data"

# Collect documents from the data folder
docs = collect_documents(data_folder)

# Calculate the number of words total for all documents
total_words = sum(len(doc.page_content.split()) for doc in docs)

# Print the number of documents and total words
print(f"{len(docs)} documents loaded with a total of {total_words:,} words.")

# Embed the docs and add them to Qdrant vector-store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(path="/tmp/obsidian_docs")

client.create_collection(
    collection_name="obsidian_docs",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="obsidian_docs",
    embedding=embeddings,
)

url = "http://qdrant.homehub.tv"
docs = docs  # put docs here
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="obsidian_docs",
)