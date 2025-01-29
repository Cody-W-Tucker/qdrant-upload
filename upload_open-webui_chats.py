from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def collect_documents(loader):
    documents = []
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    documents.extend(split_docs)
    return documents

text_splitter = SemanticChunker(
    OpenAIEmbeddings(model="text-embedding-3-large"), 
    min_chunk_size=512
)

def extract_metadata(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id", ""),
    metadata["parentId"] = record.get("parentId", ""),
    metadata["role"] = record.get("role", ""),
    metadata["modelName"] = record.get("modelName") or record.get("model", "")
    metadata["timestamp"] = record.get("timestamp", "")
    return metadata

loader = JSONLoader(
    file_path='./data/open-webui/open-webui-chat-export.json',
    text_content=False,
    json_lines=True,
    is_content_key_jq_parsable=True,
    content_key='.content',
    jq_schema='.[].chat.messages[]',
    metadata_func=extract_metadata
)

docs = collect_documents(loader)
total_words = sum(len(doc.page_content.split()) for doc in docs)
print(f"{len(docs)} docs loaded with a total of {total_words:,} words.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = QdrantClient(path="/tmp/chat_history")

client.create_collection(
    collection_name="chat_history",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="chat_history",
    embedding=embeddings,
)

batch_size = 100  # Adjust batch size as needed
for i in range(0, len(docs), batch_size):
    batch_docs = docs[i:i+batch_size]
    QdrantVectorStore.from_documents(
        batch_docs,
        embeddings,
        url="http://qdrant.homehub.tv",
        prefer_grpc=False,
        collection_name="chat_history",
    )