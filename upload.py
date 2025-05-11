import os
import sys
import argparse
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, ObsidianLoader, JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from datetime import datetime
import time

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Load configurables from environment variables with defaults
QDRANT_URL = os.getenv("QDRANT_UPLOAD_URL", "http://qdrant.homehub.tv")
DEFAULT_COLLECTION = os.getenv("QDRANT_UPLOAD_COLLECTION", "main")
EMBEDDING_MODEL = os.getenv("QDRANT_UPLOAD_MODEL", "text-embedding-3-large")
VECTOR_DIMENSIONS = int(os.getenv("QDRANT_UPLOAD_DIMENSIONS", "3072"))
DISTANCE_METRIC = os.getenv("QDRANT_UPLOAD_DISTANCE", "Cosine")
BATCH_SIZE = int(os.getenv("QDRANT_UPLOAD_BATCH_SIZE", "100"))
MIN_CONTENT_LENGTH = int(os.getenv("QDRANT_UPLOAD_MIN_LENGTH", "10"))

# Map distance metric string to Qdrant Distance enum
DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Euclid": Distance.EUCLID,
    "Dot": Distance.DOT,
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Upload documents to Qdrant vector database')
parser.add_argument('--collection', type=str, default=DEFAULT_COLLECTION, 
                   help=f'Name of the Qdrant collection to use (defaults to "{DEFAULT_COLLECTION}")')
parser.add_argument('--source', type=str, 
                   help='Custom source identifier for all documents (overrides default source paths)')
parser.add_argument('--type', type=str, required=True, choices=['general', 'obsidian', 'chat'],
                   help='Type of documents to process: general, obsidian, or chat')
parser.add_argument('--dirs', type=str, nargs='+',
                   help='Directories to process (multiple directories can be specified)')
parser.add_argument('--json-file', type=str, 
                   help='Path to JSON file (for chat type only)')
parser.add_argument('--skip-existing', action='store_true',
                   help='Skip documents that already exist in the collection')
parser.add_argument('--force-update', action='store_true',
                   help='Force update all documents even if unchanged')
args = parser.parse_args()

# Use QDRANT_FOLDERS environment variable as default if --dirs not specified
if args.dirs is None and os.environ.get('QDRANT_FOLDERS'):
    # Split the environment variable on spaces
    qdrant_folders = os.environ.get('QDRANT_FOLDERS').split()
    if qdrant_folders:
        print(f"Using default directories from QDRANT_FOLDERS environment variable:")
        for dir in qdrant_folders:
            print(f"  - {dir}")
        args.dirs = qdrant_folders

# Initialize the text splitter
text_splitter = SemanticChunker(
    OpenAIEmbeddings(model=EMBEDDING_MODEL),
    sentence_split_regex=(
        r'(?<=[.?!])\s+(?![^[]*\])|'  # Sentence endings not in brackets (e.g., for Obsidian [[links]])
        r'(?<=\n#)\s|(?<=\n##)\s|(?<=\n###)\s|(?<=\n####)\s|(?<=\n#####)\s|(?<=\n######)\s|'  # Headers
        r'(?<=\n-)\s|'  # Unordered lists
        r'(?<=\n\d\.)\s|'  # Ordered lists
        r'(?<=``````)\n'  # Code blocks
    )
)

# Function to extract metadata for chat documents
def extract_chat_metadata(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id", "")
    metadata["parentId"] = record.get("parentId", "")
    metadata["role"] = record.get("role", "")
    metadata["modelName"] = record.get("modelName") or record.get("model", "")
    metadata["timestamp"] = record.get("timestamp", "")
    if args.source:
        metadata["source"] = args.source
    else:
        metadata["source"] = f"chat_{record.get('id', 'unknown')}"
    return metadata

# Function to load general documents from multiple directories
def collect_general_documents(directories, custom_source=None):
    documents = []
    
    # Process each directory individually
    for directory_path in directories:
        if not os.path.exists(directory_path):
            print(f"Warning: Directory '{directory_path}' does not exist. Skipping.")
            continue
            
        try:
            # Find all markdown files in the directory
            markdown_files = []
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.endswith('.md'):
                        markdown_files.append(os.path.join(root, file))
            
            # Process markdown files with TextLoader
            md_docs = []
            for file_path in markdown_files:
                try:
                    loader = TextLoader(file_path)
                    file_docs = loader.load()
                    md_docs.extend(file_docs)
                    print(f"Loaded Markdown file: {file_path}")
                except Exception as e:
                    print(f"Error loading Markdown file {file_path}: {e}")
            
            # Try to process other file types with DirectoryLoader if possible
            other_docs = []
            try:
                # Exclude markdown files (we already processed them)
                loader = DirectoryLoader(
                    directory_path,
                    glob="**/*",
                    exclude=["**/*.md"],
                    recursive=True,
                    show_progress=True,
                    silent_errors=True
                )
                other_docs = loader.load()
                print(f"Loaded {len(other_docs)} non-Markdown documents from {directory_path}")
            except Exception as e:
                print(f"Warning: Could not process non-Markdown files: {e}")
                print("Continuing with Markdown files only.")
                
            # Combine all documents
            docs = md_docs + other_docs
            
            if not docs:
                print(f"No documents could be loaded from '{directory_path}'.")
                continue
                
            # Add source directory information to metadata
            for doc in docs:
                if not doc.metadata:
                    doc.metadata = {}
                # Apply custom source if provided, otherwise use file path
                if custom_source:
                    doc.metadata['source'] = custom_source
                elif 'source' not in doc.metadata:
                    doc.metadata['source'] = doc.metadata.get('path', 'unknown')
                # Add directory source information
                doc.metadata['source_directory'] = directory_path
                # Add current timestamp if not present
                if 'last_modified' not in doc.metadata:
                    doc.metadata['last_modified'] = time.time()
            
            split_docs = text_splitter.split_documents(docs)  # Split documents
            
            # Ensure chunk metadata has proper tracking
            for i, chunk in enumerate(split_docs):
                if not chunk.metadata:
                    chunk.metadata = {}
                # Preserve source information after splitting
                if custom_source:
                    chunk.metadata['source'] = custom_source
                elif 'source' not in chunk.metadata:
                    chunk.metadata['source'] = chunk.metadata.get('path', 'unknown')
                # Add chunk number for better tracking
                chunk.metadata['chunk_id'] = i
            
            documents.extend(split_docs)
            print(f"Loaded {len(split_docs)} documents from {directory_path}")
        except Exception as e:
            print(f"Error loading documents from {directory_path}: {e}")
            
    return documents

# Function to load Obsidian documents from multiple directories
def collect_obsidian_documents(directories, custom_source=None):
    documents = []
    
    # Process each directory individually
    for directory_path in directories:
        if not os.path.exists(directory_path):
            print(f"Warning: Directory '{directory_path}' does not exist. Skipping.")
            continue
            
        try:
            loader = ObsidianLoader(directory_path)
            docs = loader.load()  # Load documents from the folder
            
            # Add source info to documents
            for doc in docs:
                if not doc.metadata:
                    doc.metadata = {}
                if custom_source:
                    doc.metadata['source'] = custom_source
                elif 'source' not in doc.metadata:
                    doc.metadata['source'] = doc.metadata.get('path', 'unknown')
                doc.metadata['source_directory'] = directory_path
            
            documents.extend(docs)
            print(f"Loaded {len(docs)} documents from {directory_path}")
        except Exception as e:
            print(f"Error loading documents from {directory_path}: {e}")
            
    return documents

# Function to load chat documents
def collect_chat_documents(json_file):
    if not os.path.exists(json_file):
        print(f"Warning: JSON file '{json_file}' does not exist. No documents will be loaded.")
        return []
    
    try:
        loader = JSONLoader(
            file_path=json_file,
            text_content=False,
            json_lines=True,
            is_content_key_jq_parsable=True,
            content_key='.content',
            jq_schema='.[].chat.messages[]',
            metadata_func=extract_chat_metadata
        )
        
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        print(f"Loaded {len(split_docs)} documents from {json_file}")
        return split_docs
    except Exception as e:
        print(f"Error loading documents from {json_file}: {e}")
        return []

# Function to filter out empty or very short documents
def filter_documents(docs, min_content_length=MIN_CONTENT_LENGTH):
    filtered_docs = []
    skipped = 0
    
    for doc in docs:
        if not doc.page_content or len(doc.page_content.strip()) < min_content_length:
            skipped += 1
            continue
        filtered_docs.append(doc)
    
    if skipped > 0:
        print(f"Skipped {skipped} documents with insufficient content (less than {min_content_length} chars)")
    
    return filtered_docs

# Validate required arguments
directories_to_process = args.dirs or []
if args.type in ['general', 'obsidian'] and not directories_to_process:
    print(f"Error: No directories specified for {args.type} document type. Use --dirs to specify directories.")
    exit(1)

if args.type == 'chat' and not args.json_file:
    print("Error: --json-file is required for chat document type")
    exit(1)

# Load documents based on the specified type
if args.type == 'general':
    docs = collect_general_documents(directories_to_process, args.source)
elif args.type == 'obsidian':
    docs = collect_obsidian_documents(directories_to_process, args.source)
elif args.type == 'chat':
    docs = collect_chat_documents(args.json_file)
else:
    print(f"Error: Unknown document type: {args.type}")
    exit(1)

# Filter out empty documents
docs = filter_documents(docs)

# Calculate total words and print stats
total_words = sum(len(doc.page_content.split()) for doc in docs)
print(f"{len(docs)} documents loaded with a total of {total_words:,} words.")

# Exit early if no documents were loaded
if not docs:
    print("No documents to upload. Exiting.")
    exit(1)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Set up Qdrant client and collection
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Create collection if it doesn't exist
collection_name = args.collection
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists. Adding documents to it.")
except Exception as e:
    if "Connection refused" in str(e):
        print(f"Error: Could not connect to Qdrant server at {QDRANT_URL}")
        print("Make sure the Qdrant server is running and accessible.")
        exit(1)
    
    print(f"Creating new collection '{collection_name}'.")
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIMENSIONS, distance=DISTANCE_MAP[DISTANCE_METRIC]),
        )
    except Exception as create_error:
        if "Connection refused" in str(create_error):
            print(f"Error: Could not connect to Qdrant server at {QDRANT_URL}")
            print("Make sure the Qdrant server is running and accessible.")
            exit(1)
        else:
            raise

# Initialize vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# Process documents based on their existence and modification time
if args.type == 'obsidian':
    print("Processing Obsidian documents with update checks...")
    
    # Load documents without chunking first
    raw_docs = collect_obsidian_documents(directories_to_process, args.source)
    
    # Group documents by source to handle all chunks of the same document together
    docs_by_source = {}
    for doc in raw_docs:
        source = doc.metadata.get('source')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    # Stats
    total_processed = 0
    total_skipped = 0
    total_updated = 0
    total_added = 0
    
    # Collect documents that need processing
    docs_to_chunk = []
    
    # Process each source document
    for source, source_docs in docs_by_source.items():
        needs_processing = True
        
        if args.skip_existing:
            # Check if documents with this source already exist
            search_results = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if search_results[0]:  # If any documents found
                print(f"Skipping existing document: {source}")
                total_skipped += len(source_docs)
                needs_processing = False
        
        elif not args.force_update:
            # Check if documents need updates based on modification time
            search_results = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if search_results[0]:  # If any documents found
                # Get last_modified from the local document
                local_last_modified = source_docs[0].metadata.get('last_modified', 0)
                
                # Get last_modified from the stored document
                stored_meta = search_results[0][0].payload.get('metadata', {})
                stored_last_modified = stored_meta.get('last_modified', 0)
                
                if local_last_modified <= stored_last_modified:
                    print(f"Skipping unchanged document: {source}")
                    total_skipped += len(source_docs)
                    needs_processing = False
        
        if needs_processing:
            # If we need to process this document, add it to our chunking list
            docs_to_chunk.extend(source_docs)
            
            # First, delete any existing versions
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        )
                    ]
                )
            )
            
            # Update stats
            if search_results[0] if 'search_results' in locals() else False:
                print(f"Updated document: {source}")
                total_updated += len(source_docs)
            else:
                print(f"Added new document: {source}")
                total_added += len(source_docs)
            
            total_processed += len(source_docs)
    
    # Only chunk and embed documents that need processing
    if docs_to_chunk:
        print(f"Chunking and embedding {len(docs_to_chunk)} documents...")
        split_docs = text_splitter.split_documents(docs_to_chunk)
        
        # Now add the processed documents
        vector_store.add_documents(split_docs)
    else:
        split_docs = []
    
    # Print final stats
    print(f"\nObsidian document processing complete:")
    print(f"  - New documents added: {total_added}")
    print(f"  - Existing documents updated: {total_updated}")
    print(f"  - Documents skipped (unchanged): {total_skipped}")
    print(f"  - Total chunks processed: {len(split_docs)}")
    
else:
    # For non-Obsidian documents, just add them directly
    print(f"Uploading {len(docs)} documents to collection '{collection_name}'...")
    
    # Use batch uploading for efficiency
    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i+BATCH_SIZE]
        vector_store.add_documents(batch_docs)
        print(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(docs)-1)//BATCH_SIZE + 1} ({len(batch_docs)} documents)")

print(f"Successfully processed {len(docs)} documents to Qdrant collection '{collection_name}' at {QDRANT_URL}.")
if args.source:
    print(f"All documents tagged with custom source: '{args.source}'") 