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
from qdrant_client.http.exceptions import UnexpectedResponse
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

# Function to load chat documents (modified for streaming)
def collect_chat_documents(json_file, batch_size_for_chunking=100):
    if not os.path.exists(json_file):
        print(f"Warning: JSON file '{json_file}' does not exist. No documents will be loaded.")
        return iter([]) # Return an empty iterator
    
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
        
        doc_iterator = loader.lazy_load() # Use lazy_load for an iterator
        
        batch_for_chunking = []
        raw_docs_processed_count = 0

        print(f"Starting to stream and process documents from {json_file}...")

        for doc in doc_iterator:
            batch_for_chunking.append(doc)
            if len(batch_for_chunking) >= batch_size_for_chunking:
                current_raw_batch_size = len(batch_for_chunking)
                print(f"Collected {current_raw_batch_size} raw messages. Preparing to chunk...")
                
                start_chunk_time = time.time()
                split_docs = text_splitter.split_documents(batch_for_chunking)
                end_chunk_time = time.time()
                print(f"Finished chunking {current_raw_batch_size} raw messages in {end_chunk_time - start_chunk_time:.2f} seconds. Found {len(split_docs)} initial chunks. Filtering...")
                
                start_filter_time = time.time()
                filtered_split_docs = filter_documents(split_docs)
                end_filter_time = time.time()
                num_yielded = len(filtered_split_docs) if filtered_split_docs else 0
                print(f"Finished filtering in {end_filter_time - start_filter_time:.2f} seconds. Yielding {num_yielded} chunks.")

                if filtered_split_docs:
                    yield filtered_split_docs
                raw_docs_processed_count += current_raw_batch_size
                batch_for_chunking = [] # Reset batch

        # Process any remaining documents in the last batch
        if batch_for_chunking:
            current_raw_batch_size = len(batch_for_chunking)
            print(f"Collected final batch of {current_raw_batch_size} raw messages. Preparing to chunk...")
            
            start_chunk_time = time.time()
            split_docs = text_splitter.split_documents(batch_for_chunking)
            end_chunk_time = time.time()
            print(f"Finished chunking final {current_raw_batch_size} raw messages in {end_chunk_time - start_chunk_time:.2f} seconds. Found {len(split_docs)} initial chunks. Filtering...")
            
            start_filter_time = time.time()
            filtered_split_docs = filter_documents(split_docs)
            end_filter_time = time.time()
            num_yielded = len(filtered_split_docs) if filtered_split_docs else 0
            print(f"Finished filtering final batch in {end_filter_time - start_filter_time:.2f} seconds. Yielding {num_yielded} chunks.")

            if filtered_split_docs:
                yield filtered_split_docs
            raw_docs_processed_count += current_raw_batch_size

        print(f"Finished streaming and processing. Total raw chat messages processed from {json_file}: {raw_docs_processed_count}")

    except Exception as e:
        print(f"Error during streaming chat documents from {json_file}: {e}")
        return iter([]) # Ensure it still returns an iterable in case of error

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

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Set up Qdrant client and collection
print(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Check Qdrant connection and server status
try:
    # Try a basic operation like listing collections to verify connection
    client.get_collections() # This will raise an exception if the server is not reachable
    print(f"Successfully connected to Qdrant server at {QDRANT_URL} and was able to list collections.")
except Exception as conn_e:
    print(f"Error: Could not connect to or communicate effectively with Qdrant server at {QDRANT_URL}.")
    print(f"Details: {conn_e}")
    print("Please ensure the Qdrant server is running, accessible, and properly configured.")
    exit(1)

collection_name = args.collection
try:
    print(f"Checking for existing collection: '{collection_name}'...")
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists. Adding documents to it.")
except UnexpectedResponse as e:
    if e.status_code == 404: # Not found
        print(f"Collection '{collection_name}' not found. Creating new collection.")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_DIMENSIONS, distance=DISTANCE_MAP[DISTANCE_METRIC]),
            )
            print(f"Successfully created collection '{collection_name}'.")
        except Exception as create_error:
            print(f"Error: Could not create collection '{collection_name}'.")
            print(f"Details: {create_error}")
            exit(1)
    else:
        print(f"Error: An unexpected issue occurred while checking for collection '{collection_name}'.")
        print(f"Details: {e}")
        exit(1)
except Exception as e: # Catch other potential errors like network issues during get_collection
    print(f"Error: Could not verify collection '{collection_name}' due to an unexpected error.")
    print(f"Details: {e}")
    exit(1)

# Initialize vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

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
    docs = filter_documents(docs) # Filter empty/short documents

    if not docs:
        print("No general documents to upload after filtering. Exiting.")
        exit(0)

    total_words = sum(len(doc.page_content.split()) for doc in docs)
    print(f"{len(docs)} general documents loaded with a total of {total_words:,} words.")
    
    print(f"Uploading {len(docs)} general documents to collection '{args.collection}'...")
    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i+BATCH_SIZE]
        vector_store.add_documents(batch_docs)
        print(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(docs)-1)//BATCH_SIZE + 1} ({len(batch_docs)} documents)")
    
    print(f"Successfully processed {len(docs)} general documents to Qdrant collection '{args.collection}' at {QDRANT_URL}.")

elif args.type == 'obsidian':
    print(f"Processing Obsidian documents with update checks from: {directories_to_process}")
    
    raw_docs = collect_obsidian_documents(directories_to_process, args.source)
    # Note: Filtering for Obsidian docs is handled implicitly by checking content for updates/existence.
    # If needed, filter_documents can be applied to raw_docs before the docs_by_source grouping.

    if not raw_docs:
        print("No Obsidian documents found to process. Exiting.")
        exit(0)

    docs_by_source = {}
    for doc in raw_docs:
        source = doc.metadata.get('source')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
    
    total_processed_sources = 0
    total_skipped_sources = 0
    total_updated_sources = 0
    total_added_sources = 0
    total_chunks_uploaded = 0
    
    docs_to_chunk_and_upload = []
    
    for source, source_docs in docs_by_source.items():
        needs_processing = True
        existing_doc_found = False
        
        # Pre-filter the source_docs to remove very short content before expensive DB checks
        # We do this on a per-source basis to keep the logic contained.
        filtered_source_docs = filter_documents(source_docs)
        if not filtered_source_docs:
            print(f"Skipping source '{source}' as all its content is too short after filtering.")
            total_skipped_sources +=1
            continue

        if args.skip_existing:
            search_results = client.scroll(
                collection_name=args.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=1,
                with_payload=False, # No need for payload here, just existence check
                with_vectors=False
            )
            
            if search_results[0]:
                print(f"Skipping existing document source: {source}")
                total_skipped_sources += 1
                needs_processing = False
        
        elif not args.force_update:
            search_results = client.scroll(
                collection_name=args.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=1,
                with_payload=True, # Need payload for last_modified
                with_vectors=False
            )
            
            if search_results[0]:
                existing_doc_found = True
                local_last_modified = filtered_source_docs[0].metadata.get('last_modified', 0)
                stored_meta = search_results[0][0].payload.get('metadata', {})
                stored_last_modified = stored_meta.get('last_modified', 0)
                
                if local_last_modified <= stored_last_modified:
                    print(f"Skipping unchanged document source: {source}")
                    total_skipped_sources += 1
                    needs_processing = False
        
        if needs_processing:
            # Delete existing points for this source before adding new/updated ones
            client.delete(
                collection_name=args.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source)
                        )
                    ]
                )
            )
            
            docs_to_chunk_and_upload.extend(filtered_source_docs)
            
            if existing_doc_found:
                print(f"Marked document source for update: {source}")
                total_updated_sources += 1
            else:
                print(f"Marked new document source for adding: {source}")
                total_added_sources += 1
            total_processed_sources +=1
    
    if docs_to_chunk_and_upload:
        print(f"Chunking and embedding {len(docs_to_chunk_and_upload)} documents from {total_processed_sources} sources...")
        # ObsidianLoader already produces fairly well-defined documents; splitting further might be too granular.
        # However, SemanticChunker can help if individual obsidian notes are very long.
        # For now, we assume ObsidianLoader's output is fine for direct add, or that SemanticChunker is applied in collect_obsidian_documents if desired.
        # If SemanticChunker needs to be applied here, it would be:
        # split_docs_for_upload = text_splitter.split_documents(docs_to_chunk_and_upload)
        # For now, let's assume docs_to_chunk_and_upload are already appropriately chunked or don't need further semantic splitting.
        
        # The original script splits obsidian docs in collect_obsidian_documents. Let's ensure that happens or adjust.
        # Rereading the original: collect_obsidian_documents DOES NOT split. It loads, adds metadata.
        # The split happens later in the original script IF type is obsidian, after grouping.
        # This was an oversight in my previous analysis. Let's apply splitting here.
        
        split_docs_for_upload = text_splitter.split_documents(docs_to_chunk_and_upload)
        total_chunks_uploaded = len(split_docs_for_upload)

        print(f"Uploading {total_chunks_uploaded} chunks from {total_processed_sources} sources...")
        for i in range(0, len(split_docs_for_upload), BATCH_SIZE):
            batch_docs = split_docs_for_upload[i:i+BATCH_SIZE]
            vector_store.add_documents(batch_docs)
            print(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(split_docs_for_upload)-1)//BATCH_SIZE + 1} ({len(batch_docs)} chunks)")
    else:
        print("No Obsidian documents required chunking or uploading.")
    
    print(f"\nObsidian document processing complete:")
    print(f"  - New document sources added: {total_added_sources}")
    print(f"  - Existing document sources updated: {total_updated_sources}")
    print(f"  - Document sources skipped (unchanged or already exists): {total_skipped_sources}")
    print(f"  - Total chunks uploaded: {total_chunks_uploaded}")
    print(f"Successfully processed Obsidian documents to Qdrant collection '{args.collection}' at {QDRANT_URL}.")

elif args.type == 'chat':
    print(f"Processing chat documents from: {args.json_file} (streaming)")
    # collect_chat_documents is now a generator yielding batches of *already filtered and split* docs
    doc_batch_generator = collect_chat_documents(args.json_file, batch_size_for_chunking=100) # Internal batch size for chunking
    
    total_chunks_uploaded_chat = 0
    batch_num = 0
    any_docs_processed = False

    for doc_batch in doc_batch_generator: # doc_batch is a list of filtered, split documents
        if not doc_batch: # Should not happen if filter_documents works correctly with yield
            continue

        any_docs_processed = True
        batch_num += 1
        num_docs_in_batch = len(doc_batch)
        total_chunks_uploaded_chat += num_docs_in_batch
        
        print(f"Uploading chat document batch {batch_num} with {num_docs_in_batch} chunks to collection '{args.collection}'...")
        # Qdrant's add_documents can handle a list of documents directly for batching.
        # The BATCH_SIZE constant is for Qdrant client library internal batching, not what we do here.
        # However, the vector_store.add_documents method uses its own batching if the list is large.
        # We are essentially feeding it pre-batched chunks from our generator.
        vector_store.add_documents(doc_batch) # This will further batch if doc_batch is > BATCH_SIZE
        print(f"Uploaded chat batch {batch_num} ({num_docs_in_batch} chunks).")

    if not any_docs_processed:
        print("No chat documents were processed or yielded from the generator. Exiting.")
        exit(0)

    print(f"\nChat document streaming and upload complete:")
    print(f"  - Total chunks uploaded: {total_chunks_uploaded_chat}")
    print(f"Successfully processed chat documents to Qdrant collection '{args.collection}' at {QDRANT_URL}.")

else:
    print(f"Error: Unknown document type specified: {args.type}")
    exit(1)

# General success message parts, can be enhanced per type if needed
if args.source:
    print(f"All documents processed were intended to be tagged with custom source: '{args.source}'") 