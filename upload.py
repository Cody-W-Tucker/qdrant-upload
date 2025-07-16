import os
import argparse
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, ObsidianLoader, JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
import time
import logging
from typing import List, Iterator, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Load configurables from environment variables with defaults
QDRANT_URL = os.getenv("QDRANT_UPLOAD_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_UPLOAD_COLLECTION", "inbox")
EMBEDDING_MODEL = os.getenv("QDRANT_UPLOAD_MODEL", "nomic-embed-text:latest")
VECTOR_DIMENSIONS = int(os.getenv("QDRANT_UPLOAD_DIMENSIONS", "768"))
DISTANCE_METRIC = os.getenv("QDRANT_UPLOAD_DISTANCE", "Cosine")
BATCH_SIZE = int(os.getenv("QDRANT_UPLOAD_BATCH_SIZE", "2000"))  # Optimized for RTX 3070
MIN_CONTENT_LENGTH = int(os.getenv("QDRANT_UPLOAD_MIN_LENGTH", "50"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Async processing configuration
MAX_CONCURRENT_BATCHES = int(os.getenv("QDRANT_UPLOAD_MAX_CONCURRENT", "4"))  # Number of concurrent processing batches
ENABLE_ASYNC_CHAT = os.getenv("QDRANT_UPLOAD_ASYNC_CHAT", "true").lower() == "true"  # Enable async chat processing

# Performance Notes:
# - QDRANT_UPLOAD_MAX_CONCURRENT: Controls how many batches process simultaneously (4 is optimal for RTX 3070)
# - QDRANT_UPLOAD_ASYNC_CHAT: Enables concurrent processing pipeline for 3-5x faster chat uploads
# - Larger BATCH_SIZE + async processing = maximum GPU utilization and throughput

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
parser.add_argument('--pair-messages', action='store_true', default=True,
                   help='Group user and assistant messages into pairs for better context (default: True)')
parser.add_argument('--individual-messages', action='store_true',
                   help='Process each message individually instead of pairing (overrides --pair-messages)')
args = parser.parse_args()

# Handle message pairing logic
if args.individual_messages:
    args.pair_messages = False

# Use QDRANT_FOLDERS environment variable as default if --dirs not specified
if args.dirs is None and os.environ.get('QDRANT_FOLDERS'):
    # Split the environment variable on spaces
    qdrant_folders = os.environ.get('QDRANT_FOLDERS').split()
    if qdrant_folders:
        logger.info("Using default directories from QDRANT_FOLDERS environment variable:")
        for dir in qdrant_folders:
            logger.info(f"  - {dir}")
        args.dirs = qdrant_folders

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]
markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)

CHUNK_SIZE = int(os.getenv("QDRANT_UPLOAD_CHUNK_SIZE", "2500"))  # Increased for better chat message completeness  
CHUNK_OVERLAP = int(os.getenv("QDRANT_UPLOAD_CHUNK_OVERLAP", "200"))  # Proportional increase
USE_SEMANTIC_CHUNKER = os.getenv("QDRANT_UPLOAD_SEMANTIC_CHUNKER", "false").lower()  # auto=chat only, true=all, false=none (default: false for performance)

recursive_character_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Semantic chunker will be initialized after embeddings are available

# Function to intelligently split documents using either semantic or recursive character splitting
def split_documents_intelligently(docs, document_type="general"):
    """
    Split documents using semantic chunker if enabled, otherwise use recursive character splitter.
    Auto mode: semantic chunker for chat documents, recursive for others.
    """
    should_use_semantic = False
    
    if USE_SEMANTIC_CHUNKER == "true":
        should_use_semantic = True
    elif USE_SEMANTIC_CHUNKER == "auto":
        should_use_semantic = (document_type == "chat")
    elif USE_SEMANTIC_CHUNKER == "false":
        should_use_semantic = False
    else:  # Default to auto mode
        should_use_semantic = (document_type == "chat")
    
    if should_use_semantic and semantic_splitter:
        try:
            return semantic_splitter.split_documents(docs)
        except Exception as e:
            logger.warning(f"Semantic chunker failed, falling back to recursive character splitter: {e}")
            return recursive_character_splitter.split_documents(docs)
    else:
        return recursive_character_splitter.split_documents(docs)

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
    all_final_split_documents = []
    
    # Process each directory individually
    for directory_path in directories:
        if not os.path.exists(directory_path):
            logger.warning(f"Warning: Directory '{directory_path}' does not exist. Skipping.")
            continue
        
        # Load Markdown files
        markdown_file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.md'):
                    markdown_file_paths.append(os.path.join(root, file))
        
        for md_file_path in markdown_file_paths:
            try:
                loader = TextLoader(md_file_path)
                loaded_file_doc_list = loader.load()
                if not loaded_file_doc_list:
                    continue
                
                original_doc = loaded_file_doc_list[0]
                base_metadata = original_doc.metadata.copy() if original_doc.metadata else {}
                base_metadata['original_file_path'] = md_file_path

                if custom_source:
                    base_metadata['source'] = custom_source
                elif 'source' not in base_metadata : # TextLoader puts filename in 'source'
                    base_metadata['source'] = os.path.basename(md_file_path)
                
                base_metadata['source_directory'] = directory_path
                try:
                    base_metadata['last_modified'] = os.path.getmtime(md_file_path)
                except OSError:
                    base_metadata['last_modified'] = time.time()

                header_split_docs = markdown_header_splitter.split_text(original_doc.page_content)
                
                docs_for_recursive_split = []
                for h_split_doc in header_split_docs:
                    merged_metadata = base_metadata.copy()
                    merged_metadata.update(h_split_doc.metadata) # h_split_doc.metadata has 'Header N'
                    h_split_doc.metadata = merged_metadata
                    docs_for_recursive_split.append(h_split_doc)

                final_splits_for_file = split_documents_intelligently(docs_for_recursive_split, "general")
                
                for i, chunk in enumerate(final_splits_for_file):
                    if not chunk.metadata: chunk.metadata = {}
                    chunk.metadata['chunk_id'] = f"{base_metadata.get('source', 'unknown_md_file')}_md_part_{i}"
                
                all_final_split_documents.extend(final_splits_for_file)
                logger.info(f"Processed Markdown file: {md_file_path}, found {len(final_splits_for_file)} chunks.")
            except Exception as e:
                logger.error(f"Error processing Markdown file {md_file_path}: {e}")

        # Load other file types (non-Markdown)
        try:
            # Exclude markdown files as they are already processed
            dir_loader = DirectoryLoader(
                directory_path,
                glob="**/*",
                exclude=["**/*.md"], # Ensure we don't double process
                recursive=True,
                show_progress=True,
                silent_errors=True
            )
            other_raw_docs = dir_loader.load()
            
            processed_other_docs_for_dir = []
            if other_raw_docs:
                docs_for_recursive_split_others = []
                for doc in other_raw_docs:
                    base_metadata = doc.metadata.copy() if doc.metadata else {}
                    
                    # DirectoryLoader 'source' is usually path relative to loaded dir or absolute.
                    # Let's try to ensure original_file_path is absolute or clearly identifiable.
                    file_path_from_meta = base_metadata.get('source', base_metadata.get('path'))
                    if file_path_from_meta:
                        if not os.path.isabs(file_path_from_meta):
                           base_metadata['original_file_path'] = os.path.join(directory_path, file_path_from_meta)
                        else:
                           base_metadata['original_file_path'] = file_path_from_meta
                    else:
                        base_metadata['original_file_path'] = "unknown_path"


                    if custom_source:
                        base_metadata['source'] = custom_source
                    elif 'source' not in base_metadata and file_path_from_meta:
                         base_metadata['source'] = file_path_from_meta # Use path from loader if no custom source
                    elif 'source' not in base_metadata:
                        base_metadata['source'] = "unknown_source"


                    base_metadata['source_directory'] = directory_path
                    try:
                        fpath_for_mtime = base_metadata['original_file_path']
                        if os.path.exists(fpath_for_mtime) and os.path.isfile(fpath_for_mtime):
                            base_metadata['last_modified'] = os.path.getmtime(fpath_for_mtime)
                        else:
                            base_metadata['last_modified'] = time.time() # Fallback
                    except Exception:
                        base_metadata['last_modified'] = time.time()
                    
                    doc.metadata = base_metadata # Update doc's metadata before splitting
                    docs_for_recursive_split_others.append(doc)

                final_splits_for_others = split_documents_intelligently(docs_for_recursive_split_others, "general")
                
                for i, chunk in enumerate(final_splits_for_others):
                    if not chunk.metadata: chunk.metadata = {}
                    chunk.metadata['chunk_id'] = f"{chunk.metadata.get('source', 'unknown_other_file')}_other_part_{i}"
                
                all_final_split_documents.extend(final_splits_for_others)
                logger.info(f"Processed {len(final_splits_for_others)} non-Markdown chunks from {directory_path}")
        except Exception as e:
            logger.warning(f"Warning: Could not process non-Markdown files in {directory_path}: {e}")
            
    if not all_final_split_documents:
        logger.warning(f"No documents could be loaded or processed from any of the specified directories.")
    else:
        logger.info(f"Collected a total of {len(all_final_split_documents)} chunks from general directories.")
            
    return all_final_split_documents

# Function to load Obsidian documents from multiple directories
def collect_obsidian_documents(directories, custom_source=None):
    documents = []
    
    # Process each directory individually
    for directory_path in directories:
        if not os.path.exists(directory_path):
            logger.warning(f"Warning: Directory '{directory_path}' does not exist. Skipping.")
            continue
            
        try:
            loader = ObsidianLoader(directory_path)
            docs = loader.load()  # Load documents from the folder
            
            # Add source info to documents
            for doc in docs:
                if not doc.metadata:
                    doc.metadata = {}
                
                # ObsidianLoader provides 'path' (absolute) and 'source' (filename.md)
                # We prefer the absolute path for 'original_file_path' for consistency
                doc.metadata['original_file_path'] = doc.metadata.get('path', 'unknown_obsidian_path')

                if custom_source:
                    doc.metadata['source'] = custom_source
                elif 'source' not in doc.metadata: # Should be set by ObsidianLoader
                    doc.metadata['source'] = os.path.basename(doc.metadata.get('path', 'unknown.md'))
                
                doc.metadata['source_directory'] = directory_path
                # last_modified should be provided by ObsidianLoader, if not, fallback
                if 'last_modified' not in doc.metadata:
                    try:
                        doc.metadata['last_modified'] = os.path.getmtime(doc.metadata['original_file_path'])
                    except Exception:
                         doc.metadata['last_modified'] = time.time()
            
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {directory_path}")
        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {e}")
            
    return documents

# Function to create message pairs from chat history
def create_message_pairs(messages):
    """
    Group messages into user-assistant pairs for better context.
    Returns list of paired messages with combined content, preserving markdown formatting.
    """
    pairs = []
    current_user_msg = None
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")  # Don't strip here to preserve markdown formatting
        
        # Skip truly empty messages (but allow whitespace-only content that might be significant)
        if not content or content.isspace():
            continue
            
        if role == "user":
            # Save user message, waiting for assistant response
            current_user_msg = msg
        elif role == "assistant" and current_user_msg is not None:
            # Get content preserving markdown formatting
            user_content = current_user_msg.get('content', '')
            assistant_content = content
            
            # Create pair with markdown-preserved formatting using proper markdown structure
            pair_content = f"""## User

{user_content}

## Assistant

{assistant_content}"""
            
            # Create combined metadata
            pair_metadata = {
                "user_id": current_user_msg.get("id", ""),
                "assistant_id": msg.get("id", ""),
                "user_timestamp": current_user_msg.get("timestamp", ""),
                "assistant_timestamp": msg.get("timestamp", ""),
                "modelName": msg.get("modelName") or msg.get("model", ""),
                "conversation_pair": True,
                "pair_id": f"{current_user_msg.get('id', 'unknown')}-{msg.get('id', 'unknown')}",
                "format": "markdown"  # Indicate that this content contains markdown
            }
            
            if args.source:
                pair_metadata["source"] = args.source
            else:
                pair_metadata["source"] = f"chat_pair_{pair_metadata['pair_id']}"
            
            pairs.append({
                "content": pair_content,
                "metadata": pair_metadata
            })
            
            # Reset for next pair
            current_user_msg = None
    
    return pairs

# Function to load chat documents with individual message processing (legacy)
def collect_chat_documents_individual(json_file, batch_size_for_chunking=1000):
    if not os.path.exists(json_file):
        logger.warning(f"Warning: JSON file '{json_file}' does not exist. No documents will be loaded.")
        return iter([])
    
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
        
        doc_iterator = loader.lazy_load()
        
        batch_for_chunking = []
        raw_docs_processed_count = 0

        logger.info(f"Starting to stream and process individual messages from {json_file}...")

        for doc in doc_iterator:
            batch_for_chunking.append(doc)
            if len(batch_for_chunking) >= batch_size_for_chunking:
                current_raw_batch_size = len(batch_for_chunking)
                logger.info(f"Collected {current_raw_batch_size} raw messages. Preparing to chunk...")
                
                start_chunk_time = time.time()
                split_docs = split_documents_intelligently(batch_for_chunking, "chat")
                end_chunk_time = time.time()
                logger.info(f"Finished chunking {current_raw_batch_size} raw messages in {end_chunk_time - start_chunk_time:.2f} seconds. Found {len(split_docs)} initial chunks. Filtering...")
                
                start_filter_time = time.time()
                filtered_split_docs = filter_documents(split_docs)
                end_filter_time = time.time()
                num_yielded = len(filtered_split_docs) if filtered_split_docs else 0
                logger.info(f"Finished filtering in {end_filter_time - start_filter_time:.2f} seconds. Yielding {num_yielded} chunks.")

                if filtered_split_docs:
                    yield filtered_split_docs
                raw_docs_processed_count += current_raw_batch_size
                batch_for_chunking = []

            # Process any remaining documents in the last batch
        if batch_for_chunking:
            current_raw_batch_size = len(batch_for_chunking)
            logger.info(f"Collected final batch of {current_raw_batch_size} raw messages. Preparing to chunk...")
            
            start_chunk_time = time.time()
            split_docs = split_documents_intelligently(batch_for_chunking, "chat")
            end_chunk_time = time.time()
            logger.info(f"Finished chunking final {current_raw_batch_size} raw messages in {end_chunk_time - start_chunk_time:.2f} seconds. Found {len(split_docs)} initial chunks. Filtering...")
            
            start_filter_time = time.time()
            filtered_split_docs = filter_documents(split_docs)
            end_filter_time = time.time()
            num_yielded = len(filtered_split_docs) if filtered_split_docs else 0
            logger.info(f"Finished filtering final batch in {end_filter_time - start_filter_time:.2f} seconds. Yielding {num_yielded} chunks.")

            if filtered_split_docs:
                yield filtered_split_docs
            raw_docs_processed_count += current_raw_batch_size

        logger.info(f"Finished streaming and processing. Total raw chat messages processed from {json_file}: {raw_docs_processed_count}")

    except Exception as e:
        logger.error(f"Error during streaming chat documents from {json_file}: {e}")
        return iter([])

# Function to load chat documents with message pairing (modified for streaming)
def collect_chat_documents(json_file, batch_size_for_chunking=1000):
    # Choose processing method based on user preference
    if args.pair_messages:
        return collect_chat_documents_paired(json_file, batch_size_for_chunking)
    else:
        return collect_chat_documents_individual(json_file, batch_size_for_chunking)

# Function to load chat documents with message pairing
def collect_chat_documents_paired(json_file, batch_size_for_chunking=1000):
    if not os.path.exists(json_file):
        logger.warning(f"Warning: JSON file '{json_file}' does not exist. No documents will be loaded.")
        return iter([])
    
    try:
        import json
        
        logger.info(f"Loading and pairing chat messages from {json_file}...")
        
        # Load the entire JSON file - it's an array of chat records
        with open(json_file, 'r') as f:
            data = json.load(f)  # Load as single JSON array
        
        # Ensure data is a list
        if not isinstance(data, list):
            logger.error(f"Expected JSON array, got {type(data)}")
            return iter([])
        
        batch_for_chunking = []
        raw_pairs_processed_count = 0
        total_individual_messages = 0
        
        logger.info(f"Processing {len(data)} chat records...")
        
        # Process each chat record to extract and pair messages
        for chat_record in data:
            messages = []
            
            # Extract messages from the correct structure
            try:
                if 'chat' in chat_record and 'history' in chat_record['chat'] and 'messages' in chat_record['chat']['history']:
                    messages_dict = chat_record['chat']['history']['messages']
                    # Convert dictionary to list of messages
                    messages = list(messages_dict.values())
                elif 'chat' in chat_record and 'messages' in chat_record['chat']:
                    # Fallback for different structure
                    messages = chat_record['chat']['messages']
                else:
                    logger.warning(f"Skipping chat record with unexpected structure")
                    continue
                    
                total_individual_messages += len(messages)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping chat record with unexpected structure: {e}")
                continue
            
            if not messages:
                continue
                
            # Create message pairs from this conversation
            message_pairs = create_message_pairs(messages)
            
            # Convert pairs to Document-like objects
            for pair in message_pairs:
                # Create a document-like object with the paired content
                from langchain_core.documents import Document
                
                doc = Document(
                    page_content=pair["content"],
                    metadata=pair["metadata"]
                )
                
                batch_for_chunking.append(doc)
                
                # Process in batches
                if len(batch_for_chunking) >= batch_size_for_chunking:
                    current_batch_size = len(batch_for_chunking)
                    logger.info(f"Collected {current_batch_size} message pairs. Preparing to chunk...")
                    
                    start_chunk_time = time.time()
                    split_docs = split_documents_intelligently(batch_for_chunking, "chat")
                    end_chunk_time = time.time()
                    logger.info(f"Finished chunking {current_batch_size} message pairs in {end_chunk_time - start_chunk_time:.2f} seconds. Found {len(split_docs)} initial chunks. Filtering...")
                    
                    start_filter_time = time.time()
                    filtered_split_docs = filter_documents(split_docs)
                    end_filter_time = time.time()
                    num_yielded = len(filtered_split_docs) if filtered_split_docs else 0
                    logger.info(f"Finished filtering in {end_filter_time - start_filter_time:.2f} seconds. Yielding {num_yielded} chunks.")

                    if filtered_split_docs:
                        yield filtered_split_docs
                    raw_pairs_processed_count += current_batch_size
                    batch_for_chunking = []
        
        # Process any remaining documents in the last batch
        if batch_for_chunking:
            current_batch_size = len(batch_for_chunking)
            logger.info(f"Collected final batch of {current_batch_size} message pairs. Preparing to chunk...")
            
            start_chunk_time = time.time()
            split_docs = split_documents_intelligently(batch_for_chunking, "chat")
            end_chunk_time = time.time()
            logger.info(f"Finished chunking final {current_batch_size} message pairs in {end_chunk_time - start_chunk_time:.2f} seconds. Found {len(split_docs)} initial chunks. Filtering...")
            
            start_filter_time = time.time()
            filtered_split_docs = filter_documents(split_docs)
            end_filter_time = time.time()
            num_yielded = len(filtered_split_docs) if filtered_split_docs else 0
            logger.info(f"Finished filtering final batch in {end_filter_time - start_filter_time:.2f} seconds. Yielding {num_yielded} chunks.")

            if filtered_split_docs:
                yield filtered_split_docs
            raw_pairs_processed_count += current_batch_size
        
        logger.info(f"Finished processing chat. Total individual messages: {total_individual_messages}, Total message pairs created: {raw_pairs_processed_count}")

    except Exception as e:
        logger.error(f"Error during processing chat documents from {json_file}: {e}")
        return iter([])

# Async function for high-performance chat document processing
async def process_chat_documents_async(json_file: str, vector_store, batch_size_for_chunking: int = 1000):
    """
    High-performance async processing of chat documents with concurrent batch processing.
    """
    if not os.path.exists(json_file):
        logger.warning(f"Warning: JSON file '{json_file}' does not exist.")
        return 0
    
    try:
        logger.info(f"Starting async chat processing from: {json_file}")
        start_total_time = time.time()
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error(f"Expected JSON array, got {type(data)}")
            return 0
        
        logger.info(f"Loaded {len(data)} chat records for async processing")
        
        # Process chat records and create document batches
        all_batches = []
        current_batch = []
        total_individual_messages = 0
        
        for chat_record in data:
            try:
                messages = []
                if 'chat' in chat_record and 'history' in chat_record['chat'] and 'messages' in chat_record['chat']['history']:
                    messages_dict = chat_record['chat']['history']['messages']
                    messages = list(messages_dict.values())
                elif 'chat' in chat_record and 'messages' in chat_record['chat']:
                    messages = chat_record['chat']['messages']
                else:
                    continue
                    
                total_individual_messages += len(messages)
                
                if not messages:
                    continue
                
                # Create message pairs or individual messages based on settings
                if args.pair_messages:
                    message_pairs = create_message_pairs(messages)
                    for pair in message_pairs:
                        from langchain_core.documents import Document
                        doc = Document(page_content=pair["content"], metadata=pair["metadata"])
                        current_batch.append(doc)
                        
                        if len(current_batch) >= batch_size_for_chunking:
                            all_batches.append(current_batch)
                            current_batch = []
                else:
                    # Individual message processing (implement if needed)
                    pass
                    
            except Exception as e:
                logger.warning(f"Skipping chat record: {e}")
                continue
        
        # Add final batch
        if current_batch:
            all_batches.append(current_batch)
        
        logger.info(f"Created {len(all_batches)} batches from {total_individual_messages} individual messages")
        
        if not all_batches:
            logger.warning("No chat batches to process")
            return 0
        
        # Process batches concurrently with controlled concurrency
        total_uploaded = 0
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
        
        async def process_and_upload_batch(batch_docs, batch_id):
            async with semaphore:
                # Process documents
                processed_docs = await process_document_batch_async(batch_docs, "chat", batch_id)
                if processed_docs:
                    # Upload documents
                    success = await upload_documents_async(vector_store, processed_docs, batch_id)
                    return len(processed_docs) if success else 0
                return 0
        
        # Create tasks for all batches
        tasks = [
            process_and_upload_batch(batch, i + 1) 
            for i, batch in enumerate(all_batches)
        ]
        
        # Execute all tasks concurrently
        logger.info(f"Starting concurrent processing of {len(tasks)} batches (max {MAX_CONCURRENT_BATCHES} concurrent)")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful uploads
        for result in results:
            if isinstance(result, int):
                total_uploaded += result
            elif isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
        
        end_total_time = time.time()
        processing_time = end_total_time - start_total_time
        
        logger.info(f"\n🚀 Async chat processing complete:")
        logger.info(f"  - Total batches processed: {len(all_batches)}")
        logger.info(f"  - Total chunks uploaded: {total_uploaded}")
        logger.info(f"  - Total processing time: {processing_time:.2f}s")
        logger.info(f"  - Average time per batch: {processing_time/len(all_batches):.2f}s")
        logger.info(f"  - Processing rate: {total_uploaded/processing_time:.1f} chunks/second")
        
        return total_uploaded
        
    except Exception as e:
        logger.error(f"Error in async chat processing: {e}")
        return 0

# Function to filter out empty or very short documents
def filter_documents(docs, min_word_count=MIN_CONTENT_LENGTH):
    filtered_docs = []
    skipped = 0
    
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            skipped += 1
            continue
        word_count = len(content.split())
        if word_count < min_word_count:
            skipped += 1
            continue
        filtered_docs.append(doc)
    
    if skipped > 0:
        logger.info(f"Skipped {skipped} documents with insufficient content (less than {min_word_count} words)")
    
    return filtered_docs

# Async function to process a batch of documents
async def process_document_batch_async(batch_docs: List, document_type: str = "chat", batch_id: int = 0) -> List:
    """
    Asynchronously process a batch of documents: split, filter, and prepare for upload.
    """
    try:
        start_time = time.time()
        
        # Run document splitting in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            split_docs = await loop.run_in_executor(
                executor, 
                split_documents_intelligently, 
                batch_docs, 
                document_type
            )
            
            # Filter documents in parallel
            filtered_docs = await loop.run_in_executor(
                executor,
                filter_documents,
                split_docs
            )
        
        end_time = time.time()
        logger.info(f"Batch {batch_id}: Processed {len(batch_docs)} docs → {len(split_docs)} chunks → {len(filtered_docs)} filtered chunks in {end_time - start_time:.2f}s")
        
        return filtered_docs
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")
        return []

# Async function to upload documents to Qdrant
async def upload_documents_async(vector_store, docs: List, batch_id: int = 0) -> bool:
    """
    Asynchronously upload documents to Qdrant vector store.
    """
    try:
        start_time = time.time()
        
        # Run the upload in a thread pool since vector_store.add_documents is synchronous
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(
                executor,
                vector_store.add_documents,
                docs
            )
        
        end_time = time.time()
        logger.info(f"Batch {batch_id}: Successfully uploaded {len(docs)} chunks in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading batch {batch_id}: {e}")
        return False


# Initialize embeddings
try:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )
    logger.info(f"Initialized Ollama embeddings with model: {EMBEDDING_MODEL}")
    # Test the connection
    test_embedding = embeddings.embed_query("test")
    logger.info(f"Successfully connected to Ollama. Embedding dimension: {len(test_embedding)}")
    
    # Initialize semantic chunker based on configuration
    if USE_SEMANTIC_CHUNKER == "true":
        semantic_splitter = SemanticChunker(embeddings=embeddings)
        logger.info("Semantic chunker enabled for all document types")
    elif USE_SEMANTIC_CHUNKER == "auto":
        semantic_splitter = SemanticChunker(embeddings=embeddings)
        logger.info("Semantic chunker initialized - will auto-enable for chat documents")
    elif USE_SEMANTIC_CHUNKER == "false":
        semantic_splitter = None
        logger.info("Semantic chunker disabled for all document types")
    else:
        semantic_splitter = SemanticChunker(embeddings=embeddings)
        logger.info("Semantic chunker initialized - defaulting to auto mode")
        
except Exception as e:
    logger.error(f"Failed to initialize Ollama embeddings. Make sure Ollama is running and {EMBEDDING_MODEL} is available.")
    logger.error(f"Error: {e}")
    logger.error(f"You can install the model with: ollama pull {EMBEDDING_MODEL.split(':')[0]}")
    exit(1)

# Set up Qdrant client and collection
logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Check Qdrant connection and server status
try:
    # Try a basic operation like listing collections to verify connection
    client.get_collections() # This will raise an exception if the server is not reachable
    logger.info(f"Successfully connected to Qdrant server at {QDRANT_URL} and was able to list collections.")
except Exception as conn_e:
    logger.error(f"Could not connect to or communicate effectively with Qdrant server at {QDRANT_URL}.")
    logger.error(f"Details: {conn_e}")
    logger.error("Please ensure the Qdrant server is running, accessible, and properly configured.")
    exit(1)

collection_name = args.collection
try:
    logger.info(f"Checking for existing collection: '{collection_name}'...")
    client.get_collection(collection_name)
    logger.info(f"Collection '{collection_name}' already exists. Adding documents to it.")
except UnexpectedResponse as e:
    if e.status_code == 404: # Not found
        logger.info(f"Collection '{collection_name}' not found. Creating new collection.")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_DIMENSIONS, distance=DISTANCE_MAP[DISTANCE_METRIC]),
            )
            logger.info(f"Successfully created collection '{collection_name}'.")
        except Exception as create_error:
            logger.error(f"Could not create collection '{collection_name}'.")
            logger.error(f"Details: {create_error}")
            exit(1)
    else:
        logger.error(f"An unexpected issue occurred while checking for collection '{collection_name}'.")
        logger.error(f"Details: {e}")
        exit(1)
except Exception as e: # Catch other potential errors like network issues during get_collection
    logger.error(f"Could not verify collection '{collection_name}' due to an unexpected error.")
    logger.error(f"Details: {e}")
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
    logger.error(f"Error: No directories specified for {args.type} document type. Use --dirs to specify directories.")
    exit(1)

if args.type == 'chat' and not args.json_file:
    logger.error("Error: --json-file is required for chat document type")
    exit(1)

# Load documents based on the specified type
if args.type == 'general':
    docs = collect_general_documents(directories_to_process, args.source)
    docs = filter_documents(docs) # Filter empty/short documents

    if not docs:
        logger.warning("No general documents to upload after filtering. Exiting.")
        exit(0)

    total_words = sum(len(doc.page_content.split()) for doc in docs)
    logger.info(f"{len(docs)} general documents loaded with a total of {total_words:,} words.")
    
    logger.info(f"Uploading {len(docs)} general documents to collection '{args.collection}'...")
    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i+BATCH_SIZE]
        vector_store.add_documents(batch_docs)
        logger.info(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(docs)-1)//BATCH_SIZE + 1} ({len(batch_docs)} documents)")
    
    logger.info(f"Successfully processed {len(docs)} general documents to Qdrant collection '{args.collection}' at {QDRANT_URL}.")

elif args.type == 'obsidian':
    logger.info(f"Processing Obsidian documents with update checks from: {directories_to_process}")
    
    raw_docs = collect_obsidian_documents(directories_to_process, args.source)
    # Note: Filtering for Obsidian docs is handled implicitly by checking content for updates/existence.
    # If needed, filter_documents can be applied to raw_docs before the docs_by_source grouping.

    if not raw_docs:
        logger.info("No Obsidian documents found to process. Exiting.")
        exit(0)

    docs_by_source = {}
    for doc in raw_docs:
        source = doc.metadata.get('source')
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    # Cull remote obsidian sources that are no longer present locally
    local_sources = set(docs_by_source.keys())
    logger.info(f"Culling remote obsidian sources not present locally. Local sources: {sorted(local_sources)}")
    remote_sources = set()
    offset = None
    limit = BATCH_SIZE
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            payload = getattr(rec, 'payload', None)
            if payload is None and isinstance(rec, dict):
                payload = rec.get('payload')
            if not payload:
                continue
            metadata = payload.get('metadata', {}) or {}
            src = metadata.get('source')
            if src:
                remote_sources.add(src)
        if offset is None:
            break
    to_delete = remote_sources - local_sources
    if to_delete:
        for src in to_delete:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="metadata.source", match=MatchValue(value=src))]
                )
            )
            logger.info(f"Deleted remote obsidian source no longer present: {src}")
        logger.info(f"Culled {len(to_delete)} remote source(s) from collection.")
    else:
        logger.info("No remote obsidian sources to cull.")
    
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
            logger.info(f"Skipping source '{source}' as all its content is too short after filtering.")
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
                logger.info(f"Skipping existing document source: {source}")
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
                    logger.info(f"Skipping unchanged document source: {source}")
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
                logger.info(f"Marked document source for update: {source}")
                total_updated_sources += 1
            else:
                logger.info(f"Marked new document source for adding: {source}")
                total_added_sources += 1
            total_processed_sources +=1
    
    if docs_to_chunk_and_upload:
        logger.info(f"Applying header and recursive splitting to {len(docs_to_chunk_and_upload)} Obsidian documents from {total_processed_sources} sources...")
        all_final_chunks_for_obsidian_upload = []
        for doc_to_process in docs_to_chunk_and_upload: # These are full documents for sources that need update/add
            original_metadata = doc_to_process.metadata.copy() # Already enriched by collect_obsidian_documents
            page_content = doc_to_process.page_content

            header_splits = markdown_header_splitter.split_text(page_content)
            
            current_file_chunks_for_recursive_split = []
            for h_split in header_splits:
                merged_meta = original_metadata.copy()
                merged_meta.update(h_split.metadata) # Add Header 1, Header 2 etc.
                h_split.metadata = merged_meta
                current_file_chunks_for_recursive_split.append(h_split)
                
            final_chunks_for_file = split_documents_intelligently(current_file_chunks_for_recursive_split, "obsidian")
            
            for i, chunk in enumerate(final_chunks_for_file):
                if not chunk.metadata: chunk.metadata = {} # Should be populated
                chunk.metadata['chunk_id'] = f"{chunk.metadata.get('source', 'unknown_obsidian_file')}_obs_part_{i}"

            all_final_chunks_for_obsidian_upload.extend(final_chunks_for_file)
        logger.info(f"Finished splitting. Produced {len(all_final_chunks_for_obsidian_upload)} chunks for upload.")
        
        split_docs_for_upload = all_final_chunks_for_obsidian_upload
        total_chunks_uploaded = len(split_docs_for_upload)

        logger.info(f"Uploading {total_chunks_uploaded} chunks from {total_processed_sources} sources...")
        for i in range(0, len(split_docs_for_upload), BATCH_SIZE):
            batch_docs = split_docs_for_upload[i:i+BATCH_SIZE]
            vector_store.add_documents(batch_docs)
            logger.info(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(split_docs_for_upload)-1)//BATCH_SIZE + 1} ({len(batch_docs)} chunks)")
    else:
        logger.info("No Obsidian documents required chunking or uploading.")
    
    logger.info(f"\nObsidian document processing complete:")
    logger.info(f"  - New document sources added: {total_added_sources}")
    logger.info(f"  - Existing document sources updated: {total_updated_sources}")
    logger.info(f"  - Document sources skipped (unchanged or already exists): {total_skipped_sources}")
    logger.info(f"  - Total chunks uploaded: {total_chunks_uploaded}")
    logger.info(f"Successfully processed Obsidian documents to Qdrant collection '{args.collection}' at {QDRANT_URL}.")

elif args.type == 'chat':
    mode = "message pairing" if args.pair_messages else "individual messages"
    
    if ENABLE_ASYNC_CHAT:
        logger.info(f"🚀 Using high-performance async processing for chat documents from: {args.json_file}")
        logger.info(f"  - Mode: {mode}")
        logger.info(f"  - Max concurrent batches: {MAX_CONCURRENT_BATCHES}")
        logger.info(f"  - GPU-optimized batch size: {BATCH_SIZE}")
        
        # Use async processing for much faster performance
        async def run_async_chat_processing():
            return await process_chat_documents_async(args.json_file, vector_store, batch_size_for_chunking=1000)
        
        total_chunks_uploaded_chat = asyncio.run(run_async_chat_processing())
        
        if total_chunks_uploaded_chat == 0:
            logger.warning("No chat documents were processed. Exiting.")
            exit(0)
        
        logger.info(f"✅ High-performance async chat processing completed successfully!")
        logger.info(f"Successfully processed {total_chunks_uploaded_chat} chunks to Qdrant collection '{args.collection}' at {QDRANT_URL}.")
        
    else:
        # Fall back to legacy synchronous processing
        logger.info(f"Processing chat documents from: {args.json_file} (legacy synchronous mode with {mode})")
        logger.info("💡 Tip: Set QDRANT_UPLOAD_ASYNC_CHAT=true for much faster processing!")
        
        doc_batch_generator = collect_chat_documents(args.json_file, batch_size_for_chunking=1000)
        
        total_chunks_uploaded_chat = 0
        batch_num = 0
        any_docs_processed = False

        for doc_batch in doc_batch_generator:
            if not doc_batch:
                continue

            any_docs_processed = True
            batch_num += 1
            num_docs_in_batch = len(doc_batch)
            total_chunks_uploaded_chat += num_docs_in_batch
            
            logger.info(f"Uploading chat document batch {batch_num} with {num_docs_in_batch} chunks to collection '{args.collection}'...")
            vector_store.add_documents(doc_batch)
            logger.info(f"Uploaded chat batch {batch_num} ({num_docs_in_batch} chunks).")

        if not any_docs_processed:
            logger.warning("No chat documents were processed or yielded from the generator. Exiting.")
            exit(0)

        logger.info(f"\nChat document streaming and upload complete:")
        logger.info(f"  - Processing mode: {mode}")
        logger.info(f"  - Total chunks uploaded: {total_chunks_uploaded_chat}")
        logger.info(f"Successfully processed chat documents to Qdrant collection '{args.collection}' at {QDRANT_URL}.")

else:
    logger.error(f"Error: Unknown document type specified: {args.type}")
    exit(1)

# General success message parts, can be enhanced per type if needed
if args.source:
    logger.info(f"All documents processed were intended to be tagged with custom source: '{args.source}'") 