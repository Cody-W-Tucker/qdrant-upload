# Qdrant Document Uploader

Upload and index documents to a Qdrant vector database with OpenAI embeddings using Nix for reproducible environments.

## Workflow

```mermaid
%%{init: { 'logLevel': 'debug', 'theme': 'neutral' } }%%
graph TD  
    A[Start] --> B{Nix Environment?}  
    B -- Yes --> C[Setup Nix Environment: nix develop OR nix run . --]  
    B -- No --> B_ALT[Ensure Dependencies Manually]  
    C --> D{OpenAI API Key?}  
    B_ALT --> D  
    D -- Yes --> E[User executes: qdrant-upload type with options]  
    D -- No --> D_ERR[Create .env file and add OPENAI_API_KEY]  
    D_ERR --> A  
    E --> F{Document Type?}  
    F -- general --> G1[Load General Docs from directories]  
    F -- obsidian --> G2[Load Obsidian Vault from directories]  
    F -- chat --> G3[Load Chat History from JSON file]  
    G1 --> H[Process Documents]  
    G2 --> H  
    G3 --> H  
    H --> I{Configure via?}  
    I -- Env Vars --> J[Read environment variables for Qdrant URL, Collection, etc.]  
    I -- CLI Options --> K[Parse CLI options for collection, source, skipping, etc.]  
    J --> L[Core Processing Logic]  
    K --> L  
    L --> M[Perform Intelligent Semantic Chunking]  
    M --> N{Change Tracking for Obsidian Docs?}  
    N -- Skip Existing? / Unchanged? --> N_SKIP[Skip Document]  
    N -- Update Needed --> O[Generate OpenAI Embeddings]  
    O --> P[Batch Process and Upload to Qdrant]  
    N_SKIP --> P_END[Log Skipped Document]  
    P --> Q[Log Success and Statistics]  
    Q --> Z[End]  
    P_END --> Z  
    
    subgraph "NixOS Deployment (Optional Service)"  
        S1[Configure NixOS Module in system configuration.nix] --> S2[Define settings: qdrant URL, default collection, directories, environment file, and schedule]  
        S2 --> S3[Systemd service runs qdrant-upload automatically]  
        S3 --> E  
    end  
    
    style B_ALT fill:#ffccff,stroke:#333,stroke-width:2px  
    style D_ERR fill:#ff9999,stroke:#333,stroke-width:2px  
```

## Quick Start

### Using Nix Flakes (Recommended)

```bash
# Development shell with dependencies
nix develop

# Or run directly
nix run . -- obsidian --collection my_knowledge --dirs ~/Documents/Notes
```

### Set Up OpenAI API Key

Create a `.env` file in your working directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Document Types

```bash
# 1. General documents
qdrant-upload general --dirs /path/to/docs --collection general_docs

# 2. Obsidian vault
qdrant-upload obsidian --dirs /path/to/vault --collection obsidian_docs

# 3. Chat history
qdrant-upload chat --json-file /path/to/chat_history.json --collection chat_docs
```

## Key Features

- **Intelligent chunking**: Semantic document splitting preserves context
- **Change tracking**: Avoids reprocessing unchanged documents
- **Batch processing**: Efficiently handles large document collections 
- **Custom embedding**: Uses OpenAI's text-embedding-3-large model

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_UPLOAD_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_UPLOAD_COLLECTION` | `main` | Collection name |
| `QDRANT_UPLOAD_MODEL` | `text-embedding-3-large` | Embedding model |
| `QDRANT_FOLDERS` | `[]` | Default directories to process |

### Command Line Options

```bash
qdrant-upload --help

# Common options
  --collection NAME     Collection name (default: "main")
  --source ID           Custom source identifier
  --skip-existing       Skip documents already in collection
  --force-update        Update all documents regardless of changes
```

## NixOS Module

Add to your NixOS configuration:

```nix
# Import using flake
imports = [(builtins.getFlake "github:Cody-W-Tucker/qdrant-upload").nixosModules.default];

# Configure the service
services.qdrant-upload = {
  enable = true;
  qdrantUrl = "http://localhost:6333";
  defaultCollection = "my_documents";
  obsidianDirectories = ["/home/user/Documents/Notes"];
  
  # API key for OpenAI
  environmentFile = "/path/to/secrets/qdrant-upload.env";
  
  # Optional: Scheduled updates
  enableService = true;
  serviceSchedule = "*-*-* 02:00:00";  # Run daily at 2 AM
};
```

## Advanced Usage

### Processing Multiple Directories

```bash
# Set multiple directories in environment
export QDRANT_FOLDERS="$HOME/Documents/Notes $HOME/Documents/Research"
qdrant-upload obsidian

# Or specify directly
qdrant-upload obsidian --dirs ~/Documents/Notes ~/Documents/Research
```

### Using Custom Nix Development Shell

```bash
# Shell with custom configuration
nix develop .#custom

# With custom arguments
nix develop --arg 'config.qdrantUrl = "http://localhost:6333"'
```

## Troubleshooting

- **Authentication errors**: Check your OpenAI API key in `.env`
- **Connection issues**: Verify Qdrant server is running at specified URL
- **Embedding failures**: Ensure documents have sufficient content length
- **Missing dependencies**: Use the Nix development shell for a complete environment

## Structure

This project includes:

- Custom LangChain extensions (langchain-experimental v0.3.1, langchain-qdrant v0.2.0)
- Development environments with all dependencies
- NixOS module for system integration
- Command-line wrappers for simplified usage 