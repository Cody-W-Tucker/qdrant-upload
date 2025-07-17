# Qdrant Document Uploader

Personal document indexing service for Qdrant using Ollama embeddings. Async processing, multiple sources, code-aware chunking.

## Key Features

- **Multi-Source**: Process different document locations into separate collections
- **Document Types**: Obsidian vaults, general docs, chat conversations  
- **High Performance**: Async processing optimized for RTX 3070 (3-5x faster)
- **Code-Aware**: Preserves code blocks during chunking
- **Change Tracking**: Skip unchanged files
- **NixOS Service**: SystemD timers, individual services per source
- **Local Embeddings**: Ollama nomic-embed-text (no API keys)

## Quick Setup

### 1. Add to NixOS Flake

```nix
{
  inputs = {
    qdrant-upload = {
      url = "github:Cody-W-Tucker/Qdrant-Upload";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ qdrant-upload, ... }: {
    nixosConfigurations.your-host = nixpkgs.lib.nixosSystem {
      modules = [
        qdrant-upload.nixosModules.default
        # ... other modules
      ];
    };
  };
}
```

### 2. Configure Sources

```nix
services.qdrant-upload = {
  enable = true;
  
  # Service URLs
  qdrantUrl = "http://localhost:6333";
  ollamaUrl = "http://localhost:11434";
  
  # Performance (RTX 3070 optimized)
  batchSize = 2000;
  maxConcurrent = 4;
  asyncChat = true;
  
  sources = [
    # Personal notes
    {
      name = "personal-obsidian";
      type = "obsidian";
      collection = "personal_notes";
      directories = [ "/home/user/Documents/PersonalVault" ];
    }
    
    # Work docs
    {
      name = "work-docs";
      type = "general";
      collection = "work_docs";
      directories = [ "/home/user/Documents/Work" ];
      schedule = "*-*-* 01:00:00";  # Custom schedule
    }
    
    # Chat history
    {
      name = "chat-history";
      type = "chat";
      collection = "conversations";
      jsonFile = "/home/user/Downloads/open-webui-export.json";
      schedule = "*-*-* 04:00:00";
    }
  ];
};
```

### 3. Deploy

```bash
# Deploy
sudo nixos-rebuild switch --flake .#your-host

# Check services
systemctl list-timers | grep qdrant-upload
systemctl status qdrant-upload-personal-obsidian

# Monitor
journalctl -u qdrant-upload-personal-obsidian -f

# Manual run
sudo -u qdrant-upload qdrant-upload obsidian --collection personal_notes --dirs /home/user/Documents/PersonalVault
```

## Prerequisites

### Qdrant Server
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Ollama + Embedding Model
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
ollama serve
```

## Configuration Options

All settings can be configured directly in NixOS (no .env files needed):

```nix
services.qdrant-upload = {
  # Connection settings
  qdrantUrl = "http://localhost:6333";
  ollamaUrl = "http://localhost:11434";
  ollamaTimeout = 300;
  
  # Model settings
  embeddingModel = "nomic-embed-text:latest";
  vectorDimensions = 768;
  distanceMetric = "Cosine";
  
  # Performance tuning
  batchSize = 2000;
  maxConcurrent = 4;
  asyncChat = true;
  chunkSize = 2500;
  chunkOverlap = 200;
  semanticChunker = "auto";
  minContentLength = 50;
  
  # Service settings
  defaultSchedule = "*-*-* 02:00:00";
  environmentFile = "/path/to/optional.env"; # Only for edge cases
};
```

## Source Configuration

Each source gets its own SystemD service and timer:

```nix
{
  name = "unique-name";                    # Service suffix: qdrant-upload-{name}
  type = "obsidian" | "general" | "chat"; # Document processor type
  collection = "collection_name";          # Qdrant collection
  
  # For obsidian/general:
  directories = [ "/path/to/docs" ];       # List of directories to process
  
  # For chat:
  jsonFile = "/path/to/export.json";      # Open-WebUI export file
  
  # Optional:
  customSource = "source-tag";            # Custom source identifier
  skipExisting = false;                    # Skip unchanged documents
  forceUpdate = false;                     # Force update all
  schedule = "*-*-* 02:00:00";            # Custom schedule (default: 02:00)
}
```

## Performance Tuning

Adjust for your GPU:

```nix
# For 4GB GPU:
batchSize = 1000;
maxConcurrent = 2;

# For 16GB+ GPU:
batchSize = 4000;  
maxConcurrent = 8;
```

## Document Processing

- **Obsidian**: `.md` files with metadata, wikilinks, tags, frontmatter
- **General**: PDF, TXT, MD with metadata extraction  
- **Chat**: Full conversations preserved, code blocks protected

## Development

```bash
git clone https://github.com/Cody-W-Tucker/Qdrant-Upload
cd Qdrant-Upload
nix develop
python upload.py --type obsidian --dirs ~/Documents/Notes --collection test
```

## Service Management

```bash
# Individual services
systemctl start qdrant-upload-personal-obsidian
systemctl stop qdrant-upload-work-docs
systemctl restart qdrant-upload-chat-history

# Timers
systemctl enable qdrant-upload-personal-obsidian.timer
systemctl disable qdrant-upload-work-docs.timer

# Logs
journalctl -u qdrant-upload-personal-obsidian --since "1 hour ago"
systemctl list-units | grep qdrant-upload
```

## Troubleshooting

**Service fails:**
```bash
systemctl status qdrant-upload-source-name
journalctl -u qdrant-upload-source-name -f
```

**Connection issues:**
```bash
# Test Qdrant (check your qdrantUrl setting)
curl http://localhost:6333/collections

# Test Ollama (check your ollamaUrl setting)
curl http://localhost:11434/api/tags
ollama list | grep nomic-embed-text
```

**Performance issues:**
- Reduce `batchSize` for memory constraints
- Decrease `maxConcurrent` for CPU/GPU limits
- Set `asyncChat = false` to disable async processing
- Use `skipExisting = true` for faster incremental updates

---

Personal notes system for indexing documents with local LLM embeddings. No API keys, runs entirely locally with GPU acceleration.
