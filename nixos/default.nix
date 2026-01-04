{ lib
, stdenv
, python3
, self ? null
, version ? "0.1.0"
}:

let
  # Use python312 for langchain-experimental
  python312 = stdenv.mkDerivation {
    name = "python312-wrapper";
    buildInputs = [ python3 ];
    python312Packages = import <nixpkgs> { inherit stdenv; }.python312Packages;
  };

  # Use langchain-experimental from python312Packages
  langchain-experimental = (import <nixpkgs> { inherit stdenv; }).python312Packages.langchain-experimental;

  # Build langchain-qdrant
  langchain-qdrant = python3.pkgs.buildPythonPackage rec {
    pname = "langchain_qdrant";
    version = "1.1.0"; # Use latest version or adjust as needed
    format = "pyproject";
    src = python3.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "QzI2hFc2+XNUOsa3E3+df6URt1OTc/vGVlxYHTulk3M=";
    };
    nativeBuildInputs = [ python3.pkgs.hatchling ];
    propagatedBuildInputs = with python3.pkgs; [
      langchain
      qdrant-client
    ];
  };

  # Create a Python environment with necessary packages
  pythonWithPackages = python3.withPackages (ps: with ps; [
    python-dotenv
    qdrant-client
    langchain
    langchain-community
    langchain-text-splitters
    unstructured
    # Add our custom packages
    langchain-experimental
    langchain-qdrant
  ]);

  # Define default options
  defaultOptions = {
    qdrantUrl = "http://localhost:6333";
    embeddingModel = "nomic-embed-text:latest";
    vectorDimensions = 768;
    distanceMetric = "Cosine";
    batchSize = 2000; # Optimized for RTX 3070 GPU
    minContentLength = 50;
    # Ollama settings
    ollamaUrl = "http://localhost:11434";
    ollamaTimeout = 300;
    # New async performance settings
    chunkSize = 2500; # Larger chunks for better chat message completeness
    chunkOverlap = 200; # Proportional overlap
    semanticChunker = "false"; # auto=chat only, true=all, false=none (default: false for performance)
    maxConcurrent = 4; # Optimal for RTX 3070 (8GB VRAM)
    asyncChat = true; # Enable high-performance async processing
  };

in
stdenv.mkDerivation {
  pname = "qdrant-upload";
  inherit version;

  src = if self != null then self else ./.;

  buildInputs = [
    pythonWithPackages
  ];

  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/share/qdrant-upload

    # Copy main Python script
    cp ${../upload.py} $out/share/qdrant-upload/upload.py

    # Create wrapper script
    cat > $out/bin/qdrant-upload << EOF
    #!/bin/sh
    # Export default environment variables
    export QDRANT_UPLOAD_URL="${defaultOptions.qdrantUrl}"
    export QDRANT_UPLOAD_MODEL="${defaultOptions.embeddingModel}"
    export QDRANT_UPLOAD_DIMENSIONS="${toString defaultOptions.vectorDimensions}"
    export QDRANT_UPLOAD_DISTANCE="${defaultOptions.distanceMetric}"
    export QDRANT_UPLOAD_BATCH_SIZE="${toString defaultOptions.batchSize}"
    export QDRANT_UPLOAD_MIN_LENGTH="${toString defaultOptions.minContentLength}"
    
    # Ollama settings
    export OLLAMA_URL="${defaultOptions.ollamaUrl}"
    export OLLAMA_TIMEOUT="${toString defaultOptions.ollamaTimeout}"
    
    # Async performance variables (RTX 3070 optimized)
    export QDRANT_UPLOAD_CHUNK_SIZE="${toString defaultOptions.chunkSize}"
    export QDRANT_UPLOAD_CHUNK_OVERLAP="${toString defaultOptions.chunkOverlap}"
    export QDRANT_UPLOAD_SEMANTIC_CHUNKER="${defaultOptions.semanticChunker}"
    export QDRANT_UPLOAD_MAX_CONCURRENT="${toString defaultOptions.maxConcurrent}"
    export QDRANT_UPLOAD_ASYNC_CHAT="${if defaultOptions.asyncChat then "true" else "false"}"

    if [ -z "\$1" ]; then
      echo "Usage: qdrant-upload <type> [options]"
      echo ""
      echo "Types:"
      echo "  obsidian - Process Obsidian documents"
      echo "  general  - Process general documents"
      echo "  chat     - Process chat history from JSON file"
      echo ""
      echo "Example: qdrant-upload obsidian --collection my_docs"
      exit 1
    fi

    TYPE="\$1"
    shift

    # Load .env file if present for additional configuration
    if [ -f .env ]; then
      set -a
      source .env
      set +a
    fi
    
    # Run the upload script with the specified type and directories
    exec ${pythonWithPackages}/bin/python3 $out/share/qdrant-upload/upload.py --type "\$TYPE" "\$@"
    EOF
    
    chmod +x $out/bin/qdrant-upload
  '';

  meta = with lib; {
    description = "A tool for uploading and indexing documents to a Qdrant vector database with Ollama embeddings";
    homepage = "https://github.com/Cody-W-Tucker/qdrant-upload";
    license = licenses.mit;
    maintainers = [ "github.com/Cody-W-Tucker" ];
    platforms = platforms.all;
  };
}
