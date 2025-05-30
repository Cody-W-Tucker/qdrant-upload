{ lib
, stdenv
, python3
, self ? null
, version ? "0.1.0"
}:

let
  # Build langchain-experimental compatible with nixpkgs versions
  langchain-experimental = python3.pkgs.buildPythonPackage rec {
    pname = "langchain_experimental";
    version = "0.3.1"; # Using an older version compatible with langchain 0.3.20
    format = "pyproject";
    src = python3.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "b4moidS+K/ZQdr2NYbPTbpSFXSDxjqXAZk4AcA8q/Vg=";
    };
    nativeBuildInputs = [ python3.pkgs.poetry-core ];
    propagatedBuildInputs = with python3.pkgs; [ langchain langchain-community ];
  };

  # Build langchain-qdrant
  langchain-qdrant = python3.pkgs.buildPythonPackage rec {
    pname = "langchain_qdrant";
    version = "0.2.0"; # Use latest version or adjust as needed
    format = "pyproject";
    src = python3.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "QbhXPLsbRwb3bcdpJR2Oaz5BB+zV+pfFgUGXfsGfunU=";
    };
    nativeBuildInputs = [ python3.pkgs.poetry-core ];
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
    langchain-openai
    langchain-text-splitters
    unstructured
    # Add our custom packages
    langchain-experimental
    langchain-qdrant
  ]);

  # Define default options
  defaultOptions = {
    qdrantUrl = "http://localhost:6333";
    defaultCollection = "personal";
    embeddingModel = "text-embedding-3-large";
    vectorDimensions = 3072;
    distanceMetric = "Cosine";
    batchSize = 100;
    minContentLength = 10;
    obsidianDirectories = [ ];
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
    cp ${./upload.py} $out/share/qdrant-upload/upload.py

    # Create wrapper script
    cat > $out/bin/qdrant-upload << EOF
    #!/bin/sh
    # Export default environment variables
    export QDRANT_UPLOAD_URL="${defaultOptions.qdrantUrl}"
    export QDRANT_UPLOAD_COLLECTION="${defaultOptions.defaultCollection}"
    export QDRANT_UPLOAD_MODEL="${defaultOptions.embeddingModel}"
    export QDRANT_UPLOAD_DIMENSIONS="${toString defaultOptions.vectorDimensions}"
    export QDRANT_UPLOAD_DISTANCE="${defaultOptions.distanceMetric}"
    export QDRANT_UPLOAD_BATCH_SIZE="${toString defaultOptions.batchSize}"
    export QDRANT_UPLOAD_MIN_LENGTH="${toString defaultOptions.minContentLength}"

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

    # Add OPENAI_API_KEY to environment if not present
    if [ -z "\$OPENAI_API_KEY" ] && [ -f .env ]; then
      set -a
      source .env
      set +a
    fi

    if [ -z "\$OPENAI_API_KEY" ]; then
      echo "Error: OPENAI_API_KEY environment variable is not set"
      echo "Please set it in your environment or create a .env file"
      exit 1
    fi
    
    # Run the upload script with the specified type and directories
    exec ${pythonWithPackages}/bin/python3 $out/share/qdrant-upload/upload.py --type "\$TYPE" "\$@"
    EOF
    
    chmod +x $out/bin/qdrant-upload
  '';

  meta = with lib; {
    description = "A tool for uploading and indexing documents to a Qdrant vector database with OpenAI embeddings";
    homepage = "https://github.com/Cody-W-Tucker/qdrant-upload";
    license = licenses.mit;
    maintainers = [ "github.com/Cody-W-Tucker" ];
    platforms = platforms.all;
  };
}
