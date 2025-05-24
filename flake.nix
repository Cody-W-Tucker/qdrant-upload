{
  description = "A Nix-flake-based Python document uploader for Qdrant";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });

      # Define common Python packages and options
      makeConfig = pkgs: with pkgs.lib; let
        # Define default options (these can be overridden by package options)
        defaultOptions = {
          qdrantUrl = "http://localhost:6333";
          defaultCollection = "main";
          embeddingModel = "text-embedding-3-large";
          vectorDimensions = 3072;
          distanceMetric = "Cosine";
          batchSize = 100;
          minContentLength = 10;
          obsidianDirectories = [];
        };

        # Build langchain-experimental compatible with nixpkgs versions
        langchain-experimental = pkgs.python312Packages.buildPythonPackage rec {
          pname = "langchain_experimental";
          version = "0.3.1"; # Using an older version compatible with langchain 0.3.20
          format = "pyproject";
          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "b4moidS+K/ZQdr2NYbPTbpSFXSDxjqXAZk4AcA8q/Vg=";
          };
          nativeBuildInputs = [ pkgs.python312Packages.poetry-core ];
          propagatedBuildInputs = with pkgs.python312Packages; [ langchain langchain-community ];
        };

        # Build langchain-qdrant
        langchain-qdrant = pkgs.python312Packages.buildPythonPackage rec {
          pname = "langchain_qdrant";
          version = "0.2.0"; # Use latest version or adjust as needed
          format = "pyproject";
          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "QbhXPLsbRwb3bcdpJR2Oaz5BB+zV+pfFgUGXfsGfunU=";
          };
          nativeBuildInputs = [ pkgs.python312Packages.poetry-core ];
          propagatedBuildInputs = with pkgs.python312Packages; [ 
            langchain
            qdrant-client
          ];
        };

        # Function to create a Python environment with specific package versions and config
        mkPythonEnv = { 
          qdrantUrl ? defaultOptions.qdrantUrl,
          defaultCollection ? defaultOptions.defaultCollection,
          embeddingModel ? defaultOptions.embeddingModel, 
          vectorDimensions ? defaultOptions.vectorDimensions,
          distanceMetric ? defaultOptions.distanceMetric,
          batchSize ? defaultOptions.batchSize,
          minContentLength ? defaultOptions.minContentLength,
          obsidianDirectories ? defaultOptions.obsidianDirectories
        }: (pkgs.python312.withPackages (ps: with ps; [
          python-dotenv
          qdrant-client
          # Use packages from nixpkgs
          langchain
          langchain-community
          langchain-openai
          langchain-text-splitters
          unstructured
          jq
          # Our custom packages
          langchain-experimental
          langchain-qdrant
        ])).overrideAttrs (old: {
          passthru = { inherit qdrantUrl defaultCollection embeddingModel vectorDimensions 
                       distanceMetric batchSize minContentLength obsidianDirectories; };
        });
      in {
        inherit defaultOptions langchain-experimental langchain-qdrant mkPythonEnv;
      };

      # Helper function to create a shell with default folders
      mkDevShell = { pkgs, config ? {} }: 
        let
          configUtils = makeConfig pkgs;
          options = pkgs.lib.recursiveUpdate configUtils.defaultOptions config;
          pythonEnv = configUtils.mkPythonEnv options;
          dirs = builtins.concatStringsSep " " (map (dir: ''"${dir}"'') options.obsidianDirectories);
        in pkgs.mkShell {
          packages = [ pythonEnv pkgs.jq ];
          shellHook = ''
            # Load .env file if it exists
            if [ -f .env ]; then
              echo "Loading environment from .env file..."
              set -a
              source .env
              set +a
            fi
            
            # Export configuration environment variables with fallbacks
            export QDRANT_UPLOAD_URL="''${QDRANT_UPLOAD_URL:-${options.qdrantUrl}}"
            export QDRANT_UPLOAD_COLLECTION="''${QDRANT_UPLOAD_COLLECTION:-${options.defaultCollection}}"
            export QDRANT_UPLOAD_MODEL="''${QDRANT_UPLOAD_MODEL:-${options.embeddingModel}}"
            export QDRANT_UPLOAD_DIMENSIONS="''${QDRANT_UPLOAD_DIMENSIONS:-${toString options.vectorDimensions}}"
            export QDRANT_UPLOAD_DISTANCE="''${QDRANT_UPLOAD_DISTANCE:-${options.distanceMetric}}"
            export QDRANT_UPLOAD_BATCH_SIZE="''${QDRANT_UPLOAD_BATCH_SIZE:-${toString options.batchSize}}"
            export QDRANT_UPLOAD_MIN_LENGTH="''${QDRANT_UPLOAD_MIN_LENGTH:-${toString options.minContentLength}}"
            
            # Use QDRANT_FOLDERS if set, otherwise fall back to obsidianDirectories
            if [ -n "$QDRANT_FOLDERS" ]; then
              IFS=' ' read -r -a DIRS_TO_PROCESS <<< "$QDRANT_FOLDERS"
            else
              DIRS_TO_PROCESS=(${dirs})
              export QDRANT_FOLDERS="''${QDRANT_FOLDERS:-${dirs}}"
            fi

            # If no folders are specified, warn the user
            if [ ''${#DIRS_TO_PROCESS[@]} -eq 0 ]; then
              echo "Warning: No directories specified for processing. Set QDRANT_FOLDERS or use obsidianDirectories."
            else
              echo "Ready to process the following directories:"
              for dir in "''${DIRS_TO_PROCESS[@]}"; do
                echo "  - $dir"
              done
              echo ""
              echo "Example command to process Obsidian documents:"
              echo "python upload.py --type obsidian --dirs ''${DIRS_TO_PROCESS[@]} --collection ${options.defaultCollection}"
            fi

            # Show Python environment information
            echo ""
            echo "Python environment information:"
            echo "Python interpreter: $(which python)"
            echo "Python version: $(python --version)"
            echo "Packages available directly from Nix (no .venv needed)"
            python -c "import langchain, langchain_experimental; print(f'  - langchain: {langchain.__version__}'); print(f'  - langchain_experimental: {langchain_experimental.__version__}')"
          '';
        };

      # Create a wrapped script for uploading documents
      mkUploader = { pkgs, config ? {} }: 
        let
          configUtils = makeConfig pkgs;
          options = pkgs.lib.recursiveUpdate configUtils.defaultOptions config;
          pythonEnv = configUtils.mkPythonEnv options;
          dirs = builtins.concatStringsSep " " (map (dir: ''"${dir}"'') options.obsidianDirectories);
          dirsArg = if builtins.length options.obsidianDirectories > 0 then "--dirs ${dirs}" else "";
        in
        pkgs.writeShellApplication {
          name = "qdrant-upload";
          runtimeInputs = [ pythonEnv pkgs.jq ];
          text = ''
            # Load .env file if it exists
            if [ -f .env ]; then
              echo "Loading environment from .env file..."
              # Disable shellcheck warning for the .env source
              # shellcheck disable=SC1091
              set -a
              source .env
              set +a
            fi

            # Export configuration environment variables with fallbacks
            export QDRANT_UPLOAD_URL="''${QDRANT_UPLOAD_URL:-${options.qdrantUrl}}"
            export QDRANT_UPLOAD_COLLECTION="''${QDRANT_UPLOAD_COLLECTION:-${options.defaultCollection}}"
            export QDRANT_UPLOAD_MODEL="''${QDRANT_UPLOAD_MODEL:-${options.embeddingModel}}"
            export QDRANT_UPLOAD_DIMENSIONS="''${QDRANT_UPLOAD_DIMENSIONS:-${toString options.vectorDimensions}}"
            export QDRANT_UPLOAD_DISTANCE="''${QDRANT_UPLOAD_DISTANCE:-${options.distanceMetric}}"
            export QDRANT_UPLOAD_BATCH_SIZE="''${QDRANT_UPLOAD_BATCH_SIZE:-${toString options.batchSize}}"
            export QDRANT_UPLOAD_MIN_LENGTH="''${QDRANT_UPLOAD_MIN_LENGTH:-${toString options.minContentLength}}"
            
            if [ -z "$1" ]; then
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

            TYPE="$1"
            shift

            # Add OPENAI_API_KEY to environment if not present
            if [ -z "$OPENAI_API_KEY" ] && [ -f .env ]; then
              # Disable shellcheck warning for the .env source
              # shellcheck disable=SC1091
              set -a
              source .env
              set +a
            fi

            if [ -z "$OPENAI_API_KEY" ]; then
              echo "Error: OPENAI_API_KEY environment variable is not set"
              echo "Please set it in your environment or create a .env file"
              exit 1
            fi
            
            # Check if QDRANT_FOLDERS is set, use that for directories
            if [ -n "$QDRANT_FOLDERS" ]; then
              # Build array of dirs, properly quoted
              dirs_cmd_args=()
              for dir in $QDRANT_FOLDERS; do
                dirs_cmd_args+=("$dir")
              done
              
              # Convert array to arguments
              dirs_cmd="--dirs"
              for dir in "''${dirs_cmd_args[@]}"; do
                dirs_cmd="$dirs_cmd $dir"
              done
            else
              # Use default dirs
              dirs_cmd="${dirsArg}"
              # Also set QDRANT_FOLDERS for environment
              export QDRANT_FOLDERS="''${QDRANT_FOLDERS:-${dirs}}"
            fi

            # Run the upload script with the specified type and directories
            # shellcheck disable=SC2086
            python ${./upload.py} --type "$TYPE" $dirs_cmd "$@"
          '';
          # Allow warnings from shellcheck
          checkPhase = '''';
        };
    in
    {
      # Create a NixOS module
      nixosModules.default = import ./nixos/module.nix;
    
      # Create a standard package
      packages = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.callPackage ./nixos/default.nix {
          version = "0.1.0";
        };
        
        custom = mkUploader { 
          inherit pkgs;
          config = {}; 
        };
      });
      
      # Development shells
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = mkDevShell { 
          inherit pkgs; 
          config = { 
            obsidianDirectories = [ "$HOME/Documents/Personal/Inbox" "$HOME/Documents/Personal/Journal" "$HOME/Documents/Personal/Knowledge" "$HOME/Documents/Personal/Projects" "$HOME/Documents/Personal/Research" ]; 
          };
        };
        custom = mkDevShell { inherit pkgs; config = {}; }; # No defaults, rely on QDRANT_FOLDERS
      });
      
      # Define apps
      apps = forEachSupportedSystem ({ pkgs }: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.system}.default}/bin/qdrant-upload";
        };
      });
    };
}