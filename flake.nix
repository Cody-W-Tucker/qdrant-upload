{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          venvDir = ".venv";
          packages = with pkgs; [ python311 ] ++
            (with pkgs.python311Packages; [
              pip
              venvShellHook
              qdrant-client
              langchain
              langchain-openai
              langchain-community
              langchain-text-splitters
              pydantic
              pandas
              jupyter
              notebook
              scikit-learn
              matplotlib
              umap-learn
              langgraph
              nltk
            ]);
          shellHook = ''
            # Default OBSIDIAN_PATH
            if [ -z "$OBSIDIAN_PATH" ]; then
              export OBSIDIAN_PATH="$HOME/Documents/Personal"
            fi

            # Create data directory if it doesn't exist
            mkdir -p "$PWD/data/obsidian"

            # Define directories to link
            DIRS_TO_LINK=("Knowledge" "Goals" "Journal")

            # Create symlinks for specific directories
            for dir in "''${DIRS_TO_LINK[@]}"; do
              SOURCE_PATH="$OBSIDIAN_PATH/$dir"
              SYMLINK_PATH="$PWD/data/obsidian/$dir"
    
              if [ -d "$SOURCE_PATH" ]; then
                ln -sfn "$SOURCE_PATH" "$SYMLINK_PATH" \
                  && echo "Created symlink: $SYMLINK_PATH -> $SOURCE_PATH" \
                  || echo "Failed to create symlink: $SYMLINK_PATH -> $SOURCE_PATH" >&2
              else
                echo "Warning: Source directory does not exist: $SOURCE_PATH"
              fi
            done

            # Activate the virtual environment
            if [ -f .venv/bin/activate ]; then
              source .venv/bin/activate
            else
              echo "Warning: Virtual environment not found. Run 'python -m venv .venv' to create it."
            fi
          '';
        };
      });
    };
}
