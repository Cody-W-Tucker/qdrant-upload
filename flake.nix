{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });

      # Helper function to create a shell with default folders
      mkDevShell = { pkgs, defaultFolders ? [ ] }: pkgs.mkShell {
        venvDir = ".venv";
        packages = with pkgs; [ python312 ] ++
          (with pkgs.python312Packages; [
            pip
            venvShellHook
            qdrant-client
            langchain
            langchain-openai
            langchain-community
            langchain-text-splitters
          ]);
        shellHook = ''
          # Create data directory if it doesn't exist
          mkdir -p "$PWD/data/upload"

          # Clean up existing symlinks in data/upload/ before creating new ones
          find "$PWD/data/upload" -type l -exec rm -f {} + 2>/dev/null || echo "No old symlinks to remove"

          # Use QDRANT_FOLDERS if set, otherwise fall back to defaultFolders
          if [ -n "$QDRANT_FOLDERS" ]; then
            IFS=' ' read -r -a DIRS_TO_LINK <<< "$QDRANT_FOLDERS"
          else
            DIRS_TO_LINK=(${builtins.concatStringsSep " " defaultFolders})
          fi

          # If no folders are specified, warn the user
          if [ ''${#DIRS_TO_LINK[@]} -eq 0 ]; then
            echo "Warning: No folders specified for Qdrant upload. Set QDRANT_FOLDERS or use defaultFolders."
          fi

          # Create symlinks for specified directories
          for source_path in "''${DIRS_TO_LINK[@]}"; do
            # Ensure the path is absolute
            if [[ "$source_path" != /* ]]; then
              source_path="$PWD/$source_path"
            fi

            # Extract the base name of the folder for the symlink
            dir_name=$(basename "$source_path")
            symlink_path="$PWD/data/upload/$dir_name"

            if [ -d "$source_path" ]; then
              ln -sfn "$source_path" "$symlink_path" \
                && echo "Created symlink: $symlink_path -> $source_path" \
                || echo "Failed to create symlink: $symlink_path -> $source_path" >&2
            else
              echo "Warning: Source directory does not exist: $source_path"
            fi
          done

          # Activate the virtual environment
          if [ -f .venv/bin/activate ]; then
            source .venv/bin/activate
          else
            echo "Warning: Virtual environment not found. Run 'python -m venv .venv' to create it."
          fi

          # Optional: Hint for next steps
          echo "Folders symlinked to $PWD/data/upload/. Ready to process for Qdrant."
        '';
      };
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = mkDevShell { inherit pkgs; defaultFolders = [ "$HOME/Documents/Personal/Experiments" "$HOME/Documents/Personal/Goals" "$HOME/Documents/Personal/Inbox" "$HOME/Documents/Personal/Journal" "$HOME/Documents/Personal/Knowledge" "$HOME/Documents/Personal/Research" ]; };
        custom = mkDevShell { inherit pkgs; defaultFolders = [ ]; }; # No defaults, rely on QDRANT_FOLDERS run like: QDRANT_FOLDERS="/var/www ~/Projects/Code" nix develop .#custom
      });
    };
}