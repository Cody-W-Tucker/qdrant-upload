# Example NixOS configuration for Qdrant Upload Service
# This demonstrates the multi-source configuration system

{
  imports = [
    ./module.nix # Import the Qdrant upload module
  ];

  services.qdrant-upload = {
    enable = true;

    # Global settings shared by all sources
    qdrantUrl = "http://localhost:6333";
    embeddingModel = "nomic-embed-text:latest";

    # Ollama settings
    ollamaUrl = "http://localhost:11434";
    ollamaTimeout = 300;

    # Performance settings (optimized for RTX 3070)
    batchSize = 2000;
    maxConcurrent = 4;
    asyncChat = true;

    # Default schedule for all sources (can be overridden per source)
    defaultSchedule = "*-*-* 02:00:00"; # Daily at 2 AM

    # Multiple sources configuration - each with its own collection
    sources = [
      # Personal Obsidian vault
      {
        name = "personal-obsidian";
        type = "obsidian";
        collection = "personal_notes";
        directories = [ "/home/user/Documents/PersonalVault" ];
      }

      # Work documents
      {
        name = "work-docs";
        type = "general";
        collection = "work_docs";
        directories = [ "/home/user/Documents/Work" ];
        schedule = "*-*-* 01:00:00"; # Custom schedule: Daily at 1 AM
      }

      # Chat conversations from Open-WebUI export
      {
        name = "chat-history";
        type = "chat";
        collection = "conversations";
        jsonFile = "/home/user/Downloads/open-webui-export.json";
        schedule = "*-*-* 04:00:00"; # Custom schedule: Daily at 4 AM
      }

      # E-book collection
      {
        name = "ebooks";
        type = "ebook";
        collection = "books";
        directories = [ "/home/user/Documents/Books" ];
      }
    ];
  };
}

# This configuration creates:
# - Multiple systemd services: qdrant-upload-personal-obsidian, qdrant-upload-work-docs, etc.
# - Multiple systemd timers with different schedules
# - Each source processes into its own Qdrant collection
#
# Management commands:
# - systemctl status qdrant-upload-personal-obsidian
# - systemctl list-timers | grep qdrant-upload
# - journalctl -u qdrant-upload-work-docs -f
#
# Manual execution:
# - sudo -u qdrant-upload qdrant-upload obsidian --collection personal_notes --dirs /home/user/Documents/PersonalVault
