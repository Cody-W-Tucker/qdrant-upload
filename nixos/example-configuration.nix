# Example NixOS configuration for Qdrant Upload Service
# This demonstrates the new multi-source configuration system

{
  imports = [
    ./module.nix # Import the Qdrant upload module
  ];

  services.qdrant-upload = {
    enable = true;
    enableService = true;

    # Global settings shared by all sources
    qdrantUrl = "http://localhost:6333";
    embeddingModel = "nomic-embed-text:latest";
    vectorDimensions = 768;
    distanceMetric = "Cosine";

    # Performance settings (optimized for RTX 3070)
    batchSize = 2000;
    maxConcurrent = 4;
    asyncChat = true;
    chunkSize = 2500;
    chunkOverlap = 200;
    semanticChunker = "auto"; # Use semantic chunker for chat, recursive for others

    # Service user settings
    user = "qdrant-upload";
    group = "qdrant-upload";

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
        customSource = "personal-vault";
        skipExisting = false;
        forceUpdate = false;
        # Uses default schedule
      }

      # Work Obsidian vault
      {
        name = "work-obsidian";
        type = "obsidian";
        collection = "work_notes";
        directories = [ "/home/user/Documents/WorkVault" ];
        customSource = "work-vault";
        skipExisting = false;
        forceUpdate = false;
        schedule = "*-*-* 01:00:00"; # Custom schedule: Daily at 1 AM
      }

      # Multiple general document directories into one collection
      {
        name = "research-docs";
        type = "general";
        collection = "research_papers";
        directories = [
          "/home/user/Documents/Papers"
          "/home/user/Documents/Research"
          "/home/user/Downloads/PDFs"
        ];
        customSource = "research-collection";
        skipExisting = true; # Skip unchanged documents for faster processing
        # Uses default schedule
      }

      # Code documentation
      {
        name = "code-docs";
        type = "general";
        collection = "code_documentation";
        directories = [
          "/home/user/Projects/docs"
          "/home/user/Documents/TechnicalNotes"
        ];
        customSource = "code-docs";
        forceUpdate = true; # Always update to catch changes
        schedule = "*-*-* 03:00:00"; # Custom schedule: Daily at 3 AM
      }

      # Chat conversations from Open-WebUI export
      {
        name = "chat-history";
        type = "chat";
        collection = "conversations";
        jsonFile = "/home/user/Downloads/open-webui-export.json";
        customSource = "chat-export";
        skipExisting = false;
        schedule = "*-*-* 04:00:00"; # Custom schedule: Daily at 4 AM
      }

      # Additional specialized collections
      {
        name = "meeting-notes";
        type = "obsidian";
        collection = "meetings";
        directories = [ "/home/user/Documents/MeetingNotes" ];
        customSource = "meeting-vault";
      }

      {
        name = "reference-materials";
        type = "general";
        collection = "references";
        directories = [
          "/home/user/Documents/References"
          "/home/user/Documents/Manuals"
          "/home/user/Documents/Guides"
        ];
        customSource = "reference-docs";
        skipExisting = true;
      }
    ];

    # Environment file for additional configuration
    environmentFile = "/etc/qdrant-upload/.env";
  };

  # Create user and group for the service
  users.users.qdrant-upload = {
    isSystemUser = true;
    group = "qdrant-upload";
    home = "/var/lib/qdrant-upload";
    createHome = true;
  };

  users.groups.qdrant-upload = { };

  # Environment file with additional configuration
  environment.etc."qdrant-upload/.env".text = ''
    # Additional Ollama configuration
    OLLAMA_URL=http://localhost:11434
    
    # Optional: Override global settings for specific scenarios
    # QDRANT_UPLOAD_BATCH_SIZE=1500
    # QDRANT_UPLOAD_MAX_CONCURRENT=3
  '';

  # Ensure proper permissions on the environment file
  systemd.tmpfiles.rules = [
    "d /etc/qdrant-upload 0755 root root -"
    "f /etc/qdrant-upload/.env 0640 root qdrant-upload -"
  ];
}

# This configuration creates:
# 1. Multiple systemd services: qdrant-upload-personal-obsidian, qdrant-upload-work-obsidian, etc.
# 2. Multiple systemd timers with different schedules
# 3. Each source processes into its own Qdrant collection
# 4. Flexible per-source settings (custom schedules, skip options, etc.)
#
# Commands to manage:
# - systemctl status qdrant-upload-personal-obsidian
# - systemctl start qdrant-upload-work-obsidian  
# - systemctl list-timers | grep qdrant-upload
# - journalctl -u qdrant-upload-research-docs -f
#
# Manual execution:
# - sudo -u qdrant-upload qdrant-upload obsidian --collection personal_notes --dirs /home/user/Documents/PersonalVault 
