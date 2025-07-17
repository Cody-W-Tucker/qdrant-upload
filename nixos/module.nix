{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.qdrant-upload;

  # Define source type for individual configurations
  sourceType = types.submodule {
    options = {
      name = mkOption {
        type = types.str;
        description = "Unique name for this source configuration";
      };

      type = mkOption {
        type = types.enum [ "general" "obsidian" "chat" ];
        description = "Type of documents to process";
      };

      collection = mkOption {
        type = types.str;
        description = "Qdrant collection name for this source";
      };

      directories = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Directories to process (for general and obsidian types)";
      };

      jsonFile = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Path to JSON file (for chat type)";
      };

      customSource = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Custom source identifier for all documents";
      };

      skipExisting = mkOption {
        type = types.bool;
        default = false;
        description = "Skip documents that already exist in the collection";
      };

      forceUpdate = mkOption {
        type = types.bool;
        default = false;
        description = "Force update all documents even if unchanged";
      };

      schedule = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Custom schedule for this source (systemd timer format). If null, uses global schedule.";
      };
    };
  };
in
{
  options.services.qdrant-upload = {
    enable = mkEnableOption "Enable the Qdrant document upload service";

    package = mkOption {
      type = types.package;
      default = pkgs.callPackage ./default.nix { };
      description = "The Qdrant uploader package to use";
    };

    # Global settings
    qdrantUrl = mkOption {
      type = types.str;
      default = "http://localhost:6333";
      description = "URL of the Qdrant service";
    };

    embeddingModel = mkOption {
      type = types.str;
      default = "nomic-embed-text:latest";
      description = "Ollama embedding model to use";
    };

    vectorDimensions = mkOption {
      type = types.int;
      default = 768;
      description = "Vector dimensions for the embedding model";
    };

    distanceMetric = mkOption {
      type = types.enum [ "Cosine" "Euclid" "Dot" ];
      default = "Cosine";
      description = "Distance metric to use for vector similarity";
    };

    batchSize = mkOption {
      type = types.int;
      default = 2000;
      description = "Batch size for document uploads (optimized for RTX 3070)";
    };

    minContentLength = mkOption {
      type = types.int;
      default = 50;
      description = "Minimum content length for documents";
    };

    # Performance settings
    chunkSize = mkOption {
      type = types.int;
      default = 2500;
      description = "Chunk size for text splitting";
    };

    chunkOverlap = mkOption {
      type = types.int;
      default = 200;
      description = "Chunk overlap for text splitting";
    };

    semanticChunker = mkOption {
      type = types.enum [ "true" "false" "auto" ];
      default = "false";
      description = "Semantic chunker mode: auto=chat only, true=all, false=none";
    };

    maxConcurrent = mkOption {
      type = types.int;
      default = 4;
      description = "Maximum concurrent batches (optimal for RTX 3070)";
    };

    asyncChat = mkOption {
      type = types.bool;
      default = true;
      description = "Enable high-performance async processing for chat";
    };

    # Ollama settings
    ollamaUrl = mkOption {
      type = types.str;
      default = "http://localhost:11434";
      description = "URL of the Ollama service";
    };

    ollamaTimeout = mkOption {
      type = types.int;
      default = 300;
      description = "Timeout in seconds for Ollama API requests";
    };

    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to environment file for additional configuration (optional)";
    };

    user = mkOption {
      type = types.str;
      default = "qdrant-upload";
      description = "User to run the service as";
    };

    group = mkOption {
      type = types.str;
      default = "qdrant-upload";
      description = "Group to run the service as";
    };

    # Multiple sources configuration
    sources = mkOption {
      type = types.listOf sourceType;
      default = [ ];
      description = "List of document sources to process, each with their own collection";
      example = literalExpression ''
        [
          {
            name = "personal-obsidian";
            type = "obsidian";
            collection = "personal_notes";
            directories = [ "/home/user/Documents/PersonalVault" ];
          }
          {
            name = "work-obsidian";
            type = "obsidian"; 
            collection = "work_notes";
            directories = [ "/home/user/Documents/WorkVault" ];
          }
          {
            name = "general-docs";
            type = "general";
            collection = "general_docs";
            directories = [ "/home/user/Documents/Papers" "/home/user/Documents/References" ];
          }
          {
            name = "chat-history";
            type = "chat";
            collection = "conversations";
            jsonFile = "/home/user/Downloads/open-webui-export.json";
          }
        ]
      '';
    };

    # Service settings
    defaultSchedule = mkOption {
      type = types.str;
      default = "*-*-* 02:00:00"; # Daily at 2 AM
      description = "Default schedule for service execution in systemd timer format";
    };

    packageOnly = mkOption {
      type = types.bool;
      default = false;
      description = "Only install the package without creating SystemD services";
    };
  };

  config = mkIf cfg.enable {
    # Always install package and set environment
    environment.systemPackages = [
      cfg.package
      # Add unstructured-api for better document format support
      pkgs.unstructured-api
    ];

    environment.variables = {
      # Qdrant settings
      QDRANT_UPLOAD_URL = cfg.qdrantUrl;
      QDRANT_UPLOAD_MODEL = cfg.embeddingModel;
      QDRANT_UPLOAD_DIMENSIONS = toString cfg.vectorDimensions;
      QDRANT_UPLOAD_DISTANCE = cfg.distanceMetric;

      # Performance settings
      QDRANT_UPLOAD_BATCH_SIZE = toString cfg.batchSize;
      QDRANT_UPLOAD_MIN_LENGTH = toString cfg.minContentLength;
      QDRANT_UPLOAD_CHUNK_SIZE = toString cfg.chunkSize;
      QDRANT_UPLOAD_CHUNK_OVERLAP = toString cfg.chunkOverlap;
      QDRANT_UPLOAD_SEMANTIC_CHUNKER = cfg.semanticChunker;
      QDRANT_UPLOAD_MAX_CONCURRENT = toString cfg.maxConcurrent;
      QDRANT_UPLOAD_ASYNC_CHAT = if cfg.asyncChat then "true" else "false";

      # Ollama settings
      OLLAMA_URL = cfg.ollamaUrl;
      OLLAMA_TIMEOUT = toString cfg.ollamaTimeout;
    };

    # Create user and group for the service (unless packageOnly)
    users.users.${cfg.user} = mkIf (!cfg.packageOnly) {
      isSystemUser = true;
      group = cfg.group;
      home = "/var/lib/qdrant-upload";
      createHome = true;
    };

    users.groups.${cfg.group} = mkIf (!cfg.packageOnly) { };

    # Create systemd services for each source (unless packageOnly)
    systemd.services = mkIf (!cfg.packageOnly) (
      # Create a service for each configured source
      listToAttrs
        (map
          (source: {
            name = "qdrant-upload-${source.name}";
            value = {
              description = "Qdrant Document Upload Service for ${source.name} (${source.type})";
              after = [ "network.target" ];
              wantedBy = mkIf (source.schedule == null && cfg.defaultSchedule == null) [ "multi-user.target" ];
              path = [ cfg.package ];

              serviceConfig = {
                Type = "oneshot";
                ExecStart =
                  let
                    args = [
                      "--type ${source.type}"
                      "--collection ${source.collection}"
                    ] ++ (
                      if source.type == "chat" then
                        optional (source.jsonFile != null) "--json-file ${source.jsonFile}"
                      else
                        optional (source.directories != [ ]) "--dirs ${concatStringsSep " " source.directories}"
                    ) ++ (
                      optional (source.customSource != null) "--source '${source.customSource}'"
                    ) ++ (
                      optional source.skipExisting "--skip-existing"
                    ) ++ (
                      optional source.forceUpdate "--force-update"
                    );
                  in
                  "${cfg.package}/bin/qdrant-upload ${concatStringsSep " " args}";
                User = cfg.user;
                Group = cfg.group;
              } // (if cfg.environmentFile != null then {
                EnvironmentFile = cfg.environmentFile;
              } else { });
            };
          })
          cfg.sources)
    );

    # Create timers for each source (unless packageOnly)
    systemd.timers = mkIf (!cfg.packageOnly) (
      # Create timers for each configured source
      listToAttrs
        (map
          (source: {
            name = "qdrant-upload-${source.name}";
            value = mkIf (source.schedule != null || cfg.defaultSchedule != null) {
              description = "Timer for Qdrant Upload Service - ${source.name}";
              wantedBy = [ "timers.target" ];

              timerConfig = {
                OnCalendar = if source.schedule != null then source.schedule else cfg.defaultSchedule;
                Unit = "qdrant-upload-${source.name}.service";
                Persistent = true;
              };
            };
          })
          cfg.sources)
    );

    # Validation warnings
    warnings =
      optional (cfg.sources == [ ])
        "No sources configured. You must configure at least one source in 'sources'." ++
      (builtins.filter (x: x != null) (map
        (source:
          if source.type == "chat" && source.jsonFile == null then
            "Source '${source.name}' is type 'chat' but has no jsonFile configured"
          else if source.type != "chat" && source.directories == [ ] then
            "Source '${source.name}' is type '${source.type}' but has no directories configured"
          else null
        )
        cfg.sources));
  };
}
