{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.qdrant-upload;
in
{
  options.services.qdrant-upload = {
    enable = mkEnableOption "Enable the Qdrant document uploader service";

    package = mkOption {
      type = types.package;
      default = pkgs.callPackage ./default.nix { };
      description = "The Qdrant uploader package to use";
    };

    qdrantUrl = mkOption {
      type = types.str;
      default = "http://localhost:6333";
      description = "URL of the Qdrant service";
    };

    defaultCollection = mkOption {
      type = types.str;
      default = "personal";
      description = "Default collection name to use";
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
      default = 500;
      description = "Batch size for document uploads";
    };

    minContentLength = mkOption {
      type = types.int;
      default = 10;
      description = "Minimum content length for documents";
    };

    obsidianDirectories = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Default Obsidian vault directories to process";
    };

    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to environment file for additional configuration";
    };

    defaultDocumentType = mkOption {
      type = types.enum [ "general" "obsidian" "chat" ];
      default = "general";
      description = "Default document type to process";
    };

    user = mkOption {
      type = types.str;
      default = "nobody";
      description = "User to run the service as";
    };

    group = mkOption {
      type = types.str;
      default = "nobody";
      description = "Group to run the service as";
    };

    enableService = mkOption {
      type = types.bool;
      default = false;
      description = "Enable the Qdrant uploader systemd service";
    };

    serviceSchedule = mkOption {
      type = types.str;
      default = "*-*-* 00:00:00"; # Daily at midnight
      description = "Schedule for service execution in systemd timer format";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = [
      cfg.package
      # Add unstructured-api for better document format support
      pkgs.unstructured-api
    ];

    environment.variables = {
      QDRANT_UPLOAD_URL = cfg.qdrantUrl;
      QDRANT_UPLOAD_COLLECTION = cfg.defaultCollection;
      QDRANT_UPLOAD_MODEL = cfg.embeddingModel;
      QDRANT_UPLOAD_DIMENSIONS = toString cfg.vectorDimensions;
      QDRANT_UPLOAD_DISTANCE = cfg.distanceMetric;
      QDRANT_UPLOAD_BATCH_SIZE = toString cfg.batchSize;
      QDRANT_UPLOAD_MIN_LENGTH = toString cfg.minContentLength;
      QDRANT_FOLDERS = concatStringsSep " " cfg.obsidianDirectories;
    };

    # Add systemd service and timer if enabled
    systemd.services.qdrant-upload = mkIf cfg.enableService {
      description = "Qdrant Document Upload Service";
      after = [ "network.target" ];
      wantedBy = mkIf (cfg.serviceSchedule == null) [ "multi-user.target" ];
      path = [ cfg.package ];

      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${cfg.package}/bin/qdrant-upload ${cfg.defaultDocumentType} --collection ${cfg.defaultCollection}";
        User = cfg.user;
        Group = cfg.group;
      } // (if cfg.environmentFile != null then {
        EnvironmentFile = cfg.environmentFile;
      } else { });
    };

    # Add timer for scheduled execution
    systemd.timers.qdrant-upload = mkIf (cfg.enableService && cfg.serviceSchedule != null) {
      description = "Timer for Qdrant Document Upload Service";
      wantedBy = [ "timers.target" ];

      timerConfig = {
        OnCalendar = cfg.serviceSchedule;
        Unit = "qdrant-upload.service";
        Persistent = true;
      };
    };
  };
}
