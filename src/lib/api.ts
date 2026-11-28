import { invoke } from "@tauri-apps/api/core";

import type {
  ActiveEngineInfo,
  AgentApprovalReply,
  AutoTuneResult,
  AgentRunRequest,
  AgentUndoResult,
  AppSettings,
  BackendCapabilities,
  BinaryInstallResult,
  BinaryProbe,
  ChatRequestPayload,
  ChatSession,
  ChatBackupInfo,
  ChatImportSummary,
  ChatStoreStats,
  EngineKind,
  ManagedBinaryProbe,
  ConversionStarted,
  ConvertRequest,
  DownloadRequest,
  DownloadStarted,
  EngineSnapshot,
  GatewayStatus,
  GgufModelInfo,
  HardwareProfile,
  ModelFitEstimate,
  HfModel,
  HfQuant,
  HfRepoBundles,
  KnowledgeCollection,
  KnowledgeCollectionDetail,
  RagIndexSummary,
  RagQueryResult,
  CompanionPairing,
  GgufCompanionInfo,
  LibraryCompanions,
  ModelBundle,
  BundleCheck,
  LibraryModel,
  LibraryDisk,
  LocalModel,
  McpProbe,
  McpServerInput,
  ModelVerification,
  MemoryFact,
  ProjectIngest,
  SearchRequest,
  SetupReport,
  StorageBenchmark,
  Telemetry,
  TranscriptionResult,
  WebSearchResponse,
  WhisperInstallResult,
  WhisperModelDownload,
  WhisperStatus,
} from "../types";

export const Events = {
  engineStatus: "engine://status",
  engineLog: "engine://log",
  engineExited: "engine://exited",
  engineActive: "engine://active",
  engineTelemetry: "engine://telemetry",
  engineAdapterWarning: "engine://adapter-warning",
  engineBinaryProgress: "engine://binary-progress",
  whisperProgress: "whisper://progress",
  engineSetupStatus: "setup://status",
  gatewayStatus: "gateway://status",
  chatDelta: "chat://delta",
  chatDone: "chat://done",
  chatError: "chat://error",
  chatNotice: "chat://notice",
  chatToolApproval: "chat://tool-approval",
  memoryChanged: "memory://changed",
  convertStarted: "convert://started",
  convertLog: "convert://log",
  convertProgress: "convert://progress",
  convertFinished: "convert://finished",
  downloadProgress: "download://progress",
  mcpActivity: "mcp://activity",
  agentPlan: "agent:plan",
  agentThought: "agent:thought",
  agentToolStart: "agent:tool_start",
  agentApprovalNeeded: "agent:approval_needed",
  agentToolResult: "agent:tool_result",
  agentContextCompacted: "agent:context_compacted",
  agentDone: "agent:done",
  agentError: "agent:error",
  agentNotice: "agent:notice",
  ragIndexProgress: "rag://index-progress",
  chatRagSources: "chat://rag-sources",
} as const;

export const api = {
  getSettings: () => invoke<AppSettings>("get_settings"),
  saveSettings: (next: AppSettings) =>
    invoke<AppSettings>("save_settings", { next }),
  probeBinary: (path: string) =>
    invoke<BinaryProbe>("probe_binary", { path }),
  scanLocalModels: () => invoke<LocalModel[]>("scan_local_models"),
  inspectGgufModel: (path: string) =>
    invoke<GgufModelInfo>("inspect_gguf_model", { path }),

  getEngineStatus: () => invoke<EngineSnapshot>("get_engine_status"),
  getActiveEngine: () => invoke<ActiveEngineInfo | null>("get_active_engine"),
  getBackendCapabilities: () =>
    invoke<BackendCapabilities>("get_backend_capabilities"),
  startEngine: () => invoke<EngineSnapshot>("start_engine"),
  stopEngine: () => invoke<void>("stop_engine"),
  switchEngine: (engine: EngineKind | "none") =>
    invoke<EngineSnapshot>("switch_engine", { engine }),
  downloadAndExtractBinary: (engineType: EngineKind, downloadUrl: string) =>
    invoke<BinaryInstallResult>("download_and_extract_binary", {
      engine_type: engineType,
      download_url: downloadUrl,
    }),
  rollbackBinary: (engineType: EngineKind) =>
    invoke<BinaryInstallResult>("rollback_binary", {
      engine_type: engineType,
    }),
  probeManagedBinary: (engineType: EngineKind) =>
    invoke<ManagedBinaryProbe>("probe_managed_binary", {
      engine_type: engineType,
    }),
  getEngineSetupStatus: () =>
    invoke<SetupReport>("get_engine_setup_status"),
  rerunEngineSetup: () =>
    invoke<SetupReport>("rerun_engine_setup"),

  startGateway: () => invoke<GatewayStatus>("start_gateway"),
  stopGateway: () => invoke<GatewayStatus>("stop_gateway"),
  getGatewayStatus: () => invoke<GatewayStatus>("get_gateway_status"),

  streamChat: (request: ChatRequestPayload) =>
    invoke<void>("stream_chat", { request }),
  cancelChat: (requestId: string) =>
    invoke<void>("cancel_chat", { request_id: requestId }),
  resolveChatToolApproval: (
    requestId: string,
    callId: string,
    decision: "approve" | "reject",
  ) =>
    invoke<boolean>("chat_tool_approval", {
      request_id: requestId,
      call_id: callId,
      decision,
    }),

  chatSessionsList: () => invoke<ChatSession[]>("chat_sessions_list"),
  chatSessionSave: (session: ChatSession) =>
    invoke<ChatSession>("chat_session_save", { session }),
  chatSessionDelete: (id: string) =>
    invoke<boolean>("chat_session_delete", { id }),
  chatAssetDataUrl: (asset: string) =>
    invoke<string>("chat_asset_data_url", { asset }),
  chatExportSession: (id: string, path: string) =>
    invoke<void>("chat_export_session", { id, path }),
  chatImportSession: (path: string) =>
    invoke<ChatSession>("chat_import_session", { path }),
  chatBranchSession: (id: string, messageId: string) =>
    invoke<ChatSession>("chat_branch_session", {
      id,
      message_id: messageId,
    }),
  chatExportAll: (path: string) =>
    invoke<number>("chat_export_all", { path }),
  chatImportAll: (path: string) =>
    invoke<ChatImportSummary>("chat_import_all", { path }),
  chatBackupList: () => invoke<ChatBackupInfo[]>("chat_backup_list"),
  chatRestoreBackup: (id: string) =>
    invoke<ChatSession>("chat_restore_backup", { id }),
  chatStoreGc: () => invoke<number>("chat_store_gc"),
  chatStoreStats: () => invoke<ChatStoreStats>("chat_store_stats"),

  startConversion: (request: ConvertRequest) =>
    invoke<ConversionStarted>("start_conversion", { request }),
  cancelConversion: () => invoke<void>("cancel_conversion"),

  searchModels: (request: SearchRequest) =>
    invoke<HfModel[]>("search_models", { request }),
  listModelQuants: (repo: string) =>
    invoke<HfQuant[]>("list_model_quants", { repo }),
  inspectRepoBundles: (repo: string) =>
    invoke<HfRepoBundles>("inspect_repo_bundles", { repo }),
  startDownload: (request: DownloadRequest) =>
    invoke<DownloadStarted>("start_download", { request }),
  cancelDownload: (id: string) => invoke<boolean>("cancel_download", { id }),
  pauseDownload: (id: string) => invoke<boolean>("pause_download", { id }),
  resumeDownload: (id: string) => invoke<boolean>("resume_download", { id }),

  ragListCollections: () =>
    invoke<KnowledgeCollection[]>("rag_list_collections"),
  ragCollectionDetail: (id: string) =>
    invoke<KnowledgeCollectionDetail>("rag_collection_detail", { id }),
  ragCreateCollection: (
    name: string,
    description: string,
    sourcePaths: string[],
  ) =>
    invoke<KnowledgeCollection>("rag_create_collection", {
      name,
      description,
      source_paths: sourcePaths,
    }),
  ragDeleteCollection: (id: string) =>
    invoke<boolean>("rag_delete_collection", { id }),
  ragIndexPaths: (collectionId: string, paths: string[]) =>
    invoke<RagIndexSummary>("rag_index_paths", {
      collection_id: collectionId,
      paths,
    }),
  ragReindexCollection: (collectionId: string) =>
    invoke<RagIndexSummary>("rag_reindex_collection", {
      collection_id: collectionId,
    }),
  ragCancelIndex: (collectionId: string) =>
    invoke<boolean>("rag_cancel_index", { collection_id: collectionId }),
  ragQuery: (
    collectionIds: string[],
    query: string,
    topK?: number,
    maxTokens?: number,
  ) =>
    invoke<RagQueryResult>("rag_query", {
      collection_ids: collectionIds,
      query,
      top_k: topK ?? null,
      max_tokens: maxTokens ?? null,
    }),
  ragRemovePath: (collectionId: string, path: string) =>
    invoke<KnowledgeCollection>("rag_remove_path", {
      collection_id: collectionId,
      path,
    }),

  memoryList: () => invoke<MemoryFact[]>("memory_list"),
  saveToMemoryBank: (fact: string) =>
    invoke<MemoryFact>("save_to_memory_bank", { fact }),
  memoryUpdate: (id: string, text: string) =>
    invoke<MemoryFact>("memory_update", { id, text }),
  memoryDelete: (id: string) => invoke<boolean>("memory_delete", { id }),
  memoryClear: () => invoke<void>("memory_clear"),
  openExternal: (url: string) => invoke<void>("open_external", { url }),

  getTelemetry: (pid: number | null) =>
    invoke<Telemetry>("get_telemetry", { pid }),
  benchmarkStorage: (directory?: string) =>
    invoke<StorageBenchmark>("benchmark_storage", {
      directory: directory ?? null,
    }),
  ingestProject: (path: string) =>
    invoke<ProjectIngest>("ingest_project", { path }),
  webSearch: (query: string, pages = 3) =>
    invoke<WebSearchResponse>("web_search", { query, pages }),
  transcribeAudio: (
    audioBase64: string,
    mime: string,
    fileName?: string | null,
    model?: string | null,
    language?: string | null,
    stripNonSpeech?: boolean,
  ) =>
    invoke<TranscriptionResult>("transcribe_audio", {
      audioBase64,
      mime,
      fileName: fileName ?? null,
      model: model ?? null,
      language: language ?? null,
      stripNonSpeech: stripNonSpeech ?? true,
    }),
  transcribeAudioFile: (
    path: string,
    model?: string | null,
    language?: string | null,
  ) =>
    invoke<TranscriptionResult>("transcribe_audio_file", {
      path,
      model: model ?? null,
      language: language ?? null,
    }),
  getWhisperStatus: () => invoke<WhisperStatus>("get_whisper_status"),
  startNativeCapture: () => invoke<void>("start_native_capture"),
  stopNativeCapture: () =>
    invoke<{
      audioBase64: string;
      mime: string;
      durationMs: number;
    }>("stop_native_capture"),
  cancelNativeCapture: () => invoke<boolean>("cancel_native_capture"),
  nativeCaptureLevel: () => invoke<number>("native_capture_level"),
  installWhisperBinary: (backend?: "auto" | "cpu" | "cuda" | "vulkan") =>
    invoke<WhisperInstallResult>("install_whisper_binary", {
      backend: backend ?? "auto",
    }),
  downloadWhisperModel: (modelId: string) =>
    invoke<WhisperModelDownload>("download_whisper_model", { modelId }),
  cancelWhisperDownload: () =>
    invoke<boolean>("cancel_whisper_download"),
  libraryModels: () => invoke<LibraryModel[]>("library_models"),
  libraryDisk: () => invoke<LibraryDisk>("library_disk"),
  getHardwareProfile: () => invoke<HardwareProfile>("get_hardware_profile"),
  /**
   * Estimate one quantization against this machine. `block_count` comes from a
   * GGUF header when one is available; otherwise the backend infers layers
   * from `parameters`. `context_size` null keeps the backend default.
   */
  estimateQuantFit: (
    fileSizeBytes: number,
    blockCount: number | null,
    contextSize: number | null,
    parameters: number | null,
  ) =>
    invoke<ModelFitEstimate>("estimate_quant_fit", {
      file_size_bytes: fileSizeBytes,
      block_count: blockCount,
      context_size: contextSize,
      parameters,
    }),
  autoTuneForModel: (modelPath: string | null) =>
    invoke<AutoTuneResult>("auto_tune_for_model", {
      model_path: modelPath,
    }),
  verifyLocalModel: (path: string) =>
    invoke<ModelVerification>("verify_local_model", { path }),
  deleteLocalModel: (path: string) =>
    invoke<number>("delete_local_model", { path }),

  inspectGgufCompanion: (path: string) =>
    invoke<GgufCompanionInfo>("inspect_gguf_companion", { path }),
  inspectModelCompanions: (path: string) =>
    invoke<CompanionPairing>("inspect_model_companions", { path }),
  libraryCompanions: () => invoke<LibraryCompanions>("library_companions"),
  validateModelBundle: (bundle: ModelBundle) =>
    invoke<BundleCheck>("validate_model_bundle", { bundle }),
  saveModelBundle: (bundle: ModelBundle) =>
    invoke<AppSettings>("save_model_bundle", { bundle }),
  deleteModelBundle: (id: string) =>
    invoke<AppSettings>("delete_model_bundle", { id }),
  applyModelBundle: (id: string) =>
    invoke<AppSettings>("apply_model_bundle", { id }),
  pairModelCompanions: (
    mainPath: string,
    mmprojPath?: string | null,
    draftPath?: string | null,
    draftMaxTokens?: number,
  ) =>
    invoke<AppSettings>("pair_model_companions", {
      main_path: mainPath,
      mmproj_path: mmprojPath ?? null,
      draft_path: draftPath ?? null,
      draft_max_tokens: draftMaxTokens ?? null,
    }),
  mcpProbe: (server: McpServerInput) =>
    invoke<McpProbe>("mcp_probe", { server }),

  startAgent: (request: AgentRunRequest) =>
    invoke<void>("agent_start", { request }),
  resolveAgentApproval: (reply: AgentApprovalReply) =>
    invoke<void>("agent_resolve_approval", { reply }),
  abortAgent: (runId: string) =>
    invoke<void>("agent_abort", { run_id: runId }),
  undoAgent: (runId: string) =>
    invoke<AgentUndoResult>("agent_undo", { run_id: runId }),
  undoAgentFile: (runId: string, path: string) =>
    invoke<AgentUndoResult>("agent_undo_file", { run_id: runId, path }),
  keepAgentFile: (runId: string, path: string) =>
    invoke<AgentUndoResult>("agent_keep_file", { run_id: runId, path }),
  agentUndoAvailability: (runIds: string[]) =>
    invoke<string[]>("agent_undo_availability", { run_ids: runIds }),
  agentUndoState: (runId: string) =>
    invoke<AgentUndoResult>("agent_undo_state", { run_id: runId }),
  agentValidateWorkspace: (path: string) =>
    invoke<string | null>("agent_validate_workspace", { path }),
  agentApprovalRules: (workspace: string) =>
    invoke<string[]>("agent_approval_rules", { workspace }),
  forgetApprovalRule: (workspace: string, command: string) =>
    invoke<string[]>("agent_forget_approval_rule", { workspace, command }),
  clearApprovalRules: (workspace: string) =>
    invoke<string[]>("agent_clear_approval_rules", { workspace }),
};
