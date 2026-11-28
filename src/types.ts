export type ViewId =
  | "chat"
  | "models"
  | "library"
  | "compiler"
  | "benchmark"
  | "knowledge"
  | "mcp"
  | "settings";

export type EngineStatus = "disconnected" | "indexing" | "loading_model" | "ready";
export type EngineKind = "janas" | "llama";

export interface EngineSnapshot {
  status: EngineStatus;
  detail: string;
  pid: number | null;
  model: string | null;
  port: number;
  started_at_ms: number | null;
}

export interface EngineLog {
  stream: "stdout" | "stderr" | string;
  line: string;
  timestamp_ms: number;
}

export interface ActiveEngineInfo {
  kind: EngineKind;
  pid: number;
  host: string;
  port: number;
  model: string;
  context_window: number;
  version: string | null;
  adapter: string;
  warning: string | null;
  started_at_ms: number;
}

export interface EngineTelemetry {
  engine: EngineKind;
  pid: number | null;
  tokens_per_second: number;
  prompt_tokens_total: number;
  completion_tokens_total: number;
  kv_cache_usage: number;
  read_bytes_per_sec: number;
  read_ops_per_sec: number;
  rss_bytes: number;
  sampled_at_ms: number;
}

/** Which llama.cpp release automatic setup installs. */
export type LlamaUpdateChannel = "baseline" | "latest";

/** A GPU launch rung the supervisor can fall back to after a crash. */
export type EngineFallbackStep = "host-kv" | "cpu";

export type AdapterWarningKind = "adapter" | "fallback" | "auto-rollback";

export interface AdapterWarningEvent {
  engine: EngineKind;
  message: string;
  version: string | null;
  adapter: string;
  /** Why the warning was raised. Absent on older backends. */
  kind?: AdapterWarningKind;
  /** Set when this warning describes a fallback rung. */
  step?: EngineFallbackStep | null;
  /** The newly installed version that failed its first real launch. */
  failed_version?: string | null;
  /** The version an automatic rollback restored. */
  restored_version?: string | null;
}

export interface BinaryProgress {
  engine: EngineKind;
  phase: string;
  downloaded: number;
  total: number | null;
  percent: number | null;
  message: string;
}

export interface BinaryInstallResult {
  engine: EngineKind;
  installed_path: string;
  backup_path: string | null;
  version: string | null;
  message: string;
  rolled_back: boolean;
}

export interface ManagedBinaryProbe {
  engine: EngineKind;
  installed: boolean;
  path: string;
  version: string | null;
  build: number | null;
  message: string;
}

export type SetupStatus = "idle" | "running" | "ready" | "degraded" | "failed";
export type SetupComponentStatus =
  | "pending"
  | "present"
  | "installed"
  | "skipped"
  | "failed";

export interface SetupGpu {
  vendor: string;
  backend: string;
  name: string;
  cuda_version: string | null;
}

export interface SetupPlatform {
  os: string;
  arch: string;
  cpu_avx2: boolean;
  apple_silicon: boolean;
  gpus: SetupGpu[];
  preferred_backend: string;
}

export interface SetupRelease {
  repo: string;
  requested_tag: string;
  tag: string;
  asset: string;
  download_url: string;
  fallback: boolean;
}

export interface SetupBinary {
  name: string;
  status: SetupComponentStatus;
  path: string | null;
  version: string | null;
  source: string | null;
  message: string;
}

export interface SetupReport {
  status: SetupStatus;
  message: string;
  started_at_ms: number | null;
  finished_at_ms: number | null;
  platform: SetupPlatform | null;
  cache_dir: string | null;
  llama_release: SetupRelease | null;
  janas_release: SetupRelease | null;
  binaries: SetupBinary[];
  warnings: string[];
  /** Channel used for this run: the tested baseline or upstream latest. */
  llama_update_channel: LlamaUpdateChannel;
  /** True when a remote compat.json was fetched and applied. */
  compat_applied: boolean;
  /** Tag the manifest recommends instead of the built-in baseline. */
  compat_recommended_tag: string | null;
  /** Tags compat.json marks as broken. */
  blocked_llama_tags: string[];
  /** Broken tags resolution actually skipped this run. */
  skipped_llama_tags: string[];
  /** The tag the baseline channel asked for. */
  llama_baseline_tag: string;
  /** Highest llama.cpp build whose Vulkan KV offload is trusted. */
  vulkan_kv_offload_limit: number | null;
}

export interface AppSettings {
  engine_path: string;
  converter_path: string;
  host: string;
  port: number;
  models_dir: string;
  model_path: string;
  extra_args: string;
  api_key: string;
  /** Optional Hugging Face token for gated or private repositories. */
  hf_token: string;
  /** --ctx: token window the server allocates. */
  ctx_size: number;
  /** --cache: GiB reserved for streamed experts; 0 is automatic. */
  cache_gib: number;
  /** --mtp / --no-mtp: "auto" | "off" | "file". */
  mtp_mode: "auto" | "off" | "file" | string;
  mtp_path: string;
  draft_path: string;
  prompt_on: string;
  reply_on: string;
  drafts_on: string;
  think: "auto" | "on" | "off" | string;
  npu_int8: boolean;
  engine_mode: string;
  /** JANAS_ATTN: auto, exact (float) or fast (16-bit query). */
  janas_attention: "auto" | "exact" | "fast" | string;
  /** JANAS_KV: auto, 8 or 16 bits for both keys and values. */
  janas_kv_cache: "auto" | "8" | "16" | string;
  llama_model_path: string;
  llama_mmproj_path: string;
  /** llama.cpp: --spec-type draft-mtp / none, or auto to keep the build's default. */
  llama_mtp_mode: "auto" | "off" | "mtp" | string;
  /** llama.cpp: optional MTP head GGUF, used as the speculative draft model. */
  llama_mtp_path: string;
  /** llama.cpp: --spec-draft-model, a smaller same-family GGUF. */
  llama_draft_path: string;
  /** llama.cpp: --draft-max, how many tokens the draft proposes per step. */
  llama_draft_max: number;
  /** llama.cpp: --draft-min, the shortest accepted draft run. */
  llama_draft_min: number;
  /** llama.cpp: --draft-p-min, the minimum draft probability kept. */
  llama_draft_p_min: number;
  /** llama.cpp: where the draft runs: auto, cpu (--spec-draft-device none) or gpu. */
  llama_drafts_on: "auto" | "cpu" | "gpu" | string;
  llama_port: number;
  llama_gpu_layers: number;
  llama_ctx_size: number;
  llama_flash_attn: "auto" | "on" | "off" | string;
  llama_kv_cache_k: string;
  llama_kv_cache_v: string;
  /** llama.cpp only: execute MoE cache misses on host CPU experts. */
  llama_direct_cpu_moe: boolean;
  /** llama.cpp only: max CPU threads for direct MoE compute; 0 = default. */
  llama_cpu_threads: number;
  llama_extra_args: string;
  /** Which llama.cpp release automatic setup installs. */
  llama_update_channel: LlamaUpdateChannel;
  ssd_streaming: boolean;
  ssd_cache_dir: string;
  top_k: number;
  /** Saved main + projector + draft bundles. */
  model_bundles: ModelBundle[];
  /** The bundle the user last activated. */
  active_bundle_id: string;
  /** Curated whisper.cpp model id used for local voice input. */
  whisper_model: string;
  /** whisper-cli -l value: "auto" or a language code such as "en". */
  whisper_language: string;
  /** Show the microphone button in the chat composer. */
  voice_input_enabled: boolean;
  /** "toggle" (click to start/stop) or "hold" (push-to-talk). */
  voice_activation: "toggle" | "hold" | string;
  /** Send a finished transcript immediately instead of inserting it. */
  voice_auto_send: boolean;
  /** Drop annotations such as [BLANK_AUDIO] or (music). */
  voice_strip_non_speech: boolean;
  /** Expose the loopback OpenAI-compatible gateway. */
  gateway_enabled: boolean;
  /** Loopback port; defaults to 11435. */
  gateway_port: number;
  /** Let a request's `model` switch the loaded engine and model. */
  gateway_auto_switch_models: boolean;
  /** Add Memory Bank facts to external gateway requests. */
  gateway_inject_memory: boolean;
  /** Document Vault collections injected into external gateway requests. */
  gateway_default_rag_collections: string[];
}

/** State of the loopback OpenAI-compatible gateway. */
export interface GatewayStatus {
  running: boolean;
  /** Origin of the server, for example http://127.0.0.1:11435. */
  url: string;
  /** OpenAI base URL external clients paste into their configuration. */
  api_base: string;
  port: number;
  active_engine: EngineKind | null;
  active_model: string | null;
  request_count: number;
  last_error: string | null;
  /** Non-fatal warning about the last request's context enrichment. */
  last_context_warning: string | null;
  started_at_ms: number | null;
}

export interface WhisperModelStatus {
  id: string;
  file_name: string;
  label: string;
  size_bytes: number;
  ram_mb: number;
  multilingual: boolean;
  note: string;
  installed: boolean;
  path: string;
}

export interface WhisperStatus {
  installed: boolean;
  binary_path: string;
  library_path: string;
  version: string | null;
  models_dir: string;
  models: WhisperModelStatus[];
  selected_model: string;
  active_model: string | null;
  language: string;
  voice_enabled: boolean;
  message: string;
}

export interface WhisperInstallResult {
  installed_path: string;
  version: string | null;
  release_tag: string;
  asset_name: string;
  message: string;
}

export interface WhisperModelDownload {
  model_id: string;
  path: string;
  size_bytes: number;
  sha256: string;
  message: string;
}

export interface WhisperProgress {
  phase: string;
  model: string | null;
  downloaded: number;
  total: number | null;
  percent: number | null;
  message: string;
}

export interface TranscriptionResult {
  text: string;
  /** Wall-clock time spent decoding and running whisper-cli. */
  duration_ms: number;
  model_used: string;
  /** Approximate duration of the normalised audio, when known. */
  audio_ms: number;
}

export interface BinaryProbe {
  ok: boolean;
  resolved_path: string | null;
  version: string | null;
  message: string;
}

export interface LocalModel {
  name: string;
  path: string;
  size_bytes: number;
}

export interface GgufModelInfo {
  path: string;
  architecture: string | null;
  layer_count: number | null;
}

export interface ChatUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  janas?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Citation {
  title: string;
  url: string;
  snippet?: string;
}

/**
 * A chunk returned by the local document vault. `citation` is the bare tag
 * ("D1"); replies reference it as `[D1]`, the same shape the web pipeline uses
 * for `[S1]`.
 */
export interface RagChunk {
  chunk_id: string;
  collection_id: string;
  collection_name: string;
  /** Absolute path on this machine. */
  file_path: string;
  relative_path: string;
  start_line: number;
  end_line: number;
  content: string;
  heading: string;
  token_estimate: number;
  score: number;
  lexical_score: number;
  semantic_score: number;
  /** Bare tag (`D1`); prompts and pills add the surrounding brackets. */
  citation: string;
}

export interface RagQueryResult {
  chunks: RagChunk[];
  tokens_used: number;
  budget_tokens: number;
  candidates: number;
  elapsed_ms: number;
  collections: string[];
  note: string | null;
}

/** Emitted for the turn currently streaming, keyed by the assistant message id. */
export interface RagSourcesEvent {
  request_id: string;
  chunks: RagChunk[];
  tokens_used: number;
  collections: string[];
}

export interface KnowledgeCollection {
  id: string;
  name: string;
  description: string;
  source_paths: string[];
  file_count: number;
  chunk_count: number;
  token_count: number;
  byte_count: number;
  created_at: number;
  updated_at: number;
}

export interface IndexedKnowledgeFile {
  path: string;
  relative_path: string;
  bytes: number;
  modified_ms: number;
  sha256: string;
  chunks: number;
  tokens: number;
  indexed_at: number;
  /** Present for a readable file that produced no text, such as a scanned PDF. */
  skipped_reason?: string | null;
}

export interface KnowledgeCollectionDetail {
  collection: KnowledgeCollection;
  files: IndexedKnowledgeFile[];
}

export interface RagIndexProgress {
  collection_id: string;
  phase: "scanning" | "indexing" | "writing" | "done" | "cancelled" | "error" | string;
  files_total: number;
  files_done: number;
  chunks_done: number;
  current_file: string;
  percent: number;
  message: string;
}

export interface RagIndexSummary {
  collection: KnowledgeCollection;
  files_indexed: number;
  files_unchanged: number;
  files_removed: number;
  chunks: number;
  tokens: number;
  cancelled: boolean;
  warnings: string[];
}

export interface MemoryFact {
  id: string;
  text: string;
  created_at: number;
  source: string;
}

/**
 * Where a pinned/evicted context item came from. The id carries the same
 * prefix (`file:`, `rag:`, `message:`, …) so a session can be normalized
 * without a lookup table.
 */
export type ContextItemKind =
  | "file"
  | "rag"
  | "web"
  | "memory"
  | "message"
  | "mcp";

/**
 * One context item the user explicitly pinned. Pinned items are never dropped
 * or truncated by the context budget, and are preserved intact by agent
 * auto-compaction.
 */
export interface PinnedContextItem {
  /** Stable id: `file:src/main.rs`, `rag:<chunk>`, `message:<id>`, `memory:<id>`. */
  id: string;
  kind: ContextItemKind;
  label: string;
  /** Conservative token estimate captured when the item was pinned. */
  tokens: number;
  added_at: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  created_at: number;
  citations?: Citation[];
  images?: Array<{
    name: string;
    /** Present on freshly attached images; absent after a reload. */
    data_url?: string;
    /** Session-relative reference to an image stored on disk. */
    asset?: string;
    tokens: number;
  }>;
  /** Structured agent trace. Absent on legacy chats and normal replies. */
  agentTrace?: AgentTrace | null;
  /** Document chunks retrieved for this reply, so `[D1]` pills survive reload. */
  rag_sources?: RagChunk[];
  /** Parent in the conversation tree. Null/undefined marks a root message. */
  parent_id?: string | null;
  /** The child that was active when this message was last on the main path. */
  active_child_id?: string | null;
}

export interface ChatDraftImage {
  name: string;
  data_url?: string;
  asset?: string;
  tokens: number;
}

export interface ChatDraft {
  text: string;
  images: ChatDraftImage[];
  updated_at: number;
}

export interface ChatSession {
  id: string;
  schema_version?: number;
  title: string;
  created_at: number;
  updated_at: number;
  /** Pinned chats stay at the top of the rail. */
  pinned?: boolean;
  messages: ChatMessage[];
  /** Inactive branches preserved by an edit or regenerate. */
  archived_messages?: ChatMessage[];
  /** Composer text and pending images, persisted per chat. */
  draft?: ChatDraft;
  /** Project bundle attached to this chat, not the whole window. */
  project?: ProjectIngest;
  /** Per-chat agent sandbox override; falls back to the global preference. */
  workspace_root?: string;
  /** MCP servers this chat may use; undefined means all enabled, [] none. */
  mcp_server_ids?: string[];
  /** Server names captured with the scope, used to remap re-created servers. */
  mcp_server_names?: Record<string, string>;
  /** Per-chat instructions; undefined falls back to the saved default. */
  system_prompt?: string;
  /** Knowledge collections this chat may query; undefined or [] means none. */
  attached_collection_ids?: string[];
  /** Per-chat retrieval tuning. */
  rag_top_k?: number;
  rag_max_tokens?: number;
  /** Context items the user pinned for this chat. */
  pinned_context?: PinnedContextItem[];
  /**
   * Context items dropped from the prompt but kept in the transcript. Message
   * entries use `message:<id>`; files, RAG chunks and memories use their own
   * prefixed ids.
   */
  evicted_context_ids?: string[];
}

export interface ChatOutboundMessage {
  role: "user" | "assistant";
  content: string;
  /** The user pinned this turn; budget trimming must not drop it. */
  pinned?: boolean;
}

export interface ChatRequestPayload {
  request_id: string;
  messages: ChatOutboundMessage[];
  system_prompt: string | null;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number | null;
  repeat_penalty: number | null;
  structured: boolean;
  thinking: boolean | null;
  images: string[];
  max_tokens: number | null;
  /** Pinned project/memory context, appended to the system prompt. */
  pinned_context?: string | null;
  mcp_servers: McpServerInput[];
  rag_collections: string[];
  rag_top_k: number;
  rag_max_tokens: number;
  /** Chunk ids the user evicted; retrieval drops them before the prompt. */
  rag_excluded_chunks?: string[];
  /** MCP server names whose tool schemas must survive a tight budget. */
  pinned_mcp_servers?: string[];
}

export interface ChatDelta {
  request_id: string;
  delta: string;
  reasoning: string;
}

export interface ChatDone {
  request_id: string;
  finish_reason: string;
  usage: ChatUsage | null;
  elapsed_ms: number;
  first_token_ms: number | null;
  cancelled: boolean;
}

export interface ChatErrorEvent {
  request_id: string;
  message: string;
}

export interface ChatNotice {
  request_id: string;
  message: string;
  /** "error" renders in the banner; "info" is a non-failing progress line. */
  level?: "error" | "info" | string;
}

/** A paused MCP tool call that needs the user's decision before it runs. */
export interface ChatToolApproval {
  request_id: string;
  call_id: string;
  server: string;
  tool: string;
  arguments: string;
  risk: string;
  reason: string;
}

export interface HfFile {
  path: string;
  size: number;
}

export interface HfQuant {
  quant: string;
  files: HfFile[];
  size_bytes: number;
}

export interface HfMmprojFile {
  path: string;
  size: number;
  quant: string;
  recommended: boolean;
}

export interface HfDraftHint {
  repo_id: string;
  size_label: string;
  reason: string;
}

/** A repository split into main quants, vision projectors and a draft hint. */
export interface HfRepoBundles {
  repo: string;
  quants: HfQuant[];
  available_mmproj: HfMmprojFile[];
  recommended_mmproj: string | null;
  draft_hint: HfDraftHint | null;
}

export interface HfModel {
  id: string;
  author: string;
  name: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  tags: string[];
  parameters: number | null;
  last_modified: string | null;
  quants: HfQuant[];
}

export interface SearchRequest {
  query: string;
  gguf_only: boolean;
  limit: number;
}

export interface DownloadFile {
  path: string;
  size: number;
}

export interface DownloadRequest {
  repo: string;
  files: DownloadFile[];
  /** Build a .jns copy with the Janas converter when the download finishes. */
  auto_compile?: boolean;
}

export interface DownloadStarted {
  id: string;
}

export type DownloadStatus =
  | "preparing"
  | "downloading"
  | "paused"
  | "complete"
  | "cancelled"
  | "error";

export interface DownloadProgress {
  id: string;
  repo: string;
  file: string;
  downloaded: number;
  total: number;
  percent: number | null;
  files_done: number;
  files_total: number;
  status: DownloadStatus;
  message: string;
  saved_path: string | null;
}

export interface ConvertRequest {
  input: string;
  output: string;
  overwrite: boolean;
}

export interface ConversionStarted {
  run_id: string;
  command: string;
  converter_path: string;
  output_path: string;
}

export interface ConverterLog {
  run_id: string;
  stream: "stdout" | "stderr" | "system" | string;
  line: string;
  timestamp_ms: number;
}

export interface ConverterProgress {
  run_id: string;
  percent: number | null;
}

export interface ConversionFinished {
  run_id: string;
  success: boolean;
  code: number | null;
  elapsed_ms: number;
  output_path: string;
  output_size: number;
  message: string;
}

/** Everything the Engine Settings panel sends with a chat request, persisted
 * locally. Server-launch options live in AppSettings instead. */
export interface EnginePreferences {
  temperature: number;
  topP: number;
  /** llama.cpp / Janas top-k cut-off, sent with every request. */
  topK: number;
  minP: number;
  repeatPenalty: number;
  maxTokens: number | null;
  thinking: boolean;
  structuredOutput: boolean;
  /** How many previous turns are re-sent as context. */
  contextPrefetch: number;
  eviction: "lru" | "lfu";
  internet: boolean;
}

export interface WebPageResult {
  title: string;
  url: string;
  snippet: string;
  text: string;
  bytes: number;
}

export interface WebSearchResponse {
  query: string;
  provider: string;
  results: WebPageResult[];
  elapsed_ms: number;
  bytes: number;
  message: string;
}

export interface DeviceMetrics {
  present: boolean;
  name: string;
  utilization: number | null;
  memory_used: number | null;
  memory_total: number | null;
}

export interface Telemetry {
  pid: number | null;
  process_rss_bytes: number;
  process_cpu_percent: number;
  read_bytes_per_sec: number;
  read_ops_per_sec: number;
  model_bytes: number;
  model_name: string;
  resident_ratio: number;
  cores: number;
  mem_total_bytes: number;
  mem_available_bytes: number;
  gpu: DeviceMetrics;
  npu: DeviceMetrics;
  sampled_at_ms: number;
}

export interface StorageBenchmark {
  path: string;
  file_bytes: number;
  sequential_mb_s: number;
  random_read_iops: number;
  random_read_mb_s: number;
  average_latency_us: number;
  elapsed_ms: number;
  advice: string;
}

export interface ProjectIngest {
  root: string;
  name: string;
  files: number;
  bytes: number;
  tokens: number;
  tree: string;
  bundle: string;
  truncated: boolean;
}

export type GpuBackend = "cuda" | "rocm" | "vulkan" | "metal" | "none";

/** The machine the fit estimator measured, cached by the backend. */
export interface HardwareProfile {
  os: string;
  arch: string;
  cpu: string;
  cpu_avx2: boolean;
  ram_total_bytes: number;
  /** 0 when the platform does not expose free memory. */
  ram_available_bytes: number;
  gpu_backend: GpuBackend;
  gpu_name: string | null;
  vram_total_bytes: number;
  /** 0 when the driver publishes no used-bytes counter. */
  vram_free_bytes: number;
  /** Sequential MB/s from the last `benchmark_storage` run, if any. */
  storage_mb_s: number | null;
  detected_at_ms: number;
}

export type FitTier =
  | "fits_vram"
  | "ram_gpu_split"
  | "requires_ssd_streaming"
  | "tight_or_slow";

export interface ModelFitEstimate {
  tier: FitTier;
  /** Short badge text, e.g. "Fits VRAM". */
  label: string;
  /** The RAM/VRAM arithmetic behind the badge. */
  reason: string;
  file_size_bytes: number;
  kv_cache_bytes: number;
  runtime_overhead_bytes: number;
  total_needed_bytes: number;
  /** Layer count the estimate used, when the header was readable. */
  block_count: number | null;
  recommended_gpu_layers: number;
  recommended_ctx_size: number;
  uses_ssd_streaming: boolean;
}

export interface AutoTuneResult {
  settings: AppSettings;
  notice: string;
  /** True when the model needs janas-server instead of llama.cpp. */
  requires_engine_switch: boolean;
}

export interface LibraryModel {
  name: string;
  path: string;
  format: string;
  size_bytes: number;
  quant: string;
  architecture: string;
  parameters: string;
  publisher: string;
  repo: string | null;
  downloaded_at: string | null;
  thinking: boolean;
  tools: boolean;
  capabilities_source: string;
  duplicate: boolean;
  same_repo_count: number;
  /** VRAM/RAM verdict for this file on this machine. */
  fit: ModelFitEstimate;
}

export type ModelRole = "main_llm" | "vision_main" | "mmproj" | "draft_candidate";

/** Header-only facts about one GGUF file. No tensor data is read. */
export interface GgufCompanionInfo {
  path: string;
  file_name: string;
  role: ModelRole;
  architecture: string | null;
  name: string | null;
  tokenizer_model: string | null;
  vocab_size: number | null;
  context_length: number | null;
  parameter_estimate: number | null;
  byte_size: number;
  quant: string;
  tensor_count: number;
  vision_tensor_count: number;
  vision_markers: string[];
  vision_family: string | null;
}

export interface CompanionCandidate {
  path: string;
  file_name: string;
  architecture: string | null;
  vocab_size: number | null;
  quant: string;
  parameter_estimate: number | null;
  size_bytes: number;
  score: number;
  compatible: boolean;
  reasons: string[];
}

export interface VisionPairing {
  bound_path: string | null;
  recommended: CompanionCandidate | null;
  candidates: CompanionCandidate[];
  missing: boolean;
  message: string;
}

export interface DraftPairing {
  bound_path: string | null;
  recommended: CompanionCandidate | null;
  candidates: CompanionCandidate[];
  paired: boolean;
  message: string;
  combined_bytes: number;
}

export interface CompanionPairing {
  main: GgufCompanionInfo;
  vision: VisionPairing;
  draft: DraftPairing;
  combined_size_bytes: number;
  estimated_runtime_bytes: number;
  estimated_ram_bytes: number;
  estimated_vram_bytes: number;
  repo: string | null;
  active_bundle_id: string | null;
}

export interface LibraryCompanions {
  models: GgufCompanionInfo[];
  pairings: CompanionPairing[];
  warnings: string[];
}

export interface ModelBundle {
  id: string;
  name: string;
  main_model_path: string;
  mmproj_path: string;
  draft_model_path: string;
  draft_max_tokens: number;
  draft_min_tokens: number;
  draft_p_min: number;
  created_at_ms: number;
  updated_at_ms: number;
}

export interface BundleCheck {
  ok: boolean;
  errors: string[];
  warnings: string[];
  main: GgufCompanionInfo | null;
  mmproj: GgufCompanionInfo | null;
  draft: GgufCompanionInfo | null;
  combined_size_bytes: number;
  estimated_runtime_bytes: number;
}

export interface ModelVerification {
  path: string;
  size_bytes: number;
  sha256: string;
  expected: string | null;
  ok: boolean;
  message: string;
}

export interface LibraryDisk {
  models_dir: string;
  total_bytes: number;
  free_bytes: number | null;
}

export interface ChatBackupInfo {
  id: string;
  title: string;
  updated_at: number;
  bytes: number;
}

export interface ChatStoreStats {
  sessions: number;
  session_bytes: number;
  asset_bytes: number;
  backups: number;
  backup_bytes: number;
}

export interface ChatImportSummary {
  imported: number;
  failed: number;
  errors: string[];
}

/** Backend-advertised sampling controls. Missing keys are treated as false. */
export type BackendCapabilities = Record<string, boolean>;

export interface McpServerConfig {
  id: string;
  name: string;
  command: string;
  args: string[];
  enabled: boolean;
  /** One KEY=VALUE per line. */
  env: string;
  cwd: string;
  /** Chat runs this server's tools without an approval card. Default false. */
  autoApprove?: boolean;
  /** Tools the last successful probe exposed. Absent until the server is probed. */
  toolCount?: number;
  /** Estimated tokens for this server's tool schemas, from the last probe. */
  toolSchemaTokens?: number;
}

export interface McpServerInput {
  name: string;
  command: string;
  args: string[];
  env: string;
  cwd: string;
  auto_approve: boolean;
}

export interface McpTool {
  name: string;
  description: string;
}

export interface McpProbe {
  ok: boolean;
  server_name: string;
  server_version: string;
  tools: McpTool[];
  /** Estimated tokens these tool schemas add to every request. */
  tool_schema_tokens: number;
  log: string[];
  message: string;
  elapsed_ms: number;
}

export interface McpActivityEvent {
  server: string;
  tool: string;
  level: "info" | "error" | string;
  message: string;
}

// ---------------------------------------------------------------------------
// Agent mode
// ---------------------------------------------------------------------------

export type AgentApprovalPolicy = "destructive" | "all" | "auto";
export type AgentRisk =
  | "read_only"
  | "mutating"
  | "destructive"
  | "irreversible";
export type AgentRunStatus =
  | "running"
  | "completed"
  | "verification_failed"
  | "max_iterations"
  | "aborted"
  | "error";

export type AgentStepStatus =
  | "running"
  | "awaiting_approval"
  | "succeeded"
  | "failed"
  | "rejected"
  | "blocked";

export interface AgentStep {
  id: string;
  iteration: number;
  thought: string;
  tool: string;
  arguments: string;
  risk: AgentRisk;
  status: AgentStepStatus;
  output: string | null;
  duration_ms: number | null;
  truncated: boolean;
  created_at: number;
}

export interface AgentCompaction {
  id: string;
  dropped_messages: number;
  estimated_tokens: number;
  context_window: number;
  summary: string;
  created_at: number;
}

export interface AgentDiffPreview {
  path: string;
  diff: string;
  added: number;
  removed: number;
  truncated: boolean;
  error?: string | null;
}

export interface AgentPendingApproval {
  step_id: string;
  tool: string;
  arguments: string;
  risk: AgentRisk;
  reason: string;
  /** write_file/edit_file, which can be allowed for the rest of the run. */
  file_edit?: boolean;
  /** True when this shell command can be remembered for the workspace. */
  can_always_allow?: boolean;
  preview?: AgentDiffPreview | null;
  /** Canonical root pinned for this run. */
  workspace_root?: string;
}

export interface AgentChangedFile {
  /** Workspace-relative path. */
  path: string;
  kind: "created" | "modified" | "deleted" | string;
  restored: boolean;
}

export interface AgentTrace {
  run_id: string;
  status: AgentRunStatus;
  iterations: number;
  max_iterations: number;
  plan: string[];
  steps: AgentStep[];
  compactions: AgentCompaction[];
  pending: AgentPendingApproval | null;
  live_text: string;
  live_reasoning: string;
  notice: string | null;
  error: string | null;
  /** True while this app session still has the run's pre-images. */
  undo_available: boolean;
  /** True once this run's file changes have been reverted. */
  undone: boolean;
  undo_error: string | null;
  undo_summary: string | null;
  /** Per-file undo state and attempt history, newest attempt last. */
  undo_files: AgentUndoFile[];
  /** File summary for runs without a workspace checkpoint. */
  changes: AgentChangedFile[];
  /** Canonical workspace root pinned when this run started. */
  workspace_root: string;
  started_at: number;
  finished_at: number | null;
}

/** Persisted per machine, next to the sampling preferences. */
export interface AgentPreferences {
  enabled: boolean;
  maxIterations: number;
  approvalPolicy: AgentApprovalPolicy;
  workspaceRoot: string;
  planningStep: boolean;
  autoCompact: boolean;
  compactionThreshold: number;
  /** Runs after file edits before the agent may finish; empty disables it. */
  verifyCommand: string;
}

export interface AgentRunRequest {
  run_id: string;
  conversation_id: string;
  goal: string;
  history: ChatOutboundMessage[];
  system_prompt: string | null;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  thinking: boolean | null;
  max_tokens: number | null;
  max_iterations: number;
  approval_policy: AgentApprovalPolicy;
  workspace_root: string;
  planning_step: boolean;
  auto_compact: boolean;
  compaction_threshold: number;
  web_search: boolean;
  verify_command: string;
  mcp_servers: McpServerInput[];
  rag_collections: string[];
  rag_top_k: number;
  rag_max_tokens: number;
}

export interface AgentApprovalReply {
  run_id: string;
  step_id: string;
  decision: "approve" | "reject" | "abort";
  feedback: string;
  /** Only meaningful with "approve"; never applies to shell or MCP tools. */
  allow_file_edits?: boolean;
  /** Remember this shell command for the current workspace. */
  always_allow?: boolean;
}

export interface AgentPlanEvent {
  run_id: string;
  conversation_id: string;
  plan: string[];
}

export interface AgentThoughtEvent {
  run_id: string;
  conversation_id: string;
  iteration: number;
  delta: string;
  reasoning: string;
}

export interface AgentToolStartEvent {
  run_id: string;
  conversation_id: string;
  step_id: string;
  iteration: number;
  tool: string;
  arguments: string;
  risk: AgentRisk;
  requires_approval: boolean;
}

export interface AgentApprovalEvent {
  run_id: string;
  conversation_id: string;
  step_id: string;
  tool: string;
  arguments: string;
  risk: AgentRisk;
  reason: string;
  file_edit: boolean;
  can_always_allow?: boolean;
  preview?: AgentDiffPreview | null;
  workspace_root?: string | null;
}

/** A saved system prompt that can be applied to a chat. */
export interface Persona {
  id: string;
  name: string;
  prompt: string;
}

export type AgentUndoState =
  | "pending"
  | "restored"
  | "skipped"
  | "failed"
  | "kept";

export interface AgentUndoAttempt {
  action: "restore" | "keep";
  state: AgentUndoState;
  reason: string | null;
  at: number;
}

export interface AgentUndoFile {
  path: string;
  /** `created`, `modified` or `deleted`, for the run-change summary. */
  kind: "created" | "modified" | "deleted" | string;
  state: AgentUndoState;
  reason: string | null;
  attempts: number;
  history: AgentUndoAttempt[];
}

export interface AgentUndoResult {
  restored: number;
  deleted: number;
  files: AgentUndoFile[];
  pending: number;
  complete: boolean;
  incomplete: boolean;
}

export interface AgentToolResultEvent {
  run_id: string;
  conversation_id: string;
  step_id: string;
  tool: string;
  ok: boolean;
  output: string;
  duration_ms: number;
  truncated: boolean;
  blocked: boolean;
  rejected: boolean;
}

export interface AgentCompactEvent {
  run_id: string;
  conversation_id: string;
  dropped_messages: number;
  estimated_tokens: number;
  context_window: number;
  summary: string;
}

export interface AgentDoneEvent {
  run_id: string;
  conversation_id: string;
  status: AgentRunStatus;
  iterations: number;
  elapsed_ms: number;
  final_content: string;
  changed_files: number;
  /** Present for runs without a workspace checkpoint. */
  changes: AgentChangedFile[];
  workspace_root?: string | null;
  /** Chunks returned by `search_knowledge_base` during the run. */
  rag_sources?: RagChunk[];
}

export interface AgentErrorEvent {
  run_id: string;
  conversation_id: string;
  message: string;
}

export interface AgentNoticeEvent {
  run_id: string;
  conversation_id: string;
  message: string;
}
