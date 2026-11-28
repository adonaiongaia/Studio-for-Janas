#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent;
mod artifact_frame;
mod benchmark_runner;
mod chat_store;
mod companions;
mod converter;
mod capabilities;
mod engine;
mod engines;
#[cfg(test)]
mod dual_engine_harness;
mod error;
mod fit;
mod gateway;
mod gguf;
mod huggingface;
mod linux_media;
mod logging;
mod mcp;
mod memory;
mod native_audio;
mod provision;
mod rag;
mod settings;
mod system;
mod util;
mod whisper;

use std::sync::Arc;

use tauri::{AppHandle, Emitter, Manager, State};

use benchmark_runner::BenchmarkRunner;
use chat_store::{
    ChatBackupInfo, ChatImportSummary, ChatStore, ChatStoreStats, StoredSession,
};
use converter::{ConversionStarted, ConvertRequest, ConverterManager};
use memory::{MemoryFact, MemoryStore};
use provision::Provisioner;
use rag::{
    KnowledgeCollection, KnowledgeCollectionDetail, KnowledgeStore, RagIndexSummary, RagQueryResult,
};
use engine::{ChatRequest, EngineManager, EngineSnapshot, EngineStatus};
use engines::binary::BinaryManager;
use engines::{ActiveEngineInfo, EngineKind, EngineSupervisor};
use error::{AppError, AppResult};
use huggingface::{
    DownloadManager, DownloadRequest, DownloadStarted, HfModel, HfQuant, SearchRequest,
};
use settings::{AppSettings, BinaryProbe, LocalModel, SettingsStore};

#[tauri::command]
fn get_settings(store: State<'_, Arc<SettingsStore>>) -> AppSettings {
    store.get()
}

#[tauri::command]
fn save_settings(
    store: State<'_, Arc<SettingsStore>>,
    next: AppSettings,
) -> AppResult<AppSettings> {
    store.save(next)
}

#[tauri::command(rename_all = "snake_case")]
fn probe_binary(path: String) -> BinaryProbe {
    settings::probe_binary(&path)
}

/// Walking the models folder touches disk, so it runs on the blocking pool.
#[tauri::command]
async fn scan_local_models(app: AppHandle) -> Vec<LocalModel> {
    let store = app.state::<Arc<SettingsStore>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let current = store.get();
        settings::scan_models(&current.models_dir, Some(&current.model_path), 4)
    })
    .await
    .unwrap_or_default()
}

#[tauri::command]
fn inspect_gguf_model(path: String) -> AppResult<gguf::GgufModelInfo> {
    gguf::inspect(&path)
}

#[tauri::command]
async fn get_engine_status(
    supervisor: State<'_, Arc<EngineSupervisor>>,
) -> AppResult<EngineSnapshot> {
    Ok(supervisor.inner().clone().snapshot().await)
}

/// Start the engine the user last selected. The supervisor owns the choice, so
/// the sidebar's single Start/Stop button works for both engines.
#[tauri::command]
async fn start_engine(
    app: AppHandle,
    supervisor: State<'_, Arc<EngineSupervisor>>,
    store: State<'_, Arc<SettingsStore>>,
) -> AppResult<EngineSnapshot> {
    let supervisor = supervisor.inner().clone();
    let settings = store.get();
    let target = supervisor.preferred_kind();
    supervisor.switch(&app, &settings, Some(target)).await
}

#[tauri::command]
async fn stop_engine(
    app: AppHandle,
    supervisor: State<'_, Arc<EngineSupervisor>>,
    store: State<'_, Arc<SettingsStore>>,
) -> AppResult<()> {
    let supervisor = supervisor.inner().clone();
    let settings = store.get();
    supervisor
        .switch(&app, &settings, None)
        .await
        .map(|_| ())
}

#[tauri::command]
async fn switch_engine(
    app: AppHandle,
    supervisor: State<'_, Arc<EngineSupervisor>>,
    store: State<'_, Arc<SettingsStore>>,
    engine: String,
) -> AppResult<EngineSnapshot> {
    let target = match engine.trim().to_ascii_lowercase().as_str() {
        "janas" | "janas-server" => Some(EngineKind::Janas),
        "llama" | "llama.cpp" | "llamacpp" => Some(EngineKind::Llama),
        "none" | "stop" => None,
        other => {
            return Err(AppError::message(format!(
                "Unknown engine \"{other}\". Use \"janas\", \"llama\", or \"none\"."
            )))
        }
    };
    let supervisor = supervisor.inner().clone();
    let settings = store.get();
    supervisor.switch(&app, &settings, target).await
}

#[tauri::command]
async fn get_active_engine(
    supervisor: State<'_, Arc<EngineSupervisor>>,
) -> AppResult<Option<ActiveEngineInfo>> {
    Ok(supervisor.inner().clone().active_info().await)
}

#[tauri::command]
async fn stream_chat(
    app: AppHandle,
    engine: State<'_, Arc<EngineManager>>,
    supervisor: State<'_, Arc<EngineSupervisor>>,
    store: State<'_, Arc<SettingsStore>>,
    memory: State<'_, Arc<MemoryStore>>,
    knowledge: State<'_, Arc<KnowledgeStore>>,
    request: ChatRequest,
) -> AppResult<()> {
    let engine = engine.inner().clone();
    let memory = memory.inner().clone();
    let knowledge = knowledge.inner().clone();
    let active = supervisor
        .active_info()
        .await
        .ok_or(AppError::EngineNotReady)?;
    let status = supervisor.snapshot().await;
    if status.status != EngineStatus::Ready {
        return Err(AppError::EngineNotReady);
    }

    // The existing SSE proxy is provider-neutral. Point it at whichever engine
    // the supervisor has active instead of always using Janas's settings port.
    let mut config = store.get();
    config.host = active.host;
    config.port = active.port;
    // The proxy also uses this value to decide whether MCP tool schemas fit.
    // Give it the running child's window, not a stale saved preference.
    config.ctx_size = active.context_window;
    config.llama_ctx_size = active.context_window;
    config.model_path = active.model;
    engine
        .stream_chat(&app, &config, &memory, knowledge, request)
        .await
}

// ---------------------------------------------------------------------------
// Local document vault
// ---------------------------------------------------------------------------

#[tauri::command]
fn rag_list_collections(store: State<'_, Arc<KnowledgeStore>>) -> Vec<KnowledgeCollection> {
    store.list()
}

#[tauri::command]
fn rag_collection_detail(
    store: State<'_, Arc<KnowledgeStore>>,
    id: String,
) -> AppResult<KnowledgeCollectionDetail> {
    store.detail(&id)
}

#[tauri::command(rename_all = "snake_case")]
fn rag_create_collection(
    store: State<'_, Arc<KnowledgeStore>>,
    name: String,
    description: String,
    source_paths: Vec<String>,
) -> AppResult<KnowledgeCollection> {
    store.create(&name, &description, &source_paths)
}

#[tauri::command(rename_all = "snake_case")]
fn rag_delete_collection(
    store: State<'_, Arc<KnowledgeStore>>,
    id: String,
) -> AppResult<bool> {
    store.delete(&id)
}

/// Indexing reads and chunks files, so it runs on the blocking pool and streams
/// `rag://index-progress` back to the vault view. Cancelling sets a flag the
/// worker checks between files; the index keeps every file it finished.
#[tauri::command(rename_all = "snake_case")]
async fn rag_index_paths(
    app: AppHandle,
    store: State<'_, Arc<KnowledgeStore>>,
    collection_id: String,
    paths: Vec<String>,
) -> AppResult<RagIndexSummary> {
    let store = store.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        store.index_paths(Some(&app), &collection_id, &paths)
    })
    .await
    .map_err(|err| AppError::message(format!("Indexing stopped: {err}")))?
}

#[tauri::command(rename_all = "snake_case")]
async fn rag_reindex_collection(
    app: AppHandle,
    store: State<'_, Arc<KnowledgeStore>>,
    collection_id: String,
) -> AppResult<RagIndexSummary> {
    let store = store.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        // An empty path list re-walks the collection's source paths; unchanged
        // files are skipped by mtime and SHA-256.
        store.index_paths(Some(&app), &collection_id, &[])
    })
    .await
    .map_err(|err| AppError::message(format!("Re-indexing stopped: {err}")))?
}

#[tauri::command(rename_all = "snake_case")]
fn rag_cancel_index(store: State<'_, Arc<KnowledgeStore>>, collection_id: String) -> bool {
    store.cancel(&collection_id)
}

#[tauri::command(rename_all = "snake_case")]
async fn rag_query(
    store: State<'_, Arc<KnowledgeStore>>,
    collection_ids: Vec<String>,
    query: String,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
) -> AppResult<RagQueryResult> {
    let store = store.inner().clone();
    let top_k = top_k.unwrap_or(rag::DEFAULT_TOP_K);
    let max_tokens = max_tokens.unwrap_or(rag::DEFAULT_MAX_TOKENS);
    tauri::async_runtime::spawn_blocking(move || {
        store.query(&collection_ids, &query, top_k, max_tokens)
    })
    .await
    .map_err(|err| AppError::message(format!("The knowledge search stopped: {err}")))?
}

#[tauri::command(rename_all = "snake_case")]
fn rag_remove_path(
    store: State<'_, Arc<KnowledgeStore>>,
    collection_id: String,
    path: String,
) -> AppResult<KnowledgeCollection> {
    store.remove_path(&collection_id, &path)
}

fn publish_memory(app: &AppHandle, memory: &MemoryStore) {
    let _ = app.emit("memory://changed", memory.list());
}

#[tauri::command]
fn memory_list(memory: State<'_, Arc<MemoryStore>>) -> Vec<MemoryFact> {
    memory.list()
}

/// The backend handler for the `save_to_memory_bank` chat tool. The streaming
/// loop calls the same `MemoryStore::save` method directly for model-initiated
/// saves; this command covers the UI and future callers.
#[tauri::command]
fn save_to_memory_bank(
    app: AppHandle,
    memory: State<'_, Arc<MemoryStore>>,
    fact: String,
) -> AppResult<MemoryFact> {
    let outcome = memory.save(&fact, "user")?;
    if outcome.created {
        publish_memory(&app, &memory);
    }
    Ok(outcome.fact)
}

#[tauri::command]
fn memory_update(
    app: AppHandle,
    memory: State<'_, Arc<MemoryStore>>,
    id: String,
    text: String,
) -> AppResult<MemoryFact> {
    let fact = memory.update(&id, &text)?;
    publish_memory(&app, &memory);
    Ok(fact)
}

#[tauri::command]
fn memory_delete(
    app: AppHandle,
    memory: State<'_, Arc<MemoryStore>>,
    id: String,
) -> AppResult<bool> {
    let removed = memory.delete(&id)?;
    if removed {
        publish_memory(&app, &memory);
    }
    Ok(removed)
}

#[tauri::command]
fn memory_clear(app: AppHandle, memory: State<'_, Arc<MemoryStore>>) -> AppResult<()> {
    memory.clear()?;
    publish_memory(&app, &memory);
    Ok(())
}

#[tauri::command(rename_all = "snake_case")]
fn cancel_chat(engine: State<'_, Arc<EngineManager>>, request_id: String) {
    engine.cancel_chat(&request_id);
}

#[tauri::command(rename_all = "snake_case")]
fn chat_tool_approval(
    engine: State<'_, Arc<EngineManager>>,
    request_id: String,
    call_id: String,
    decision: String,
) -> bool {
    engine.resolve_chat_approval(&request_id, &call_id, decision == "approve")
}

/// Chat files can be large, and the store is read on startup and written after
/// every streamed turn. Both run on the blocking pool so serializing a big
/// session never stalls the webview.
#[tauri::command]
async fn chat_sessions_list(app: AppHandle) -> Vec<StoredSession> {
    let store = app.state::<Arc<ChatStore>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || store.list())
        .await
        .unwrap_or_default()
}

#[tauri::command]
async fn chat_session_save(
    store: State<'_, Arc<ChatStore>>,
    session: StoredSession,
) -> AppResult<StoredSession> {
    let store = store.inner().clone();
    tauri::async_runtime::spawn_blocking(move || store.save(session))
        .await
        .map_err(|err| AppError::message(format!("The chat could not be saved: {err}")))?
}

#[tauri::command(rename_all = "snake_case")]
fn chat_session_delete(store: State<'_, Arc<ChatStore>>, id: String) -> AppResult<bool> {
    store.delete(&id)
}

#[tauri::command]
async fn chat_asset_data_url(
    store: State<'_, Arc<ChatStore>>,
    asset: String,
) -> AppResult<String> {
    let store = store.inner().clone();
    tauri::async_runtime::spawn_blocking(move || store.asset_data_url(&asset))
        .await
        .map_err(|err| AppError::message(format!("That image could not be opened: {err}")))?
}

#[tauri::command(rename_all = "snake_case")]
fn artifact_save_file(path: String, contents: String, base64: bool) -> AppResult<String> {
    system::save_artifact_file(&path, &contents, base64)
}

#[tauri::command(rename_all = "snake_case")]
fn artifact_register_frame(
    frames: State<'_, artifact_frame::ArtifactFrames>,
    html: String,
    allow_network: bool,
) -> AppResult<String> {
    frames.register(html, allow_network)
}

#[tauri::command]
fn chat_export_session(
    store: State<'_, Arc<ChatStore>>,
    id: String,
    path: String,
) -> AppResult<()> {
    store.export(&id, &path)
}

#[tauri::command]
fn chat_import_session(store: State<'_, Arc<ChatStore>>, path: String) -> AppResult<StoredSession> {
    store.import(&path)
}

#[tauri::command(rename_all = "snake_case")]
fn chat_branch_session(
    store: State<'_, Arc<ChatStore>>,
    id: String,
    message_id: String,
) -> AppResult<StoredSession> {
    store.branch_session(&id, &message_id)
}

#[tauri::command]
fn chat_export_all(store: State<'_, Arc<ChatStore>>, path: String) -> AppResult<usize> {
    store.export_all(&path)
}

#[tauri::command]
fn chat_import_all(store: State<'_, Arc<ChatStore>>, path: String) -> AppResult<ChatImportSummary> {
    store.import_all(&path)
}

#[tauri::command]
fn chat_backup_list(store: State<'_, Arc<ChatStore>>) -> Vec<ChatBackupInfo> {
    store.backup_list()
}

#[tauri::command]
fn chat_restore_backup(
    store: State<'_, Arc<ChatStore>>,
    id: String,
) -> AppResult<StoredSession> {
    store.restore_backup(&id)
}

#[tauri::command]
fn chat_store_gc(store: State<'_, Arc<ChatStore>>) -> AppResult<u64> {
    store.gc_assets()
}

#[tauri::command]
fn chat_store_stats(store: State<'_, Arc<ChatStore>>) -> ChatStoreStats {
    store.stats()
}

#[tauri::command]
fn start_conversion(
    app: AppHandle,
    manager: State<'_, Arc<ConverterManager>>,
    store: State<'_, Arc<SettingsStore>>,
    request: ConvertRequest,
) -> AppResult<ConversionStarted> {
    manager.start(&app, &store.get(), request)
}

#[tauri::command]
fn cancel_conversion(
    app: AppHandle,
    manager: State<'_, Arc<ConverterManager>>,
) -> AppResult<()> {
    manager.cancel(&app)
}

#[tauri::command]
async fn search_models(
    store: State<'_, Arc<SettingsStore>>,
    request: SearchRequest,
) -> AppResult<Vec<HfModel>> {
    huggingface::search(request, &store.get().hf_token).await
}

#[tauri::command]
async fn list_model_quants(
    store: State<'_, Arc<SettingsStore>>,
    repo: String,
) -> AppResult<Vec<HfQuant>> {
    huggingface::list_quants(repo, &store.get().hf_token).await
}

#[tauri::command]
async fn inspect_repo_bundles(
    store: State<'_, Arc<SettingsStore>>,
    repo: String,
) -> AppResult<huggingface::HfRepoBundles> {
    huggingface::inspect_repo_bundles(repo, &store.get().hf_token).await
}

#[tauri::command]
fn start_download(
    app: AppHandle,
    manager: State<'_, Arc<DownloadManager>>,
    store: State<'_, Arc<SettingsStore>>,
    request: DownloadRequest,
) -> AppResult<DownloadStarted> {
    let manager = manager.inner().clone();
    manager.start(app, &store.get(), request)
}

#[tauri::command(rename_all = "snake_case")]
fn cancel_download(manager: State<'_, Arc<DownloadManager>>, id: String) -> bool {
    manager.cancel(&id)
}

#[tauri::command(rename_all = "snake_case")]
fn pause_download(manager: State<'_, Arc<DownloadManager>>, id: String) -> bool {
    manager.pause(&id)
}

#[tauri::command(rename_all = "snake_case")]
fn resume_download(manager: State<'_, Arc<DownloadManager>>, id: String) -> bool {
    manager.resume(&id)
}

fn main() {
    if let Err(error) = logging::init() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // A dedicated scheme is the only way to run generated inline scripts
        // under a strict app CSP: srcdoc inherits `script-src 'self'`.
        .register_uri_scheme_protocol(artifact_frame::SCHEME, |ctx, request| {
            use tauri::Manager;
            match ctx
                .app_handle()
                .try_state::<artifact_frame::ArtifactFrames>()
            {
                Some(frames) => artifact_frame::respond(frames.inner(), request),
                None => artifact_frame::unavailable(),
            }
        })
        .setup(|app| {
            let config_dir = app.path().app_config_dir()?;
            let data_dir = app.path().app_data_dir()?;
            let local_data_dir = app.path().app_local_data_dir()?;
            // The MikroTitan rename changed the bundle identifier, and Tauri
            // derives every default folder from it. Carry an existing profile
            // over before the stores open, and keep the absolute paths it
            // saved pointing at the moved files.
            let data_dir_migrated = settings::migrate_legacy_dir(&data_dir);
            settings::migrate_legacy_dir(&config_dir);
            settings::migrate_legacy_dir(&local_data_dir);
            let legacy_data_dir = data_dir_migrated
                .then(|| settings::legacy_sibling(&data_dir))
                .flatten();
            if let Ok(bin) = engines::managed_bin_dir(app.handle()) {
                settings::set_managed_dir(bin);
            }

            let store = Arc::new(SettingsStore::load(
                config_dir,
                data_dir.clone(),
                legacy_data_dir,
            )?);
            // WebKitGTK disables getUserMedia and denies permission requests by
            // default. Enable audio-only capture for this trusted app origin so
            // the Composer's microphone works on Linux without a sidecar.
            if let Some(window) = app.get_webview_window("main") {
                linux_media::enable_media_capture(window.as_ref());
            }
            app.manage(artifact_frame::ArtifactFrames::default());
            let port = store.get().port;
            let memory = Arc::new(MemoryStore::load(data_dir.join("memory.json")));
            let chat_store = ChatStore::load(data_dir.join("chat-sessions"));
            // The vault lives beside the chats in the *local* data directory, so
            // a roaming profile or a synced home never copies an index of the
            // user's documents to another machine.
            let knowledge = Arc::new(KnowledgeStore::load(local_data_dir.join("knowledge")));
            // Benchmark suites live beside the chats in the local data dir so
            // a roaming profile never copies hardware measurements between
            // machines with different drives.
            let benchmarks = BenchmarkRunner::load(local_data_dir.join("benchmarks"))?;

            engines::cleanup_stale_processes(app.handle())?;
            let engine = Arc::new(EngineManager::new(port));
            let supervisor = Arc::new(EngineSupervisor::new(engine.clone()));
            let binaries = Arc::new(BinaryManager::new());
            let provisioner = Arc::new(Provisioner::new());

            app.manage(store.clone());
            app.manage(Arc::new(benchmarks));
            app.manage(Arc::new(chat_store));
            app.manage(knowledge.clone());
            app.manage(memory.clone());
            app.manage(engine);
            app.manage(supervisor.clone());
            app.manage(binaries.clone());
            app.manage(provisioner.clone());
            app.manage(Arc::new(ConverterManager::new()));
            app.manage(Arc::new(DownloadManager::new()));
            app.manage(Arc::new(whisper::WhisperManager::new()));
            app.manage(Arc::new(native_audio::NativeAudioCapture::new()));
            app.manage(Arc::new(agent::AgentManager::with_checkpoints(
                data_dir.join("agent-checkpoints"),
            )));
            app.manage(Arc::new(agent::ApprovalRuleStore::load(
                data_dir.join("agent-approvals.json"),
            )));

            // The gateway is a loopback-only HTTP server. It is managed before
            // the async boot so RunEvent::Exit can always stop it, and it is
            // started here when the user left it enabled.
            let gateway_manager = Arc::new(gateway::GatewayManager::new());
            app.manage(gateway_manager.clone());
            let gateway_enabled = store.get().gateway_enabled;
            if gateway_enabled {
                let app_for_gateway = app.handle().clone();
                let gateway_for_boot = gateway_manager.clone();
                let store_for_gateway = store.clone();
                let supervisor_for_gateway = supervisor.clone();
                let memory_for_gateway = memory.clone();
                let knowledge_for_gateway = knowledge.clone();
                tauri::async_runtime::spawn(async move {
                    // A busy port must not abort the studio's boot; the status
                    // event carries the bind error to the Settings card.
                    let _ = gateway_for_boot
                        .start(
                            &app_for_gateway,
                            store_for_gateway,
                            supervisor_for_gateway,
                            memory_for_gateway,
                            knowledge_for_gateway,
                        )
                        .await;
                });
            }

            // Provisioning is intentionally silent and off the UI thread. The
            // report is available immediately via get_engine_setup_status and
            // is replayed over setup://status as the task advances.
            let app_for_setup = app.handle().clone();
            let provisioner_for_setup = provisioner.clone();
            let manager_for_setup = binaries.clone();
            let supervisor_for_setup = supervisor.clone();
            tauri::async_runtime::spawn(async move {
                provisioner_for_setup
                    .run(
                        &app_for_setup,
                        manager_for_setup,
                        supervisor_for_setup,
                        false,
                    )
                    .await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            save_settings,
            probe_binary,
            scan_local_models,
            inspect_gguf_model,
            get_engine_status,
            start_engine,
            stop_engine,
            switch_engine,
            get_active_engine,
            capabilities::get_backend_capabilities,
            stream_chat,
            cancel_chat,
            chat_tool_approval,
            chat_sessions_list,
            chat_session_save,
            chat_session_delete,
            chat_asset_data_url,
            chat_export_session,
            artifact_save_file,
            artifact_register_frame,
            chat_import_session,
            chat_branch_session,
            chat_export_all,
            chat_import_all,
            chat_backup_list,
            chat_restore_backup,
            chat_store_gc,
            chat_store_stats,
            memory_list,
            save_to_memory_bank,
            memory_update,
            memory_delete,
            memory_clear,
            rag_list_collections,
            rag_collection_detail,
            rag_create_collection,
            rag_delete_collection,
            rag_index_paths,
            rag_reindex_collection,
            rag_cancel_index,
            rag_query,
            rag_remove_path,
            start_conversion,
            cancel_conversion,
            search_models,
            list_model_quants,
            inspect_repo_bundles,
            start_download,
            cancel_download,
            pause_download,
            resume_download,
            system::get_telemetry,
            system::benchmark_storage,
            system::ingest_project,
            system::web_search,
            system::open_external,
            system::transcribe_audio,
            system::transcribe_audio_file,
            native_audio::start_native_capture,
            native_audio::stop_native_capture,
            native_audio::cancel_native_capture,
            native_audio::native_capture_level,
            whisper::get_whisper_status,
            whisper::install_whisper_binary,
            whisper::download_whisper_model,
            whisper::cancel_whisper_download,
            benchmark_runner::benchmark_workload_presets,
            benchmark_runner::start_benchmark_suite,
            benchmark_runner::cancel_benchmark,
            benchmark_runner::list_benchmark_suites,
            benchmark_runner::delete_benchmark_suite,
            benchmark_runner::export_benchmark_markdown,
            system::library_models,
            system::library_disk,
            system::verify_local_model,
            system::delete_local_model,
            gateway::start_gateway,
            gateway::stop_gateway,
            gateway::get_gateway_status,
            companions::inspect_gguf_companion,
            companions::inspect_model_companions,
            companions::library_companions,
            companions::validate_model_bundle,
            companions::save_model_bundle,
            companions::delete_model_bundle,
            companions::apply_model_bundle,
            companions::pair_model_companions,
            mcp::mcp_probe,
            engines::binary::download_and_extract_binary,
            engines::binary::rollback_binary,
            engines::binary::probe_managed_binary,
            provision::get_engine_setup_status,
            provision::rerun_engine_setup,
            fit::get_hardware_profile,
            fit::estimate_quant_fit,
            fit::auto_tune_for_model,
            agent::agent_start,
            agent::agent_resolve_approval,
            agent::agent_approval_rules,
            agent::agent_forget_approval_rule,
            agent::agent_clear_approval_rules,
            agent::agent_validate_workspace,
            agent::agent_abort,
            agent::agent_undo,
            agent::agent_undo_file,
            agent::agent_keep_file,
            agent::agent_undo_availability,
            agent::agent_undo_state,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build MikroTitan");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(gateway) = app_handle.try_state::<Arc<gateway::GatewayManager>>() {
                gateway.inner().stop_blocking();
            }
            if let Some(supervisor) = app_handle.try_state::<Arc<EngineSupervisor>>() {
                supervisor.inner().shutdown_blocking(app_handle);
            }
        }
    });
}
