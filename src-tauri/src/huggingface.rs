use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::AsyncWriteExt;

use crate::{
    converter::{ConvertRequest, ConverterManager},
    error::{AppError, AppResult},
    settings::{AppSettings, SettingsStore},
};

const USER_AGENT: &str = "Studio-for-Janas/0.1 (+https://github.com/prabanta-dev/janas)";
const HF_API: &str = "https://huggingface.co/api/models";
const HF_RESOLVE: &str = "https://huggingface.co";

#[derive(Debug, Clone, Serialize)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfQuant {
    pub quant: String,
    pub files: Vec<HfFile>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfMmprojFile {
    pub path: String,
    pub size: u64,
    pub quant: String,
    /// The projector the dialog selects by default (f16, then bf16, then q8_0).
    pub recommended: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfDraftHint {
    pub repo_id: String,
    pub size_label: String,
    pub reason: String,
}

/// A repository split into the parts a smart bundle needs: main quantizations,
/// vision projectors and an optional tiny sibling draft model.
#[derive(Debug, Clone, Serialize)]
pub struct HfRepoBundles {
    pub repo: String,
    pub quants: Vec<HfQuant>,
    pub available_mmproj: Vec<HfMmprojFile>,
    pub recommended_mmproj: Option<String>,
    pub draft_hint: Option<HfDraftHint>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfModel {
    pub id: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub parameters: Option<u64>,
    pub last_modified: Option<String>,
    pub quants: Vec<HfQuant>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub gguf_only: bool,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    24
}

#[derive(Debug, Deserialize)]
struct ApiModel {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    last_modified: Option<String>,
    #[serde(default)]
    safetensors: Option<SafetensorsInfo>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsInfo {
    #[serde(default)]
    total: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct LfsInfo {
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    oid: Option<String>,
}

fn metadata_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Add the optional access token. Callers in this module build only
/// huggingface.co URLs; reqwest removes the header on cross-host redirects.
fn authorized(builder: reqwest::RequestBuilder, token: &str) -> reqwest::RequestBuilder {
    let token = token.trim();
    if token.is_empty() {
        builder
    } else {
        builder.bearer_auth(token)
    }
}

fn access_error(status: reqwest::StatusCode) -> String {
    if status == reqwest::StatusCode::UNAUTHORIZED {
        "Hugging Face rejected the access token (HTTP 401). The token may be missing, invalid or expired; update it in Settings, then retry.".into()
    } else if status == reqwest::StatusCode::FORBIDDEN {
        "Hugging Face denied access (HTTP 403). This repository may be gated or private: accept its licence on huggingface.co, then add or update the access token in Settings and retry.".into()
    } else if status == reqwest::StatusCode::NOT_FOUND {
        format!(
            "Hugging Face returned HTTP {status}. The repository or file may be missing, or private and in need of an access token in Settings."
        )
    } else {
        format!("Hugging Face returned HTTP {status}.")
    }
}

pub async fn search(request: SearchRequest, token: &str) -> AppResult<Vec<HfModel>> {
    let limit = request.limit.clamp(1, 40);
    let mut url = format!("{HF_API}?limit={limit}&sort=downloads&direction=-1&full=true");
    let query = request.query.trim();
    if !query.is_empty() {
        url.push_str("&search=");
        url.push_str(&urlencoding::encode(query));
    }
    if request.gguf_only {
        url.push_str("&filter=gguf");
    }

    let client = metadata_client();
    let response = authorized(client.get(&url), token)
        .send()
        .await
        .map_err(|err| AppError::message(format!("Hugging Face search failed: {err}")))?;

    if !response.status().is_success() {
        return Err(AppError::message(access_error(response.status())));
    }

    let models: Vec<ApiModel> = response
        .json()
        .await
        .map_err(|err| AppError::message(format!("Could not read the Hugging Face response: {err}")))?;

    let mut enriched = futures_util::stream::iter(models.into_iter().map(|model| {
        let client = client.clone();
        let token = token.to_string();
        async move {
            let quants = fetch_quants(&client, &model.id, &token)
                .await
                .unwrap_or_default();
            let author = model
                .author
                .clone()
                .or_else(|| model.id.split('/').next().map(str::to_string))
                .unwrap_or_else(|| "unknown".into());
            let name = model
                .id
                .split('/')
                .nth(1)
                .map(str::to_string)
                .unwrap_or_else(|| model.id.clone());
            HfModel {
                id: model.id,
                author,
                name,
                downloads: model.downloads.unwrap_or(0),
                likes: model.likes.unwrap_or(0),
                pipeline_tag: model.pipeline_tag,
                tags: model.tags,
                parameters: model.safetensors.and_then(|info| info.total),
                last_modified: model.last_modified,
                quants,
            }
        }
    }))
    .buffer_unordered(8)
    .collect::<Vec<HfModel>>()
    .await;

    enriched.sort_by(|a, b| b.downloads.cmp(&a.downloads));
    Ok(enriched)
}

pub async fn list_quants(repo: String, token: &str) -> AppResult<Vec<HfQuant>> {
    validate_repo(&repo)?;
    let client = metadata_client();
    fetch_quants(&client, &repo, token)
        .await
        .map_err(|err| AppError::message(format!("Could not list this model's files: {err}")))
}

pub async fn inspect_repo_bundles(repo: String, token: &str) -> AppResult<HfRepoBundles> {
    validate_repo(&repo)?;
    let client = metadata_client();
    let entries = fetch_tree_from(&client, HF_API, &repo, token)
        .await
        .map_err(|err| AppError::message(format!("Could not inspect this repository: {err}")))?;
    let quants = build_quants(entries.clone());
    let mut available_mmproj = build_mmproj_files(&entries);
    // One recommendation only: the first file after the preference sort.
    let recommended_mmproj = available_mmproj
        .first()
        .map(|projector| projector.path.clone());
    if let Some(first) = available_mmproj.first_mut() {
        first.recommended = true;
    }
    let draft_hint = detect_draft_hint(&client, &repo, token).await;

    Ok(HfRepoBundles {
        repo,
        quants,
        available_mmproj,
        recommended_mmproj,
        draft_hint,
    })
}

fn build_mmproj_files(entries: &[TreeEntry]) -> Vec<HfMmprojFile> {
    let mut projectors: Vec<HfMmprojFile> = entries
        .iter()
        .filter(|entry| {
            entry
                .kind
                .as_deref()
                .map(|kind| kind == "file")
                .unwrap_or(true)
        })
        .filter(|entry| {
            let lower = entry.path.to_ascii_lowercase();
            lower.ends_with(".gguf") && lower.contains("mmproj")
        })
        .map(|entry| {
            let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
            HfMmprojFile {
                path: entry.path.clone(),
                size: entry
                    .lfs
                    .as_ref()
                    .and_then(|lfs| lfs.size)
                    .or(entry.size)
                    .unwrap_or(0),
                quant: detect_quant(name).unwrap_or_else(|| "GGUF".into()),
                recommended: false,
            }
        })
        .collect();
    projectors.sort_by(|a, b| {
        projector_rank(&a.quant)
            .cmp(&projector_rank(&b.quant))
            .then(a.size.cmp(&b.size))
            .then(a.path.to_lowercase().cmp(&b.path.to_lowercase()))
    });
    projectors
}

/// f16 is the reference projector format; bf16 is equivalent in quality on
/// bf16-trained vision towers; q8_0 is a smaller, slightly lossy fallback.
fn projector_rank(quant: &str) -> usize {
    match quant.to_ascii_uppercase().as_str() {
        "F16" => 0,
        "BF16" => 1,
        "Q8_0" => 2,
        "F32" => 3,
        _ => 9,
    }
}

/// Look for a 0.5B / 1B / 1.5B sibling under the same publisher. This is a
/// hint, not a pairing decision: nothing is downloaded or bound from it.
async fn detect_draft_hint(
    client: &reqwest::Client,
    repo: &str,
    token: &str,
) -> Option<HfDraftHint> {
    let (author, name) = repo.split_once('/')?;
    let term = draft_search_term(name)?;
    let url = format!(
        "{HF_API}?author={}&search={}&limit=50&filter=gguf",
        urlencoding::encode(author),
        urlencoding::encode(&term)
    );
    let response = authorized(client.get(url), token)
        .timeout(Duration::from_secs(8))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let models: Vec<ApiModel> = response.json().await.ok()?;
    models
        .iter()
        .filter(|model| !model.id.eq_ignore_ascii_case(repo))
        .find_map(|model| {
            let size_label = tiny_draft_label(&model.id)?;
            let reason = format!(
                "{} also publishes a {size_label} model in this family; a same-tokenizer draft can speed up generation.",
                author
            );
            Some(HfDraftHint {
                repo_id: model.id.clone(),
                size_label,
                reason,
            })
        })
}

fn tiny_draft_label(model_id: &str) -> Option<String> {
    // Keep the dot inside `0.5b`; splitting on it would break the size token.
    for segment in model_id.to_ascii_lowercase().split(['-', '_', '/']) {
        if matches!(segment, "0.5b" | "1b" | "1.5b") {
            return Some(segment.to_uppercase());
        }
    }
    None
}

fn draft_search_term(name: &str) -> Option<String> {
    let trimmed = name.trim_end_matches("-GGUF").trim_end_matches("-gguf");
    let mut kept: Vec<String> = Vec::new();
    for segment in trimmed.split('-') {
        let lower = segment.to_ascii_lowercase();
        if lower.is_empty() {
            continue;
        }
        if matches!(
            lower.as_str(),
            "gguf" | "instruct" | "it" | "chat" | "base" | "text" | "hf"
        ) {
            continue;
        }
        let starts_with_digit = lower
            .chars()
            .next()
            .map(|character| character.is_ascii_digit())
            .unwrap_or(false);
        let parameter_size = starts_with_digit
            && (lower.ends_with('b')
                || lower
                    .chars()
                    .all(|character| character.is_ascii_digit() || character == '.'));
        if parameter_size {
            continue;
        }
        kept.push(segment.to_string());
    }
    if kept.is_empty() {
        None
    } else {
        Some(kept.join("-"))
    }
}

async fn fetch_quants(
    client: &reqwest::Client,
    repo: &str,
    token: &str,
) -> Result<Vec<HfQuant>, String> {
    fetch_quants_from(client, HF_API, repo, token).await
}

async fn fetch_quants_from(
    client: &reqwest::Client,
    api_base: &str,
    repo: &str,
    token: &str,
) -> Result<Vec<HfQuant>, String> {
    Ok(build_quants(
        fetch_tree_from(client, api_base, repo, token).await?,
    ))
}

async fn fetch_tree_from(
    client: &reqwest::Client,
    api_base: &str,
    repo: &str,
    token: &str,
) -> Result<Vec<TreeEntry>, String> {
    let url = format!("{api_base}/{repo}/tree/main?recursive=true&expand=true");
    let response = authorized(client.get(url), token)
        .timeout(Duration::from_secs(8))
        .send()
        .await
        .map_err(|err| format!("Hugging Face request failed: {err}"))?;
    if !response.status().is_success() {
        return Err(access_error(response.status()));
    }
    response
        .json()
        .await
        .map_err(|err| format!("Unexpected Hugging Face response: {err}"))
}

async fn fetch_checksums(
    client: &reqwest::Client,
    repo: &str,
    token: &str,
) -> Result<HashMap<String, String>, String> {
    fetch_checksums_from(client, HF_API, repo, token).await
}

async fn fetch_checksums_from(
    client: &reqwest::Client,
    api_base: &str,
    repo: &str,
    token: &str,
) -> Result<HashMap<String, String>, String> {
    Ok(fetch_tree_from(client, api_base, repo, token)
        .await?
        .into_iter()
        .filter_map(|entry| entry.lfs.and_then(|lfs| lfs.oid).map(|oid| (entry.path, oid)))
        .collect())
}

fn build_quants(entries: Vec<TreeEntry>) -> Vec<HfQuant> {
    let mut groups: BTreeMap<String, HfQuant> = BTreeMap::new();

    for entry in entries {
        if entry
            .kind
            .as_deref()
            .map(|kind| kind != "file")
            .unwrap_or(false)
        {
            continue;
        }
        let lower = entry.path.to_ascii_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        let file_name = lower.rsplit('/').next().unwrap_or(&lower);
        // Vision encoders, draft/MTP blocks and importance matrices are not the
        // chat model itself; keep them out of the download picker.
        if file_name.starts_with("mmproj")
            || file_name.starts_with("mtp-")
            || lower.contains("imatrix")
            || lower.contains("vision")
        {
            continue;
        }
        let quant = detect_quant(&lower).unwrap_or_else(|| "GGUF".into());
        let size = entry
            .lfs
            .and_then(|lfs| lfs.size)
            .or(entry.size)
            .unwrap_or(0);
        let group = groups.entry(quant.clone()).or_insert_with(|| HfQuant {
            quant,
            files: Vec::new(),
            size_bytes: 0,
        });
        group.size_bytes = group.size_bytes.saturating_add(size);
        group.files.push(HfFile {
            path: entry.path,
            size,
        });
    }

    let mut quants: Vec<HfQuant> = groups.into_values().collect();
    for quant in &mut quants {
        quant
            .files
            .sort_by(|a, b| a.path.to_lowercase().cmp(&b.path.to_lowercase()));
    }
    quants.sort_by(|a, b| {
        quant_rank(&a.quant)
            .cmp(&quant_rank(&b.quant))
            .then(a.size_bytes.cmp(&b.size_bytes))
    });
    quants
}

const QUANT_TOKENS: &[&str] = &[
    "UD-Q6_K_XL",
    "UD-Q5_K_XL",
    "UD-Q4_K_XL",
    "UD-Q3_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q4_1",
    "Q4_0",
    "Q5_1",
    "Q3_K_M",
    "Q3_K_S",
    "Q3_K_L",
    "Q2_K",
    "IQ4_XS",
    "IQ4_NL",
    "IQ3_S",
    "MXFP4_MOE",
    "BF16",
    "F16",
];

fn detect_quant(file_name: &str) -> Option<String> {
    let upper = file_name.to_ascii_uppercase();
    QUANT_TOKENS
        .iter()
        .find(|token| upper.contains(**token))
        .map(|token| token.to_string())
}

fn quant_rank(quant: &str) -> usize {
    QUANT_TOKENS
        .iter()
        .position(|token| token.eq_ignore_ascii_case(quant))
        .unwrap_or(99)
}

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadFile {
    pub path: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadRequest {
    pub repo: String,
    pub files: Vec<DownloadFile>,
    /// Build a `.jns` copy with the Janas converter once every file verifies.
    #[serde(default)]
    pub auto_compile: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadStarted {
    pub id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub id: String,
    pub repo: String,
    pub file: String,
    pub downloaded: u64,
    pub total: u64,
    pub percent: Option<f64>,
    pub files_done: usize,
    pub files_total: usize,
    pub status: String,
    pub message: String,
    pub saved_path: Option<String>,
}

#[derive(Default)]
pub struct DownloadManager {
    tasks: Mutex<HashMap<String, Arc<AtomicBool>>>,
    paused: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl DownloadManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(
        self: &Arc<Self>,
        app: AppHandle,
        settings: &AppSettings,
        request: DownloadRequest,
    ) -> AppResult<DownloadStarted> {
        validate_repo(&request.repo)?;
        if request.files.is_empty() {
            return Err(AppError::message("Choose at least one GGUF file to download."));
        }
        if settings.models_dir.trim().is_empty() {
            return Err(AppError::message("Set a models folder in Settings first."));
        }
        for file in &request.files {
            safe_relative(&file.path)?;
        }

        let id = uuid::Uuid::new_v4().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        self.tasks
            .lock()
            .map_err(|_| AppError::message("Download state is locked."))?
            .insert(id.clone(), cancel.clone());
        self.paused
            .lock()
            .map_err(|_| AppError::message("Download state is locked."))?
            .insert(id.clone(), pause.clone());

        let manager = Arc::clone(self);
        let repo = request.repo.clone();
        let files = request.files.clone();
        let destination = PathBuf::from(settings.models_dir.trim());
        let token = settings.hf_token.trim().to_string();
        let auto_compile = request.auto_compile;
        let task_id = id.clone();
        let app_handle = app.clone();

        let manifest_root = destination.clone();
        crate::system::record_download(
            &manifest_root,
            crate::system::DownloadRecord {
                repo: repo.clone(),
                quant: quant_from_files(&files),
                files: files.iter().map(|file| file.path.clone()).collect(),
                saved_at: crate::util::now_ms().to_string(),
                size_bytes: files.iter().map(|file| file.size).sum(),
                checksums: HashMap::new(),
            },
        );

        tauri::async_runtime::spawn(async move {
            run_download(
                app_handle,
                task_id.clone(),
                cancel,
                pause,
                repo,
                files,
                destination,
                token,
                auto_compile,
            )
            .await;
            manager.finish(&task_id);
        });

        Ok(DownloadStarted { id })
    }

    pub fn cancel(&self, id: &str) -> bool {
        let Ok(guard) = self.tasks.lock() else {
            return false;
        };
        if let Some(flag) = guard.get(id) {
            flag.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Pausing keeps the HTTP stream open but stops consuming it, so a resume
    /// continues from the exact byte the disk already holds.
    pub fn pause(&self, id: &str) -> bool {
        self.paused
            .lock()
            .ok()
            .and_then(|guard| guard.get(id).map(|flag| (flag.store(true, Ordering::Relaxed), true).1))
            .unwrap_or(false)
    }

    pub fn resume(&self, id: &str) -> bool {
        self.paused
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .get(id)
                    .map(|flag| (flag.store(false, Ordering::Relaxed), true).1)
            })
            .unwrap_or(false)
    }

    fn finish(&self, id: &str) {
        if let Ok(mut guard) = self.tasks.lock() {
            guard.remove(id);
        }
        if let Ok(mut guard) = self.paused.lock() {
            guard.remove(id);
        }
    }
}

async fn run_download(
    app: AppHandle,
    id: String,
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    repo: String,
    files: Vec<DownloadFile>,
    models_dir: PathBuf,
    token: String,
    auto_compile: bool,
) {
    let files_total = files.len();
    let known_total: Option<u64> = if files.iter().all(|file| file.size > 0) {
        Some(files.iter().map(|file| file.size).sum())
    } else {
        None
    };
    let root = models_dir.join(repo.replace('/', std::path::MAIN_SEPARATOR_STR));
    let client = match reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .build()
    {
        Ok(client) => client,
        Err(_) => reqwest::Client::new(),
    };

    emit_progress(
        &app,
        DownloadProgress {
            id: id.clone(),
            repo: repo.clone(),
            file: String::new(),
            downloaded: 0,
            total: known_total.unwrap_or(0),
            percent: Some(0.0),
            files_done: 0,
            files_total,
            status: "preparing".into(),
            message: format!("Preparing {files_total} file(s)…"),
            saved_path: Some(root.to_string_lossy().to_string()),
        },
    );

    let expected = match fetch_checksums(&client, &repo, &token).await {
        Ok(expected) => expected,
        Err(message) => {
            // Do not download a file we cannot verify. A missing checksum
            // response is especially dangerous on a gated repository, where
            // it often means the token or licence is no longer valid.
            if let Some(file) = files.first() {
                emit_error(&app, &id, &repo, file, files_total, message);
            }
            return;
        }
    };
    let mut checksums: HashMap<String, String> = HashMap::new();
    let mut downloaded_total = 0u64;

    for (index, file) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            emit_progress(
                &app,
                cancelled_payload(&id, &repo, files_total, downloaded_total, known_total, &root),
            );
            return;
        }

        let relative = match safe_relative(&file.path) {
            Ok(path) => path,
            Err(err) => {
                emit_error(
                    &app,
                    &id,
                    &repo,
                    file,
                    files_total,
                    format!("Unsafe file path: {err}"),
                );
                return;
            }
        };
        let destination = root.join(&relative);
        if let Some(parent) = destination.parent() {
            if let Err(err) = tokio::fs::create_dir_all(parent).await {
                emit_error(
                    &app,
                    &id,
                    &repo,
                    file,
                    files_total,
                    format!("Could not create {}: {err}", parent.display()),
                );
                return;
            }
        }

        let url = format!(
            "{HF_RESOLVE}/{}/resolve/main/{}?download=true",
            repo.split('/')
                .map(|segment| urlencoding::encode(segment).to_string())
                .collect::<Vec<_>>()
                .join("/"),
            file.path
                .split('/')
                .map(|segment| urlencoding::encode(segment).to_string())
                .collect::<Vec<_>>()
                .join("/")
        );

        let response = match authorized(client.get(url), &token).send().await {
            Ok(response) => response,
            Err(err) => {
                emit_error(
                    &app,
                    &id,
                    &repo,
                    file,
                    files_total,
                    format!("Download failed: {err}"),
                );
                return;
            }
        };
        if !response.status().is_success() {
            emit_error(
                &app,
                &id,
                &repo,
                file,
                files_total,
                access_error(response.status()),
            );
            return;
        }

        let file_total = response
            .content_length()
            .filter(|length| *length > 0)
            .unwrap_or(file.size);
        let total_for_display = known_total.unwrap_or(downloaded_total + file_total);
        let part = destination.with_extension(format!(
            "{}.part",
            destination
                .extension()
                .map(|ext| ext.to_string_lossy().to_string())
                .unwrap_or_default()
        ));

        let mut handle = match tokio::fs::File::create(&part).await {
            Ok(handle) => handle,
            Err(err) => {
                emit_error(
                    &app,
                    &id,
                    &repo,
                    file,
                    files_total,
                    format!("Could not write {}: {err}", part.display()),
                );
                return;
            }
        };

        let mut stream = response.bytes_stream();
        let mut file_downloaded = 0u64;
        let mut last_emit = Instant::now();
        let mut failed = None;
        let mut hasher = Sha256::new();

        while let Some(chunk) = stream.next().await {
            if pause.load(Ordering::Relaxed) {
                if let Err(err) = handle.flush().await {
                    failed = Some(format!("Could not flush the file: {err}"));
                    break;
                }
                emit_progress(
                    &app,
                    DownloadProgress {
                        id: id.clone(),
                        repo: repo.clone(),
                        file: file.path.clone(),
                        downloaded: downloaded_total,
                        total: total_for_display,
                        percent: Some(if total_for_display > 0 {
                            (downloaded_total as f64 / total_for_display as f64 * 100.0)
                                .clamp(0.0, 100.0)
                        } else {
                            0.0
                        }),
                        files_done: index,
                        files_total,
                        status: "paused".into(),
                        message: format!("Paused at {} - resume any time", file.path),
                        saved_path: Some(root.to_string_lossy().to_string()),
                    },
                );
                while pause.load(Ordering::Relaxed) {
                    if cancel.load(Ordering::Relaxed) {
                        drop(handle);
                        let _ = tokio::fs::remove_file(&part).await;
                        emit_progress(
                            &app,
                            cancelled_payload(
                                &id,
                                &repo,
                                files_total,
                                downloaded_total,
                                known_total,
                                &root,
                            ),
                        );
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(150)).await;
                }
                emit_progress(
                    &app,
                    DownloadProgress {
                        id: id.clone(),
                        repo: repo.clone(),
                        file: file.path.clone(),
                        downloaded: downloaded_total,
                        total: total_for_display,
                        percent: Some(if total_for_display > 0 {
                            (downloaded_total as f64 / total_for_display as f64 * 100.0)
                                .clamp(0.0, 100.0)
                        } else {
                            0.0
                        }),
                        files_done: index,
                        files_total,
                        status: "downloading".into(),
                        message: format!("Resumed {}", file.path),
                        saved_path: Some(root.to_string_lossy().to_string()),
                    },
                );
            }
            if cancel.load(Ordering::Relaxed) {
                drop(handle);
                let _ = tokio::fs::remove_file(&part).await;
                emit_progress(
                    &app,
                    cancelled_payload(
                        &id,
                        &repo,
                        files_total,
                        downloaded_total,
                        known_total,
                        &root,
                    ),
                );
                return;
            }
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    failed = Some(format!("Transfer interrupted: {err}"));
                    break;
                }
            };
            if let Err(err) = handle.write_all(&chunk).await {
                failed = Some(format!("Could not write to disk: {err}"));
                break;
            }
            let length = chunk.len() as u64;
            hasher.update(&chunk);
            file_downloaded = file_downloaded.saturating_add(length);
            downloaded_total = downloaded_total.saturating_add(length);

            if last_emit.elapsed() >= Duration::from_millis(140) {
                last_emit = Instant::now();
                emit_progress(
                    &app,
                    DownloadProgress {
                        id: id.clone(),
                        repo: repo.clone(),
                        file: file.path.clone(),
                        downloaded: downloaded_total,
                        total: total_for_display,
                        percent: Some(if total_for_display > 0 {
                            (downloaded_total as f64 / total_for_display as f64 * 100.0)
                                .clamp(0.0, 100.0)
                        } else {
                            0.0
                        }),
                        files_done: index,
                        files_total,
                        status: "downloading".into(),
                        message: format!("Downloading {}", file.path),
                        saved_path: Some(root.to_string_lossy().to_string()),
                    },
                );
            }
        }

        if let Some(message) = failed {
            drop(handle);
            let _ = tokio::fs::remove_file(&part).await;
            emit_error(&app, &id, &repo, file, files_total, message);
            return;
        }

        if let Err(err) = handle.flush().await {
            drop(handle);
            let _ = tokio::fs::remove_file(&part).await;
            emit_error(
                &app,
                &id,
                &repo,
                file,
                files_total,
                format!("Could not flush the file: {err}"),
            );
            return;
        }
        drop(handle);

        let digest = format!("{:x}", hasher.finalize());
        if let Some(expected_hash) = expected.get(&file.path) {
            if !expected_hash.eq_ignore_ascii_case(&digest) {
                let _ = tokio::fs::remove_file(&part).await;
                emit_error(
                    &app,
                    &id,
                    &repo,
                    file,
                    files_total,
                    "Checksum mismatch. The downloaded file was discarded.".into(),
                );
                return;
            }
        }
        checksums.insert(file.path.clone(), digest);

        if let Err(err) = tokio::fs::rename(&part, &destination).await {
            let _ = tokio::fs::remove_file(&part).await;
            emit_error(
                &app,
                &id,
                &repo,
                file,
                files_total,
                format!("Could not finalize {}: {err}", destination.display()),
            );
            return;
        }

        emit_progress(
            &app,
            DownloadProgress {
                id: id.clone(),
                repo: repo.clone(),
                file: file.path.clone(),
                downloaded: downloaded_total,
                total: known_total.unwrap_or(downloaded_total),
                percent: Some(if let Some(total) = known_total {
                    (downloaded_total as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
                } else {
                    (index + 1) as f64 / files_total as f64 * 100.0
                }),
                files_done: index + 1,
                files_total,
                status: "downloading".into(),
                message: format!("Saved {}", file.path),
                saved_path: Some(root.to_string_lossy().to_string()),
            },
        );
    }

    // The conversion belongs to the job, not the view that started it: the
    // dialog can close and the user can switch tabs while this runs.
    let message = if auto_compile {
        let note = match auto_compile_download(&app, &root, &files) {
            AutoCompile::Started { file } => format!(
                " Compiling {file} to .jns for Janas SSD streaming — progress is in the Compiler tab."
            ),
            AutoCompile::Failed(reason) => format!(" The .jns build could not start: {reason}"),
            AutoCompile::Skipped => " No .gguf file was found to compile.".to_string(),
        };
        format!("Downloaded {files_total} file(s).{note}")
    } else {
        format!("Downloaded {files_total} file(s) to the models folder. Pair them in Library.")
    };

    emit_progress(
        &app,
        DownloadProgress {
            id,
            repo: repo.clone(),
            file: String::new(),
            downloaded: downloaded_total,
            total: known_total.unwrap_or(downloaded_total),
            percent: Some(100.0),
            files_done: files_total,
            files_total,
            status: "complete".into(),
            message,
            saved_path: Some(root.to_string_lossy().to_string()),
        },
    );

    crate::system::record_download(
        &models_dir,
        crate::system::DownloadRecord {
            repo: repo.clone(),
            quant: quant_from_files(&files),
            files: files.iter().map(|file| file.path.clone()).collect(),
            saved_at: crate::util::now_ms().to_string(),
            size_bytes: files.iter().map(|file| file.size).sum(),
            checksums,
        },
    );
}

/// What a finished download does with its `.jns` request.
enum AutoCompile {
    Started { file: String },
    Failed(String),
    /// The job had nothing convertible (for example only a projector).
    Skipped,
}

/// The main weights of a finished download: the first `.gguf` the job fetched.
/// The explorer requests the quantization's file before any projector, so this
/// is the file the converter expects.
fn compile_target(root: &Path, files: &[DownloadFile]) -> Option<(PathBuf, PathBuf)> {
    let file = files
        .iter()
        .find(|file| file.path.to_ascii_lowercase().ends_with(".gguf"))?;
    let input = root.join(safe_relative(&file.path).ok()?);
    let output = input.with_extension("jns");
    Some((input, output))
}

/// Hand a verified download to the Janas converter. Both managers are app
/// state, so this runs from the download task with no UI involved.
fn auto_compile_download(app: &AppHandle, root: &Path, files: &[DownloadFile]) -> AutoCompile {
    let Some((input, output)) = compile_target(root, files) else {
        return AutoCompile::Skipped;
    };
    let (Some(converter), Some(store)) = (
        app.try_state::<Arc<ConverterManager>>(),
        app.try_state::<Arc<SettingsStore>>(),
    ) else {
        return AutoCompile::Failed("the converter is not available".into());
    };
    let file = input
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| input.to_string_lossy().to_string());
    let request = ConvertRequest {
        input: input.to_string_lossy().to_string(),
        output: output.to_string_lossy().to_string(),
        overwrite: false,
    };
    match converter.start(app, &store.get(), request) {
        Ok(_) => AutoCompile::Started { file },
        Err(err) => AutoCompile::Failed(err.to_string()),
    }
}

fn quant_from_files(files: &[DownloadFile]) -> String {
    let haystack = files
        .iter()
        .map(|file| file.path.to_ascii_uppercase())
        .collect::<Vec<_>>()
        .join(" ");
    QUANT_TOKENS
        .iter()
        .find(|token| haystack.contains(**token))
        .map(|token| token.to_string())
        .unwrap_or_else(|| "GGUF".into())
}

fn cancelled_payload(
    id: &str,
    repo: &str,
    files_total: usize,
    downloaded: u64,
    total: Option<u64>,
    root: &Path,
) -> DownloadProgress {
    DownloadProgress {
        id: id.to_string(),
        repo: repo.to_string(),
        file: String::new(),
        downloaded,
        total: total.unwrap_or(downloaded),
        percent: None,
        files_done: 0,
        files_total,
        status: "cancelled".into(),
        message: "Download cancelled.".into(),
        saved_path: Some(root.to_string_lossy().to_string()),
    }
}

fn emit_progress(app: &AppHandle, payload: DownloadProgress) {
    let _ = app.emit("download://progress", payload);
}

fn emit_error(
    app: &AppHandle,
    id: &str,
    repo: &str,
    file: &DownloadFile,
    files_total: usize,
    message: String,
) {
    emit_progress(
        app,
        DownloadProgress {
            id: id.to_string(),
            repo: repo.to_string(),
            file: file.path.clone(),
            downloaded: 0,
            total: file.size,
            percent: None,
            files_done: 0,
            files_total,
            status: "error".into(),
            message,
            saved_path: None,
        },
    );
}

fn validate_repo(repo: &str) -> AppResult<()> {
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || *part == "."
                || *part == ".."
                || !part
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        })
    {
        return Err(AppError::message(
            "A Hugging Face model id must look like \"owner/model\".",
        ));
    }
    Ok(())
}

fn safe_relative(path: &str) -> AppResult<PathBuf> {
    let mut out = PathBuf::new();
    for segment in path.split('/') {
        if segment.is_empty()
            || segment == "."
            || segment == ".."
            || segment.contains('\\')
            || segment.chars().any(|ch| ch == '\0')
        {
            return Err(AppError::message(
                "The file path contains an unsafe segment.",
            ));
        }
        out.push(segment);
    }
    if out.as_os_str().is_empty() {
        return Err(AppError::message("The file path is empty."));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::now_ms;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        time::{Duration, Instant},
    };

    fn entry(path: &str, size: u64) -> TreeEntry {
        TreeEntry {
            kind: Some("file".into()),
            path: path.into(),
            size: Some(size),
            lfs: None,
        }
    }

    #[test]
    fn detects_common_quants_without_matching_ud_prefix() {
        assert_eq!(
            detect_quant("Qwen_Qwen3-4B-Q4_K_M.gguf"),
            Some("Q4_K_M".into())
        );
        assert_eq!(
            detect_quant("qwen3-30b-a3b-ud-q4_k_xl.gguf"),
            Some("UD-Q4_K_XL".into())
        );
        assert_eq!(detect_quant("model.gguf"), None);
    }

    #[test]
    fn groups_split_shards_and_skips_sidecars() {
        let quants = build_quants(vec![
            entry("Q4_K_M/model-00001-of-00002.gguf", 100),
            entry("Q4_K_M/model-00002-of-00002.gguf", 100),
            entry("mmproj-model-f16.gguf", 50),
            entry("Q8_0/model.gguf", 400),
            entry("README.md", 2),
        ]);
        assert_eq!(quants.len(), 2);
        assert_eq!(quants[0].quant, "Q4_K_M");
        assert_eq!(quants[0].files.len(), 2);
        assert_eq!(quants[0].size_bytes, 200);
        assert_eq!(quants[1].quant, "Q8_0");
    }

    #[test]
    fn splits_mmproj_from_main_quants_and_prefers_f16() {
        let entries = vec![
            entry("Qwen2.5-VL-7B-Q4_K_M.gguf", 100),
            entry("Qwen2.5-VL-7B-Q8_0.gguf", 400),
            entry("mmproj-model-q8_0.gguf", 20),
            entry("mmproj-model-f16.gguf", 50),
            entry("mmproj-model-bf16.gguf", 45),
        ];
        let mains = build_quants(entries.clone());
        assert!(
            mains
                .iter()
                .all(|quant| quant.files.iter().all(|file| !file.path.contains("mmproj"))),
            "projectors must stay out of the main quantization list"
        );

        let projectors = build_mmproj_files(&entries);
        assert_eq!(projectors.len(), 3);
        assert_eq!(projectors[0].quant, "F16");
        assert_eq!(projectors[1].quant, "BF16");
        assert_eq!(projectors[2].quant, "Q8_0");
    }

    #[test]
    fn draft_search_term_drops_parameter_and_format_suffixes() {
        assert_eq!(
            draft_search_term("Qwen2.5-7B-Instruct-GGUF").as_deref(),
            Some("Qwen2.5")
        );
        assert_eq!(
            draft_search_term("gemma-3-4b-it-GGUF").as_deref(),
            Some("gemma")
        );
        assert_eq!(
            draft_search_term("Llama-3.2-3B-Instruct-GGUF").as_deref(),
            Some("Llama")
        );
    }

    #[test]
    fn tiny_draft_labels_only_match_whole_size_tokens() {
        assert_eq!(
            tiny_draft_label("Qwen/Qwen2.5-0.5B-Instruct-GGUF").as_deref(),
            Some("0.5B")
        );
        assert_eq!(
            tiny_draft_label("org/model-1.5b-it").as_deref(),
            Some("1.5B")
        );
        assert_eq!(tiny_draft_label("org/model-1B-it").as_deref(), Some("1B"));
        assert_eq!(tiny_draft_label("org/model-7B-it"), None);
        assert_eq!(tiny_draft_label("org/model-21B-it"), None);
    }

    #[test]
    fn rejects_path_traversal() {
        assert!(safe_relative("../../etc/passwd").is_err());
        assert!(safe_relative("sub/file.gguf").is_ok());
        assert_eq!(
            safe_relative("sub/file.gguf").unwrap(),
            Path::new("sub").join("file.gguf")
        );
    }

    #[test]
    fn validates_repo_ids() {
        assert!(validate_repo("Qwen/Qwen3-4B-GGUF").is_ok());
        assert!(validate_repo("not-a-repo").is_err());
        assert!(validate_repo("../evil").is_err());
    }

    #[test]
    fn access_errors_separate_authentication_from_permission() {
        let unauthorized = access_error(reqwest::StatusCode::UNAUTHORIZED);
        assert!(unauthorized.contains("401"), "{unauthorized}");
        assert!(unauthorized.contains("rejected the access token"), "{unauthorized}");
        assert!(unauthorized.contains("invalid or expired"), "{unauthorized}");
        assert!(!unauthorized.contains("accept its licence"), "{unauthorized}");

        let forbidden = access_error(reqwest::StatusCode::FORBIDDEN);
        assert!(forbidden.contains("403"), "{forbidden}");
        assert!(forbidden.contains("gated or private"), "{forbidden}");
        assert!(forbidden.contains("accept its licence"), "{forbidden}");
        assert!(forbidden.contains("access token"), "{forbidden}");

        let missing = access_error(reqwest::StatusCode::NOT_FOUND);
        assert!(missing.contains("404"));
        assert!(missing.contains("access token"));
    }

    fn mock_tree_server(
        status: &'static str,
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        listener
            .set_nonblocking(true)
            .expect("nonblocking mock server");
        let address = listener.local_addr().expect("mock address");
        let handle = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(pair) => break pair,
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "mock server accept timed out");
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(err) => panic!("mock server accept failed: {err}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("mock read timeout");
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                let read = stream.read(&mut buffer).expect("mock request");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .expect("mock response");
            String::from_utf8_lossy(&request).into_owned()
        });
        (format!("http://{address}/api/models"), handle)
    }

    #[tokio::test]
    async fn mock_gated_tree_sends_token_and_names_the_fix() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("mock client");
        for (status, code, expected) in [
            ("401 Unauthorized", "401", "rejected the access token"),
            ("403 Forbidden", "403", "accept its licence"),
        ] {
            let (base, server) = mock_tree_server(status, "{}");
            let error = fetch_quants_from(&client, &base, "org/private", "hf_mock_token")
                .await
                .expect_err("gated tree must fail");
            let request = server.join().expect("mock server result");
            let lower = request.to_ascii_lowercase();
            assert!(
                request.starts_with(
                    "GET /api/models/org/private/tree/main?recursive=true&expand=true HTTP/1.1"
                ),
                "unexpected request target: {request}"
            );
            assert!(
                lower.contains("authorization: bearer hf_mock_token"),
                "token was not sent: {request}"
            );
            assert!(error.contains(code), "{error}");
            assert!(error.contains(expected), "{error}");
            assert!(!error.contains("hf_mock_token"), "token leaked: {error}");
        }
    }

    #[tokio::test]
    async fn mock_gated_tree_without_token_stays_unauthenticated() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("mock client");
        let (base, server) = mock_tree_server("401 Unauthorized", "{}");
        let error = fetch_quants_from(&client, &base, "org/private", "")
            .await
            .expect_err("gated tree must fail");
        let request = server.join().expect("mock server result").to_ascii_lowercase();
        assert!(!request.contains("authorization:"), "{request}");
        assert!(error.contains("401"), "{error}");
        assert!(error.contains("rejected the access token"), "{error}");
    }

    #[tokio::test]
    async fn mock_public_tree_omits_blank_token() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("mock client");
        let (base, server) = mock_tree_server("200 OK", "[]");
        let quants = fetch_quants_from(&client, &base, "org/public", "   ")
            .await
            .expect("public tree");
        assert!(quants.is_empty());
        let request = server.join().expect("mock server result");
        assert!(
            request.starts_with("GET /api/models/org/public/tree/main?recursive=true&expand=true HTTP/1.1"),
            "{request}"
        );
        assert!(
            !request.to_ascii_lowercase().contains("authorization:"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn mock_checksum_lookup_rejects_auth_errors() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("mock client");
        let (base, server) = mock_tree_server("403 Forbidden", "{}");
        let error = fetch_checksums_from(&client, &base, "org/private", "hf_mock_token")
            .await
            .expect_err("checksum lookup must fail");
        let request = server.join().expect("mock server result");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer hf_mock_token"),
            "token was not sent: {request}"
        );
        assert!(error.contains("403"), "{error}");
        assert!(error.contains("gated or private"), "{error}");
        assert!(!error.contains("hf_mock_token"), "token leaked: {error}");
    }

    #[tokio::test]
    async fn mock_checksum_lookup_returns_lfs_oids() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("mock client");
        let body = r#"[{"path":"model.gguf","lfs":{"size":3,"oid":"abc123"}}]"#;
        let (base, server) = mock_tree_server("200 OK", body);
        let checksums = fetch_checksums_from(&client, &base, "org/public", "")
            .await
            .expect("checksum lookup");
        assert_eq!(checksums.get("model.gguf"), Some(&"abc123".to_string()));
        let request = server.join().expect("mock server result").to_ascii_lowercase();
        assert!(!request.contains("authorization:"), "{request}");
    }

    #[test]
    fn timestamps_are_available() {
        assert!(now_ms() > 0);
    }

    #[test]
    fn compiles_the_first_gguf_of_a_finished_download() {
        let root = Path::new("/models/org/repo");
        let files = [
            DownloadFile {
                path: "Qwen-Q4_K_M.gguf".into(),
                size: 1,
            },
            DownloadFile {
                path: "mmproj-F16.gguf".into(),
                size: 1,
            },
        ];
        let (input, output) = compile_target(root, &files).expect("a target");
        assert_eq!(input, root.join("Qwen-Q4_K_M.gguf"));
        assert_eq!(output, root.join("Qwen-Q4_K_M.jns"));
    }

    #[test]
    fn skips_jobs_with_nothing_to_compile() {
        assert!(compile_target(Path::new("/models"), &[]).is_none());
        let readme = [DownloadFile {
            path: "README.md".into(),
            size: 1,
        }];
        assert!(compile_target(Path::new("/models"), &readme).is_none());
        let escape = [DownloadFile {
            path: "../escape.gguf".into(),
            size: 1,
        }];
        assert!(compile_target(Path::new("/models"), &escape).is_none());
    }
}
