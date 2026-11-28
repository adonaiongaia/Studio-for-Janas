//! Hardware-aware model fit estimation.
//!
//! Before a download starts, and before the engine loads a local file, the
//! studio can say whether the weights will live in VRAM, need a RAM/GPU split,
//! or have to stream from the SSD. The estimate only needs numbers the app
//! already has:
//!
//! * RAM from [`crate::system::memory_snapshot`]
//! * VRAM from [`crate::system::vram_snapshot`] (sysfs, so it works with the
//!   NVIDIA proprietary driver, amdgpu and i915 alike)
//! * the GPU backend from [`crate::provision::detect_platform`]
//! * the last 64 MB storage benchmark from
//!   [`crate::system::last_storage_benchmark`]
//! * the GGUF block count and context length from [`crate::gguf`] and
//!   [`crate::companions`]
//!
//! None of the arithmetic reads model weights: the KV-cache estimate uses the
//! transformer block count and a per-token-per-layer byte figure that matches
//! an 8 KV-head, 128-wide, f16 cache (the shape used by most open models).

use std::{
    fs,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tauri::State;

use crate::{
    error::{AppError, AppResult},
    provision,
    settings::{expand_home, AppSettings, SettingsStore},
};

const GIB: u64 = 1024 * 1024 * 1024;
/// Runtime buffers, graph and allocator slack on top of the weights themselves.
const RUNTIME_OVERHEAD_RATIO: f64 = 0.15;
/// VRAM the desktop compositor and driver keep for themselves.
const VRAM_HEADROOM_BYTES: u64 = 3 * GIB / 2;
/// f16 K+V for 8 KV heads x 128 dims x 2 tensors.
const KV_BYTES_PER_TOKEN_PER_LAYER: u64 = 4096;
/// Used when a file's block count cannot be read (HF rows, .jns files).
const DEFAULT_BLOCK_COUNT: u32 = 32;
const DEFAULT_CONTEXT: u32 = 8_192;
const MIN_CONTEXT: u32 = 512;
const MAX_CONTEXT: u32 = 1_048_576;
/// Below this the drive becomes the bottleneck while streaming experts.
const SLOW_STORAGE_MB_S: f64 = 500.0;
const PROFILE_TTL: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    #[default]
    None,
}

/// The machine as the fit estimator sees it. Sizes are bytes so the UI can
/// format them without guessing units.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub cpu_avx2: bool,
    pub ram_total_bytes: u64,
    /// 0 when the platform does not expose free memory.
    pub ram_available_bytes: u64,
    pub gpu_backend: GpuBackend,
    pub gpu_name: Option<String>,
    pub vram_total_bytes: u64,
    /// 0 when the driver does not publish a used-bytes counter.
    pub vram_free_bytes: u64,
    /// Sequential MB/s from the last `benchmark_storage` run.
    pub storage_mb_s: Option<f64>,
    pub detected_at_ms: u64,
}

impl HardwareProfile {
    fn vram_budget(&self) -> u64 {
        if self.vram_free_bytes > 0 {
            self.vram_free_bytes
        } else {
            self.vram_total_bytes
        }
    }

    fn ram_budget(&self) -> u64 {
        if self.ram_total_bytes == 0 {
            // Unknown RAM must not be reported as "does not fit".
            return u64::MAX;
        }
        if self.ram_available_bytes > 0 {
            self.ram_available_bytes.min(self.ram_total_bytes)
        } else {
            self.ram_total_bytes
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitTier {
    /// Weights + KV cache fit inside the detected VRAM.
    FitsVram,
    /// Exceeds VRAM but fits in system RAM: offload what fits, run the rest on
    /// the CPU (`--n-gpu-layers` below the block count).
    RamGpuSplit,
    /// Exceeds system RAM: stream the `.jns` from the SSD with janas-server.
    RequiresSsdStreaming,
    /// Exceeds system RAM and the measured storage is too slow to stream.
    TightOrSlow,
}

impl FitTier {
    pub fn label(self) -> &'static str {
        match self {
            FitTier::FitsVram => "Fits VRAM",
            FitTier::RamGpuSplit => "RAM + GPU Split",
            FitTier::RequiresSsdStreaming => "SSD Stream (.jns)",
            FitTier::TightOrSlow => "Exceeds RAM (Slow Disk)",
        }
    }

    fn needs_ssd(self) -> bool {
        matches!(self, FitTier::RequiresSsdStreaming | FitTier::TightOrSlow)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFitEstimate {
    pub tier: FitTier,
    /// Short badge text, kept alongside the tier so every surface agrees.
    pub label: String,
    /// Human-readable RAM/VRAM arithmetic for the tooltip.
    pub reason: String,
    pub file_size_bytes: u64,
    pub kv_cache_bytes: u64,
    pub runtime_overhead_bytes: u64,
    /// Weights + runtime + KV cache at the requested context.
    pub total_needed_bytes: u64,
    /// The layer count the estimate used (read or assumed).
    pub block_count: Option<u32>,
    pub recommended_gpu_layers: u32,
    pub recommended_ctx_size: u32,
    pub uses_ssd_streaming: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoTuneResult {
    /// Settings after the applied changes (unchanged when nothing applied).
    pub settings: AppSettings,
    pub notice: String,
    /// True when the model needs janas-server instead of llama.cpp.
    pub requires_engine_switch: bool,
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

fn kv_cache_bytes(context_size: u32, block_count: u32) -> u64 {
    (context_size as u64)
        .saturating_mul(block_count as u64)
        .saturating_mul(KV_BYTES_PER_TOKEN_PER_LAYER)
}

fn implied_block_count(parameters: u64) -> u32 {
    let billions = parameters as f64 / 1_000_000_000.0;
    if billions < 2.0 {
        24
    } else if billions < 8.0 {
        32
    } else if billions < 20.0 {
        48
    } else if billions < 45.0 {
        64
    } else if billions < 90.0 {
        80
    } else {
        96
    }
}

/// Classify one file against one machine.
///
/// `block_count` is the GGUF `<architecture>.block_count`; when it is missing
/// the estimate assumes a 32-layer model. `context_size` is the context the
/// user wants to run, not the model's trained maximum.
pub fn estimate_model_fit(
    file_size_bytes: u64,
    block_count: Option<u32>,
    context_size: u32,
    hw: &HardwareProfile,
) -> ModelFitEstimate {
    let block_count = block_count.filter(|value| *value > 0);
    let layers = block_count.unwrap_or(DEFAULT_BLOCK_COUNT);
    let context = context_size.clamp(MIN_CONTEXT, MAX_CONTEXT);
    let kv = kv_cache_bytes(context, layers);
    let runtime = (file_size_bytes as f64 * RUNTIME_OVERHEAD_RATIO).round() as u64;
    let total = file_size_bytes
        .saturating_add(runtime)
        .saturating_add(kv);

    let vram_budget = hw.vram_budget();
    let ram_budget = hw.ram_budget();
    let ram_known = hw.ram_total_bytes > 0;
    // A full offload must hold the KV cache at the requested context in VRAM,
    // but a RAM/GPU split can shrink its context instead of streaming, so that
    // tier is decided by the weights plus a minimal cache. A model whose
    // weights alone exceed total RAM has no such escape.
    let minimal_total = file_size_bytes
        .saturating_add(runtime)
        .saturating_add(kv_cache_bytes(MIN_CONTEXT, layers));
    let ram_capacity = if ram_known { hw.ram_total_bytes } else { u64::MAX };

    let tier = if vram_budget > 0 && total <= vram_budget {
        FitTier::FitsVram
    } else if minimal_total <= ram_capacity {
        FitTier::RamGpuSplit
    } else if hw
        .storage_mb_s
        .map(|throughput| throughput < SLOW_STORAGE_MB_S)
        .unwrap_or(false)
    {
        FitTier::TightOrSlow
    } else {
        FitTier::RequiresSsdStreaming
    };

    // Offloading stops 1.5 GiB before the VRAM ceiling so the compositor keeps
    // its buffers; a partial offload always leaves at least one layer on the
    // CPU, otherwise llama.cpp is being asked for a full offload that did not
    // fit in the first place.
    let offered_budget = vram_budget.saturating_sub(VRAM_HEADROOM_BYTES);
    let recommended_gpu_layers = match tier {
        FitTier::FitsVram => layers,
        FitTier::RamGpuSplit => {
            if vram_budget == 0 {
                0
            } else {
                let ratio = offered_budget as f64 / total.max(1) as f64;
                let proportional = ((ratio * layers as f64).floor() as u32).min(layers);
                if proportional >= layers {
                    layers.saturating_sub(1)
                } else {
                    proportional
                }
            }
        }
        FitTier::RequiresSsdStreaming | FitTier::TightOrSlow => 0,
    };

    let budget_free = match tier {
        FitTier::FitsVram => vram_budget,
        _ => ram_budget,
    };
    let fixed = file_size_bytes.saturating_add(runtime);
    let per_token = (layers as u64).saturating_mul(KV_BYTES_PER_TOKEN_PER_LAYER);
    let available = budget_free.saturating_sub(fixed);
    let ceiling = if per_token == 0 {
        context
    } else {
        (available / per_token).min(u64::from(MAX_CONTEXT)) as u32
    };
    let recommended_ctx_size = ((ceiling.min(context)) / 512 * 512).clamp(MIN_CONTEXT, MAX_CONTEXT);

    let mut estimate = ModelFitEstimate {
        tier,
        label: tier.label().to_string(),
        reason: String::new(),
        file_size_bytes,
        kv_cache_bytes: kv,
        runtime_overhead_bytes: runtime,
        total_needed_bytes: total,
        block_count,
        recommended_gpu_layers,
        recommended_ctx_size,
        uses_ssd_streaming: tier.needs_ssd(),
    };
    estimate.reason = fit_reason(&estimate, hw, context);
    estimate
}

/// The sentence shown under a badge. Reads the finished estimate so the
/// arithmetic and the numbers in the sentence cannot drift apart.
fn fit_reason(estimate: &ModelFitEstimate, hw: &HardwareProfile, requested_ctx: u32) -> String {
    let layers = estimate.block_count.unwrap_or(DEFAULT_BLOCK_COUNT);
    let capped = if estimate.recommended_ctx_size < requested_ctx {
        format!(
            " Context capped at {} tokens to keep the cache in memory.",
            estimate.recommended_ctx_size
        )
    } else {
        String::new()
    };
    let weights = estimate.file_size_bytes;
    let kv = estimate.kv_cache_bytes;
    let total = estimate.total_needed_bytes;
    let gpu_layers = estimate.recommended_gpu_layers;
    match estimate.tier {
        FitTier::FitsVram => format!(
            "{} model + {} KV cache fits in {} VRAM ({} free, {}).",
            gb(weights),
            gb(kv),
            gb(hw.vram_total_bytes),
            gb(hw.vram_budget()),
            backend_label(hw.gpu_backend),
        ),
        FitTier::RamGpuSplit if hw.ram_total_bytes == 0 => format!(
            "{} model + {} KV cache does not fit in detected VRAM; system RAM could not be measured, so plan a GPU/CPU split.{capped}",
            gb(weights),
            gb(kv),
        ),
        FitTier::RamGpuSplit => format!(
            "{} model + {} KV cache ({}) exceeds {} VRAM; {gpu_layers} of {layers} layers fit leaving 1.5 GB headroom, the rest runs from {} system RAM.{capped}",
            gb(weights),
            gb(kv),
            gb(total),
            gb(hw.vram_budget()),
            gb(hw.ram_total_bytes),
        ),
        FitTier::RequiresSsdStreaming => format!(
            "{} working set exceeds {} system RAM; stream the .jns from the SSD with janas-server.",
            gb(total),
            gb(hw.ram_total_bytes),
        ),
        FitTier::TightOrSlow => format!(
            "{} working set exceeds {} system RAM and the last storage benchmark was {} MB/s (below {:.0} MB/s): SSD streaming will be I/O-bound.",
            gb(total),
            gb(hw.ram_total_bytes),
            hw.storage_mb_s.unwrap_or(0.0).round() as u64,
            SLOW_STORAGE_MB_S,
        ),
    }
}

fn backend_label(backend: GpuBackend) -> &'static str {
    match backend {
        GpuBackend::Cuda => "CUDA",
        GpuBackend::Rocm => "ROCm",
        GpuBackend::Vulkan => "Vulkan",
        GpuBackend::Metal => "Metal",
        GpuBackend::None => "CPU only",
    }
}

/// Human-readable memory size, using the same binary units as `formatBytes`
/// in the UI so a tooltip and the "Detected" line never disagree. Below a
/// gigabyte the decimal form would print an iGPU carve-out as "0.0 GB", so
/// small budgets stay in megabytes.
fn gb(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    if (bytes as f64) < GIB {
        format!("{:.0} MB", bytes as f64 / MIB)
    } else {
        format!("{:.1} GB", bytes as f64 / GIB)
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

static PROFILE_CACHE: OnceLock<Mutex<Option<(Instant, HardwareProfile)>>> = OnceLock::new();

fn profile_cache() -> &'static Mutex<Option<(Instant, HardwareProfile)>> {
    PROFILE_CACHE.get_or_init(|| Mutex::new(None))
}

/// The host's capabilities, cached for a minute because detection shells out to
/// `nvidia-smi`/`vulkaninfo`. The storage figure is re-read on every call so a
/// benchmark run is visible immediately.
pub fn hardware_profile() -> HardwareProfile {
    let cached = profile_cache()
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
        .filter(|(at, _)| at.elapsed() < PROFILE_TTL)
        .map(|(_, profile)| profile);
    let mut profile = match cached {
        Some(profile) => profile,
        None => {
            let profile = detect_hardware();
            if let Ok(mut guard) = profile_cache().lock() {
                *guard = Some((Instant::now(), profile.clone()));
            }
            profile
        }
    };
    profile.storage_mb_s = storage_snapshot();
    profile
}

fn detect_hardware() -> HardwareProfile {
    let platform = provision::detect_platform();
    let summary = crate::system::hardware_summary();
    let (ram_total, ram_available) = crate::system::memory_snapshot();
    let backend = gpu_backend_from(&platform.preferred_backend);
    let (vram_total, vram_free) = if backend == GpuBackend::Metal {
        (ram_total, ram_available)
    } else {
        // Hybrid laptops list an iGPU carve-out next to the discrete card; the
        // detected backend decides which one the estimate is allowed to use.
        let preference = match backend {
            GpuBackend::Cuda => crate::system::VramPreference::Cuda,
            GpuBackend::Rocm => crate::system::VramPreference::Rocm,
            _ => crate::system::VramPreference::Any,
        };
        crate::system::vram_snapshot(preference)
    };
    let ram_total = ram_total.max(summary.mem_total_bytes);
    let cpu = if summary.cpu.trim().is_empty() || summary.cpu == "Unknown CPU" {
        format!("{} {}", platform.arch, platform.os)
    } else {
        summary.cpu
    };

    HardwareProfile {
        os: summary.os,
        arch: summary.arch,
        cpu,
        cpu_avx2: platform.cpu_avx2,
        ram_total_bytes: ram_total,
        ram_available_bytes: ram_available.min(ram_total),
        gpu_backend: backend,
        gpu_name: platform.gpus.first().map(|gpu| gpu.name.clone()),
        vram_total_bytes: vram_total,
        vram_free_bytes: vram_free.min(vram_total),
        storage_mb_s: None,
        detected_at_ms: crate::util::now_ms(),
    }
}

fn gpu_backend_from(preferred: &str) -> GpuBackend {
    match preferred.trim().to_ascii_lowercase().as_str() {
        "cuda" => GpuBackend::Cuda,
        "rocm" => GpuBackend::Rocm,
        "vulkan" => GpuBackend::Vulkan,
        "metal" => GpuBackend::Metal,
        _ => GpuBackend::None,
    }
}

fn storage_snapshot() -> Option<f64> {
    crate::system::last_storage_benchmark()
        .map(|benchmark| benchmark.sequential_mb_s)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_hardware_profile() -> HardwareProfile {
    tauri::async_runtime::spawn_blocking(hardware_profile)
        .await
        .unwrap_or_default()
}

/// Fit one quantization before anything is downloaded. `parameters` fills in a
/// block count when the Hub metadata does not include a GGUF header.
#[tauri::command(rename_all = "snake_case")]
pub async fn estimate_quant_fit(
    file_size_bytes: u64,
    block_count: Option<u32>,
    context_size: Option<u32>,
    parameters: Option<u64>,
) -> ModelFitEstimate {
    let context = context_size.unwrap_or(DEFAULT_CONTEXT);
    let layers = block_count
        .filter(|value| *value > 0)
        .or_else(|| parameters.map(implied_block_count));
    tauri::async_runtime::spawn_blocking(move || {
        let profile = hardware_profile();
        estimate_model_fit(file_size_bytes, layers, context, &profile)
    })
    .await
    .unwrap_or_else(|_| {
        estimate_model_fit(
            file_size_bytes,
            layers,
            context,
            &HardwareProfile::default(),
        )
    })
}

/// Apply the calculated llama.cpp (or Janas) settings for one model.
#[tauri::command(rename_all = "snake_case")]
pub async fn auto_tune_for_model(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    model_path: Option<String>,
) -> AppResult<AutoTuneResult> {
    let store = store.inner().clone();
    let settings = store.get();
    let requested = model_path.unwrap_or_default();
    let target = if !requested.trim().is_empty() {
        requested
    } else if !settings.llama_model_path.trim().is_empty() {
        settings.llama_model_path.clone()
    } else {
        settings.model_path.clone()
    };
    let path = expand_home(target.trim());
    if target.trim().is_empty() {
        return Err(AppError::message(
            "Select a model first: Auto-Tune needs an active .gguf or .jns file.",
        ));
    }
    if !path.is_file() {
        return Err(AppError::message(format!(
            "Auto-Tune could not read that model because {} does not exist.",
            path.display()
        )));
    }

    let is_gguf = path
        .extension()
        .map(|extension| extension.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    let size_bytes = fs::metadata(&path)
        .map_err(|err| AppError::message(format!("Could not read that model: {err}")))?
        .len();

    let display = path.to_string_lossy().to_string();
    let (layer_count, trained_context) = if is_gguf {
        let layers = crate::gguf::layer_count(&display);
        let trained = crate::companions::inspect_gguf_metadata(&display)
            .ok()
            .and_then(|info| info.context_length)
            .filter(|value| *value > 0)
            .map(|value| value.min(u64::from(MAX_CONTEXT)) as u32);
        (layers, trained)
    } else {
        (None, None)
    };

    let estimate = tauri::async_runtime::spawn_blocking(move || {
        let requested_ctx = if is_gguf {
            settings.llama_ctx_size
        } else {
            settings.ctx_size
        };
        let context = trained_context
            .map(|trained| requested_ctx.min(trained))
            .unwrap_or(requested_ctx);
        let profile = hardware_profile();
        estimate_model_fit(size_bytes, layer_count, context, &profile)
    })
    .await
    .map_err(|err| AppError::message(format!("The Auto-Tune task failed: {err}")))?;

    let mut next = store.get();
    let mut changed: Vec<String> = Vec::new();

    // A GGUF above RAM has nothing to tune: llama.cpp cannot load it, and a
    // `.jns` build is the only path that can stream it.
    if estimate.uses_ssd_streaming && is_gguf {
        // Left alone on purpose; the notice explains the compile step.
    } else if is_gguf {
        let layers = estimate.recommended_gpu_layers as i32;
        if next.llama_gpu_layers != layers {
            next.llama_gpu_layers = layers;
            changed.push(format!("--n-gpu-layers → {layers}"));
        }
        if next.llama_ctx_size != estimate.recommended_ctx_size {
            next.llama_ctx_size = estimate.recommended_ctx_size;
            changed.push(format!("context size → {}", estimate.recommended_ctx_size));
        }
        if !next.llama_flash_attn.eq_ignore_ascii_case("on") {
            next.llama_flash_attn = "on".into();
            changed.push("Flash Attention → on".into());
        }
    } else {
        if next.ctx_size != estimate.recommended_ctx_size {
            next.ctx_size = estimate.recommended_ctx_size;
            changed.push(format!("context size → {}", estimate.recommended_ctx_size));
        }
        if estimate.uses_ssd_streaming && !next.ssd_streaming {
            next.ssd_streaming = true;
            changed.push("SSD streaming → on".into());
        }
    }

    let notice = if estimate.uses_ssd_streaming && is_gguf {
        format!(
            "{} llama.cpp cannot hold this model in RAM. Compile it to .jns in the Compiler tab, switch to Janas, then Auto-Tune again.",
            estimate.reason
        )
    } else if changed.is_empty() {
        format!("Nothing to change. {}", estimate.reason)
    } else if is_gguf {
        format!(
            "{} Applied: {}. Restart the engine to use them.",
            estimate.reason,
            changed.join(", ")
        )
    } else {
        format!(
            "{} Applied: {}. Restart the engine to use it.",
            estimate.reason,
            changed.join(", ")
        )
    };

    let saved = if changed.is_empty() {
        next
    } else {
        store.save(next)?
    };
    let requires_engine_switch = is_gguf && estimate.uses_ssd_streaming;

    Ok(AutoTuneResult {
        settings: saved,
        notice,
        requires_engine_switch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(vram_gib: u64, ram_gib: u64, storage: Option<f64>) -> HardwareProfile {
        HardwareProfile {
            os: "linux".into(),
            arch: "x86_64".into(),
            cpu: "Test CPU".into(),
            cpu_avx2: true,
            ram_total_bytes: ram_gib * GIB,
            ram_available_bytes: ram_gib * GIB,
            gpu_backend: if vram_gib > 0 {
                GpuBackend::Cuda
            } else {
                GpuBackend::None
            },
            gpu_name: (vram_gib > 0).then(|| "Test GPU".to_string()),
            vram_total_bytes: vram_gib * GIB,
            vram_free_bytes: vram_gib * GIB,
            storage_mb_s: storage,
            detected_at_ms: 1,
        }
    }

    #[test]
    fn four_gigabyte_model_on_eight_gigabyte_gpu_fits_vram() {
        let estimate =
            estimate_model_fit(4_000_000_000, Some(32), 8_192, &machine(8, 32, None));

        assert_eq!(estimate.tier, FitTier::FitsVram);
        assert_eq!(estimate.recommended_gpu_layers, 32);
        assert_eq!(estimate.recommended_ctx_size, 8_192);
        assert!(!estimate.uses_ssd_streaming);
        assert!(estimate.reason.contains("fits in"), "{}", estimate.reason);
    }

    #[test]
    fn eighteen_gigabyte_model_splits_across_gpu_and_ram() {
        let estimate =
            estimate_model_fit(18_000_000_000, Some(40), 8_192, &machine(8, 32, None));

        assert_eq!(estimate.tier, FitTier::RamGpuSplit);
        assert!(!estimate.uses_ssd_streaming);
        // (8 GiB - 1.5 GiB headroom) / ~22.0 GB working set * 40 layers = 12.
        assert_eq!(estimate.recommended_gpu_layers, 12);
        assert!(estimate.recommended_gpu_layers > 0);
        assert!(estimate.recommended_gpu_layers < 40);
        assert!(estimate.reason.contains("12 of 40 layers"), "{}", estimate.reason);
    }

    #[test]
    fn forty_two_gigabyte_model_recommends_janas_streaming() {
        let estimate =
            estimate_model_fit(42_000_000_000, Some(80), 8_192, &machine(0, 16, None));

        assert_eq!(estimate.tier, FitTier::RequiresSsdStreaming);
        assert_eq!(estimate.recommended_gpu_layers, 0);
        assert!(estimate.uses_ssd_streaming);
        assert!(estimate.reason.contains("exceeds"), "{}", estimate.reason);
    }

    #[test]
    fn a_slow_drive_turns_streaming_into_a_warning() {
        let estimate =
            estimate_model_fit(42_000_000_000, Some(80), 8_192, &machine(0, 16, Some(120.0)));

        assert_eq!(estimate.tier, FitTier::TightOrSlow);
        assert!(estimate.uses_ssd_streaming);
        assert_eq!(estimate.label, "Exceeds RAM (Slow Disk)");
        assert!(estimate.reason.contains("120 MB/s"), "{}", estimate.reason);
    }

    #[test]
    fn a_fast_drive_keeps_the_ssd_streaming_tier() {
        let estimate =
            estimate_model_fit(42_000_000_000, Some(80), 8_192, &machine(0, 16, Some(3_200.0)));

        assert_eq!(estimate.tier, FitTier::RequiresSsdStreaming);
        assert!(estimate.reason.contains("janas-server"), "{}", estimate.reason);
    }

    #[test]
    fn context_shrinks_to_fit_the_remaining_memory() {
        let estimate =
            estimate_model_fit(7_000_000_000, Some(32), 8_192, &machine(0, 8, None));

        assert_eq!(estimate.tier, FitTier::RamGpuSplit);
        assert_eq!(estimate.recommended_ctx_size, 4_096);
        assert!(estimate.recommended_ctx_size < 8_192);
    }

    #[test]
    fn kv_cache_scales_with_context_and_layers() {
        assert_eq!(kv_cache_bytes(8_192, 32), 1_073_741_824);
        assert_eq!(
            kv_cache_bytes(16_384, 32),
            kv_cache_bytes(8_192, 32) * 2
        );
        assert_eq!(
            kv_cache_bytes(8_192, 64),
            kv_cache_bytes(8_192, 32) * 2
        );
    }

    #[test]
    fn unknown_ram_is_never_reported_as_not_fitting() {
        let mut hw = machine(8, 0, None);
        hw.ram_total_bytes = 0;
        hw.ram_available_bytes = 0;
        let estimate = estimate_model_fit(40_000_000_000, Some(40), 8_192, &hw);

        assert_eq!(estimate.tier, FitTier::RamGpuSplit);
        assert!(estimate.reason.contains("could not be measured"));
    }

    #[test]
    fn hub_rows_without_a_header_use_an_implied_layer_count() {
        assert_eq!(implied_block_count(1_000_000_000), 24);
        assert_eq!(implied_block_count(8_000_000_000), 48);
        assert_eq!(implied_block_count(70_000_000_000), 80);
    }

    /// Manual diagnostic, not a fixture: measures the machine this runs on and
    /// prints the profile plus representative fits.
    ///
    /// ```text
    /// cargo test live_hardware_profile_and_fits -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "manual diagnostic: reads this machine's RAM, VRAM, disk and GPU"]
    fn live_hardware_profile_and_fits() {
        let dir = std::env::temp_dir().join(format!(
            "mikrotitan-fit-probe-{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        match crate::system::run_benchmark(dir.clone()) {
            Ok(benchmark) => println!(
                "storage: {:.0} MB/s sequential, {:.0} MB/s random, {:.0} IOPS ({})",
                benchmark.sequential_mb_s,
                benchmark.random_read_mb_s,
                benchmark.random_read_iops,
                benchmark.advice
            ),
            Err(err) => println!("storage: benchmark failed: {err}"),
        }
        println!("vram devices: {:#?}", crate::system::vram_devices());
        let profile = hardware_profile();
        println!("hardware profile: {profile:#?}");
        for (label, bytes) in [
            ("4 GB Q4", 4_000_000_000u64),
            ("18 GB Q5", 18_000_000_000),
            ("42 GB Q6", 42_000_000_000),
        ] {
            let estimate = estimate_model_fit(bytes, Some(36), 8_192, &profile);
            println!(
                "{label}: {} ({} GPU layers) — {}",
                estimate.label, estimate.recommended_gpu_layers, estimate.reason
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
