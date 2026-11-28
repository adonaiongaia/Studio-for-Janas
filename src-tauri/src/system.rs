//! Local machine inspection: hardware telemetry, drive benchmarks, project
//! ingestion, web search, the model library and audio transcription. Everything
//! here talks to the machine or the local engine —
//! no third-party account is involved except the search providers the user
//! explicitly enables.

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};

use crate::{
    error::{AppError, AppResult},
    settings::{expand_home, AppSettings, SettingsStore},
    util::now_ms,
};

// ---------------------------------------------------------------------------
// Artifact export
// ---------------------------------------------------------------------------
/// Persist an exported artifact after the native save dialog (or a chat
/// workspace write) has chosen the path. `base64` is used only for PNG data,
/// and a `data:image/...;base64,` prefix is accepted for convenience.
pub fn save_artifact_file(path: &str, contents: &str, base64: bool) -> AppResult<String> {
    const MAX_BYTES: usize = 32 * 1024 * 1024;
    // The webview can only send text or a base64 data URL. Reject an
    // over-large payload before base64 decoding allocates another copy.
    const MAX_INPUT_BYTES: usize = MAX_BYTES * 2;

    let target = path.trim();
    if target.is_empty() {
        return Err(AppError::message("Choose a file path before saving."));
    }
    if target.contains('\0') {
        return Err(AppError::message("That file path is not valid."));
    }
    if contents.len() > MAX_INPUT_BYTES {
        return Err(AppError::message(format!(
            "This artifact is larger than the {} MiB save limit.",
            MAX_BYTES / (1024 * 1024)
        )));
    }

    let bytes = if base64 {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let payload = contents
            .split_once(',')
            .filter(|(head, _)| head.starts_with("data:") && head.ends_with(";base64"))
            .map(|(_, body)| body)
            .unwrap_or(contents);
        STANDARD.decode(payload.trim()).map_err(|error| {
            AppError::message(format!("The image data could not be decoded: {error}"))
        })?
    } else {
        contents.as_bytes().to_vec()
    };
    if bytes.len() > MAX_BYTES {
        return Err(AppError::message(format!(
            "This artifact is larger than the {} MiB save limit.",
            MAX_BYTES / (1024 * 1024)
        )));
    }

    let destination = PathBuf::from(target);
    if destination.is_dir() {
        return Err(AppError::message("That path is a directory, not a file."));
    }
    if let Some(parent) = destination.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&destination, bytes)?;
    Ok(destination.display().to_string())
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceMetrics {
    pub present: bool,
    pub name: String,
    pub utilization: Option<f32>,
    pub memory_used: Option<u64>,
    pub memory_total: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSummary {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub cores: u32,
    pub mem_total_bytes: u64,
    pub gpu: DeviceMetrics,
    pub npu: DeviceMetrics,
}

/// A snapshot of the machine a benchmark ran on. It is stored with every
/// suite so a report copied months later still says which hardware produced it.
pub fn hardware_summary() -> HardwareSummary {
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                if key.trim().eq_ignore_ascii_case("model name") {
                    let value = value.trim();
                    (!value.is_empty()).then(|| value.to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| "Unknown CPU".into());
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();

    HardwareSummary {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu,
        cores: std::thread::available_parallelism()
            .map(|value| value.get() as u32)
            .unwrap_or(1),
        mem_total_bytes: parse_kib_field(&meminfo, "MemTotal:").unwrap_or(0),
        gpu: gpu_metrics(),
        npu: npu_metrics(),
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Telemetry {
    pub pid: Option<u32>,
    pub process_rss_bytes: u64,
    pub process_cpu_percent: f32,
    pub read_bytes_per_sec: f64,
    pub read_ops_per_sec: f64,
    pub model_bytes: u64,
    pub model_name: String,
    pub resident_ratio: f32,
    pub cores: u32,
    pub mem_total_bytes: u64,
    pub mem_available_bytes: u64,
    pub gpu: DeviceMetrics,
    pub npu: DeviceMetrics,
    pub sampled_at_ms: u64,
}

#[derive(Clone)]
struct Sample {
    at: Instant,
    read_bytes: u64,
    read_ops: u64,
    cpu_ticks: u64,
    clock_ticks: f64,
}

static LAST_SAMPLE: OnceLock<Mutex<Option<Sample>>> = OnceLock::new();

fn last_sample() -> &'static Mutex<Option<Sample>> {
    LAST_SAMPLE.get_or_init(|| Mutex::new(None))
}

fn read_to_string(path: &str) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn parse_kib_field(contents: &str, key: &str) -> Option<u64> {
    let line = contents.lines().find(|line| line.starts_with(key))?;
    line.split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib.saturating_mul(1024))
}

/// `/proc/<pid>/io` reports bytes actually read from block devices, which is
/// the closest honest measurement of SSD traffic without root access.
fn process_io(pid: u32) -> Option<(u64, u64)> {
    let contents = read_to_string(&format!("/proc/{pid}/io"))?;
    let read_bytes = contents
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes:"))?
        .trim()
        .parse()
        .ok()?;
    let syscr = contents
        .lines()
        .find_map(|line| line.strip_prefix("syscr:"))?
        .trim()
        .parse()
        .ok()?;
    Some((read_bytes, syscr))
}

fn process_rss(pid: u32) -> u64 {
    read_to_string(&format!("/proc/{pid}/status"))
        .and_then(|contents| parse_kib_field(&contents, "VmRSS:"))
        .unwrap_or(0)
}

/// utime + stime from `/proc/<pid>/stat`, in clock ticks.
fn process_cpu_ticks(pid: u32) -> u64 {
    let Some(stat) = read_to_string(&format!("/proc/{pid}/stat")) else {
        return 0;
    };
    let Some(close) = stat.rfind(')') else {
        return 0;
    };
    let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
    let utime = fields.get(11).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let stime = fields.get(12).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    utime.saturating_add(stime)
}

fn clock_ticks_per_second() -> f64 {
    #[cfg(unix)]
    unsafe {
        let ticks = libc::sysconf(libc::_SC_CLK_TCK);
        if ticks > 0 {
            return ticks as f64;
        }
    }
    100.0
}

fn read_sysfs_number(path: &str) -> Option<f64> {
    fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
}

fn gpu_metrics() -> DeviceMetrics {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return DeviceMetrics::default();
    };
    let mut cards: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().starts_with("card"))
                .unwrap_or(false)
        })
        .collect();
    cards.sort();

    for card in cards {
        let device = card.join("device");
        if !device.exists() {
            continue;
        }
        let busy = read_sysfs_number(&device.join("gpu_busy_percent").to_string_lossy())
            .map(|value| value as f32);
        let used = read_sysfs_number(&device.join("mem_info_vram_used").to_string_lossy())
            .map(|value| value as u64);
        let total = read_sysfs_number(&device.join("mem_info_vram_total").to_string_lossy())
            .map(|value| value as u64);
        if busy.is_some() || used.is_some() {
            let name = read_to_string(&device.join("uevent").to_string_lossy())
                .and_then(|text| {
                    text.lines()
                        .find_map(|line| line.strip_prefix("DRIVER="))
                        .map(str::to_string)
                })
                .unwrap_or_else(|| {
                    card.file_name()
                        .map(|name| name.to_string_lossy().to_string())
                        .unwrap_or_else(|| "GPU".into())
                });
            return DeviceMetrics {
                present: true,
                name,
                utilization: busy,
                memory_used: used,
                memory_total: total,
            };
        }
    }
    DeviceMetrics::default()
}

fn npu_metrics() -> DeviceMetrics {
    let Ok(entries) = fs::read_dir("/sys/class/accel") else {
        return DeviceMetrics::default();
    };
    let mut devices: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    devices.sort();
    for device in devices {
        let base = device.join("device");
        let mut utilization = None;
        for candidate in ["busy_percent", "utilization", "busy", "load"] {
            if let Some(value) =
                read_sysfs_number(&base.join(candidate).to_string_lossy()).filter(|v| *v <= 100.0)
            {
                utilization = Some(value as f32);
                break;
            }
        }
        let name = device
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "NPU".into());
        return DeviceMetrics {
            present: true,
            name,
            utilization,
            memory_used: None,
            memory_total: None,
        };
    }
    DeviceMetrics::default()
}

/// Sampling reads `/proc` and sysfs, which can block behind a busy or wedged
/// GPU driver. It runs on the blocking pool so a slow sample can never freeze
/// the webview; the panel polls this once a second.
#[tauri::command]
pub async fn get_telemetry(app: AppHandle, pid: Option<u32>) -> Telemetry {
    let store = app.state::<std::sync::Arc<SettingsStore>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || sample_telemetry(&store, pid))
        .await
        .unwrap_or_default()
}

fn sample_telemetry(store: &SettingsStore, pid: Option<u32>) -> Telemetry {
    let settings = store.get();
    let model_bytes = fs::metadata(expand_home(&settings.model_path))
        .map(|meta| meta.len())
        .unwrap_or(0);
    let model_name = settings.model_display_name().unwrap_or_else(|| "—".into());

    let mut read_bytes_per_sec = 0.0;
    let mut read_ops_per_sec = 0.0;
    let mut process_cpu_percent = 0.0;
    let mut rss = 0u64;

    if let Some(pid) = pid {
        rss = process_rss(pid);
        let io = process_io(pid);
        let cpu_ticks = process_cpu_ticks(pid);

        if let Ok(mut guard) = last_sample().lock() {
            if let Some(previous) = guard.as_ref() {
                let elapsed = previous.at.elapsed().as_secs_f64().max(0.001);
                if let Some((read_bytes, read_ops)) = io {
                    read_bytes_per_sec =
                        read_bytes.saturating_sub(previous.read_bytes) as f64 / elapsed;
                    read_ops_per_sec =
                        read_ops.saturating_sub(previous.read_ops) as f64 / elapsed;
                }
                let cpu_delta = cpu_ticks.saturating_sub(previous.cpu_ticks) as f64;
                process_cpu_percent =
                    (cpu_delta / previous.clock_ticks / elapsed * 100.0) as f32;
            }
            *guard = Some(Sample {
                at: Instant::now(),
                read_bytes: io.map(|value| value.0).unwrap_or(0),
                read_ops: io.map(|value| value.1).unwrap_or(0),
                cpu_ticks,
                clock_ticks: clock_ticks_per_second(),
            });
        }
    }

    let meminfo = read_to_string("/proc/meminfo").unwrap_or_default();
    let mem_total_bytes = parse_kib_field(&meminfo, "MemTotal:").unwrap_or(0);
    let mem_available_bytes = parse_kib_field(&meminfo, "MemAvailable:").unwrap_or(0);
    let cores = std::thread::available_parallelism()
        .map(|value| value.get() as u32)
        .unwrap_or(1);

    Telemetry {
        pid,
        process_rss_bytes: rss,
        process_cpu_percent: process_cpu_percent.clamp(0.0, 100.0 * cores as f32),
        read_bytes_per_sec,
        read_ops_per_sec,
        model_bytes,
        model_name,
        resident_ratio: if model_bytes > 0 {
            (rss as f64 / model_bytes as f64).clamp(0.0, 1.0) as f32
        } else {
            0.0
        },
        cores,
        mem_total_bytes,
        mem_available_bytes,
        gpu: gpu_metrics(),
        npu: npu_metrics(),
        sampled_at_ms: now_ms(),
    }
}

// ---------------------------------------------------------------------------
// Storage benchmark
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct StorageBenchmark {
    pub path: String,
    pub file_bytes: u64,
    pub sequential_mb_s: f64,
    pub random_read_iops: f64,
    pub random_read_mb_s: f64,
    pub average_latency_us: f64,
    pub elapsed_ms: u64,
    pub advice: String,
}

static LAST_STORAGE_BENCH: std::sync::OnceLock<Mutex<Option<StorageBenchmark>>> =
    std::sync::OnceLock::new();

fn last_storage_bench() -> &'static Mutex<Option<StorageBenchmark>> {
    LAST_STORAGE_BENCH.get_or_init(|| Mutex::new(None))
}

/// The most recent 64 MB benchmark, when one has run this session. The fit
/// estimator uses its sequential figure to tell a fast NVMe from a drive that
/// would make SSD streaming painful.
pub fn last_storage_benchmark() -> Option<StorageBenchmark> {
    last_storage_bench()
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Total and currently available system RAM. Available is 0 on platforms that
/// do not expose it (the fit estimator treats 0 as "use total").
pub fn memory_snapshot() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let total = parse_kib_field(&meminfo, "MemTotal:").unwrap_or(0);
        let available = parse_kib_field(&meminfo, "MemAvailable:").unwrap_or(0);
        if total > 0 {
            return (total, available);
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(total) = String::from_utf8_lossy(&output.stdout).trim().parse::<u64>() {
                if total > 0 {
                    return (total, 0);
                }
            }
        }
    }
    (0, 0)
}

/// Which GPU's memory a caller needs. Hybrid laptops expose at least two DRM
/// cards, and an iGPU's small carve-out must never stand in for a discrete
/// card, so the caller names the backend it detected. Cards are only summed
/// across multiple devices of that same vendor (a multi-GPU CUDA/ROCm rig).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramPreference {
    Cuda,
    Rocm,
    /// No backend match: report the largest single card.
    Any,
}

/// One DRM card that publishes a VRAM counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramDevice {
    /// "nvidia", "amd", "intel" or "unknown".
    pub vendor: String,
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl VramDevice {
    fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
}

/// Every DRM card that publishes a VRAM counter, with its vendor. The vendor
/// comes from the PCI id, falling back to the kernel driver name.
pub fn vram_devices() -> Vec<VramDevice> {
    vram_devices_in(Path::new("/sys/class/drm"))
}

fn vram_devices_in(root: &Path) -> Vec<VramDevice> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .map(|name| {
                    let name = name.to_string_lossy();
                    name.starts_with("card")
                        && name
                            .chars()
                            .skip(4)
                            .all(|character| character.is_ascii_digit())
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    let mut devices = Vec::new();
    for path in paths {
        let device = path.join("device");
        let total = read_sysfs_number(&device.join("mem_info_vram_total").to_string_lossy())
            .map(|value| value as u64)
            .unwrap_or(0);
        if total == 0 {
            continue;
        }
        let used = read_sysfs_number(&device.join("mem_info_vram_used").to_string_lossy())
            .map(|value| value as u64)
            .unwrap_or(0);
        devices.push(VramDevice {
            vendor: device_vendor(&device),
            total_bytes: total,
            used_bytes: used.min(total),
        });
    }
    devices
}

fn device_vendor(device: &Path) -> String {
    if let Some(raw) = read_to_string(&device.join("vendor").to_string_lossy()) {
        match raw.trim().to_ascii_lowercase().as_str() {
            "0x10de" => return "nvidia".into(),
            "0x1002" => return "amd".into(),
            "0x8086" => return "intel".into(),
            _ => {}
        }
    }
    let Some(uevent) = read_to_string(&device.join("uevent").to_string_lossy()) else {
        return "unknown".into();
    };
    let Some(driver) = uevent
        .lines()
        .find_map(|line| line.strip_prefix("DRIVER="))
    else {
        return "unknown".into();
    };
    match driver.trim().to_ascii_lowercase().as_str() {
        "nvidia" | "nvidia-drm" => "nvidia".into(),
        "amdgpu" | "radeon" => "amd".into(),
        "i915" | "xe" => "intel".into(),
        _ => "unknown".into(),
    }
}

/// The budget for one backend. `(0, 0)` means no card of that vendor publishes
/// counters, so the caller must try another source instead of borrowing a
/// different GPU's memory.
pub fn select_vram(devices: &[VramDevice], preference: VramPreference) -> (u64, u64) {
    let wanted = match preference {
        VramPreference::Cuda => Some("nvidia"),
        VramPreference::Rocm => Some("amd"),
        VramPreference::Any => None,
    };
    if let Some(vendor) = wanted {
        let mut total = 0u64;
        let mut free = 0u64;
        let mut matched = false;
        for device in devices.iter().filter(|device| device.vendor == vendor) {
            matched = true;
            total = total.saturating_add(device.total_bytes);
            free = free.saturating_add(device.free_bytes());
        }
        return if matched { (total, free) } else { (0, 0) };
    }
    devices
        .iter()
        .max_by_key(|device| device.total_bytes)
        .map(|device| (device.total_bytes, device.free_bytes()))
        .unwrap_or((0, 0))
}

/// Total and free VRAM for one backend. The DRM sysfs counters cover `amdgpu`,
/// `i915` and the NVIDIA proprietary driver; a driver that hides them (an
/// older driver, a sandboxed sysfs, a MIG-only view) is queried through
/// `nvidia-smi` instead, so a real GPU never looks like it has no memory.
pub fn vram_snapshot(preference: VramPreference) -> (u64, u64) {
    let selected = select_vram(&vram_devices(), preference);
    if selected.0 > 0 {
        return selected;
    }
    // `nvidia-smi` only ever speaks for NVIDIA. Asking it for a ROCm budget
    // would hand an AMD backend someone else's card.
    if preference == VramPreference::Rocm {
        return (0, 0);
    }
    nvidia_smi_vram().unwrap_or((0, 0))
}

/// Sum `nvidia-smi`'s per-GPU memory counters. Values are MiB; a device that
/// reports `[N/A]` or `[Not Supported]` is skipped rather than counted as 0.
fn parse_nvidia_smi_memory(text: &str) -> (u64, u64) {
    const MIB: u64 = 1024 * 1024;
    let mut total_mib = 0u64;
    let mut used_mib = 0u64;
    for line in text.lines() {
        let mut fields = line.split(',').map(str::trim);
        let (Some(total), Some(used)) = (fields.next(), fields.next()) else {
            continue;
        };
        let Ok(total) = total.parse::<u64>() else {
            continue;
        };
        let used = used.parse::<u64>().unwrap_or(0).min(total);
        total_mib = total_mib.saturating_add(total);
        used_mib = used_mib.saturating_add(used);
    }
    (
        total_mib.saturating_mul(MIB),
        total_mib.saturating_sub(used_mib).saturating_mul(MIB),
    )
}

fn nvidia_smi_vram() -> Option<(u64, u64)> {
    let text = crate::provision::run_short(
        "nvidia-smi",
        &[
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_secs(3),
    )?;
    let snapshot = parse_nvidia_smi_memory(&text);
    (snapshot.0 > 0).then_some(snapshot)
}

fn bench_seq_read(file: &mut File, size: u64) -> std::io::Result<f64> {
    file.seek(SeekFrom::Start(0))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let started = Instant::now();
    let mut read_total = 0u64;
    while read_total < size {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        read_total += read as u64;
    }
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    Ok(read_total as f64 / elapsed / (1024.0 * 1024.0))
}

#[cfg(unix)]
fn bench_random(file: &File, size: u64, count: usize) -> std::io::Result<f64> {
    use std::os::unix::fs::FileExt;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut buffer = [0u8; 4096];
    let mut bytes = 0u64;
    let started = Instant::now();
    for _ in 0..count {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let offset = seed % (size.saturating_sub(4096).max(1) / 4096) * 4096;
        let read = file.read_exact_at(&mut buffer, offset)?;
        let _ = read;
        bytes += buffer.len() as u64;
    }
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    Ok(bytes as f64 / elapsed / (1024.0 * 1024.0))
}

#[cfg(not(unix))]
fn bench_random(file: &File, size: u64, count: usize) -> std::io::Result<f64> {
    let mut buffer = [0u8; 4096];
    let mut bytes = 0u64;
    let started = Instant::now();
    for index in 0..count {
        let offset = (index as u64 * 4096 * 17) % size.max(1);
        let mut handle = file.try_clone()?;
        handle.seek(SeekFrom::Start(offset))?;
        handle.read_exact(&mut buffer)?;
        bytes += buffer.len() as u64;
    }
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    Ok(bytes as f64 / elapsed / (1024.0 * 1024.0))
}

#[tauri::command]
pub async fn benchmark_storage(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    directory: Option<String>,
) -> AppResult<StorageBenchmark> {
    let target = directory
        .filter(|value| !value.trim().is_empty())
        .map(|value| expand_home(&value))
        .unwrap_or_else(|| expand_home(&store.get().models_dir));
    tauri::async_runtime::spawn_blocking(move || run_benchmark(target))
        .await
        .map_err(|err| AppError::message(format!("The benchmark stopped: {err}")))?
}

/// The 64 MB sequential/random pass behind `benchmark_storage`. It also stores
/// its result for the hardware fit estimator, so this is `pub(crate)` for the
/// manual hardware diagnostic.
pub(crate) fn run_benchmark(target: PathBuf) -> AppResult<StorageBenchmark> {
    fs::create_dir_all(&target)?;
    let path = target.join(format!(".janas-bench-{}.bin", std::process::id()));
    let size: u64 = 64 * 1024 * 1024;
    let started = Instant::now();

    {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        let chunk = vec![0xA5u8; 1024 * 1024];
        let mut written = 0u64;
        while written < size {
            file.write_all(&chunk)?;
            written += chunk.len() as u64;
        }
        file.sync_all()?;
    }

    let result = (|| -> std::io::Result<(f64, f64, f64)> {
        let mut file = File::open(&path)?;
        let sequential = bench_seq_read(&mut file, size)?;
        let random_started = Instant::now();
        let random_mb_s = bench_random(&file, size, 1_500)?;
        let elapsed = random_started.elapsed().as_secs_f64().max(0.001);
        let iops = 1_500.0 / elapsed;
        Ok((sequential, random_mb_s, iops))
    })();

    let _ = fs::remove_file(&path);
    let (sequential_mb_s, random_read_mb_s, random_read_iops) = result?;
    let elapsed_secs = elapsed_from(size, sequential_mb_s, random_read_iops);
    let advice = if random_read_iops >= 60_000.0 && sequential_mb_s >= 1_800.0 {
        "NVMe-class. A 48 GB or 70 B+ mixture can stream experts without the drive becoming the bottleneck.".into()
    } else if random_read_iops >= 20_000.0 {
        "Solid NVMe/SATA SSD. Large mixtures will run, but expect the initial prompt read to be I/O-bound.".into()
    } else {
        "This drive will bottleneck large mixtures. Choose a smaller quantization, resident experts, or a faster NVMe drive.".into()
    };

    let benchmark = StorageBenchmark {
        path: path.to_string_lossy().to_string(),
        file_bytes: size,
        sequential_mb_s,
        random_read_iops,
        random_read_mb_s,
        average_latency_us: if random_read_iops > 0.0 {
            1_000_000.0 / random_read_iops
        } else {
            0.0
        },
        elapsed_ms: (started.elapsed().as_millis() as u64).max(elapsed_secs as u64),
        advice,
    };
    if let Ok(mut guard) = last_storage_bench().lock() {
        *guard = Some(benchmark.clone());
    }
    Ok(benchmark)
}

fn elapsed_from(size: u64, sequential_mb_s: f64, _iops: f64) -> f64 {
    if sequential_mb_s <= 0.0 {
        return 0.0;
    }
    size as f64 / (sequential_mb_s * 1024.0 * 1024.0)
}

// ---------------------------------------------------------------------------
// Project ingestion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ProjectIngest {
    pub root: String,
    pub name: String,
    pub files: usize,
    pub bytes: u64,
    pub tokens: u64,
    pub tree: String,
    pub bundle: String,
    pub truncated: bool,
}

pub(crate) const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".cache",
    "vendor",
    "Pods",
];

const IGNORED_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "ico", "bmp", "pdf", "zip", "gz", "tgz", "tar", "7z",
    "rar", "mp4", "mov", "mkv", "mp3", "wav", "onnx", "bin", "so", "dylib", "dll", "exe", "o",
    "a", "woff", "woff2", "ttf", "otf", "class", "jar", "wasm", "pyc", "gguf", "jns", "safetensors",
];

fn collect_project_files(
    root: &Path,
    dir: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
    truncated: &mut bool,
) {
    if depth > 12 || out.len() >= 320 {
        *truncated = true;
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') && name != ".env.example" && depth == 0 && name != ".gitignore" {
            // Hidden folders are noise; hidden files at the root are skipped
            // too unless they are ordinary dotfiles like .gitignore.
            if path.is_dir() {
                continue;
            }
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            if IGNORED_DIRS.iter().any(|ignored| ignored.eq_ignore_ascii_case(&name)) {
                continue;
            }
            collect_project_files(root, &path, depth + 1, out, truncated);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        // Never bundle credentials or dotenv files into a model prompt.
        let lower_name = name.to_ascii_lowercase();
        if lower_name.starts_with(".env")
            && !matches!(
                lower_name.as_str(),
                ".env.example" | ".env.sample" | ".env.template"
            )
        {
            continue;
        }
        if matches!(
            lower_name.as_str(),
            ".npmrc" | ".netrc" | "id_rsa" | "id_ed25519" | "credentials.json"
        ) {
            continue;
        }
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_ascii_lowercase();
            if IGNORED_EXTENSIONS.contains(&ext.as_str()) {
                continue;
            }
        }
        if meta.len() > 256 * 1024 {
            *truncated = true;
            continue;
        }
        out.push(path);
        if out.len() >= 320 {
            *truncated = true;
            return;
        }
    }
}

#[tauri::command]
pub async fn ingest_project(path: String) -> AppResult<ProjectIngest> {
    let input = expand_home(&path);
    tauri::async_runtime::spawn_blocking(move || build_project_bundle(input))
        .await
        .map_err(|err| AppError::message(format!("Project ingestion stopped: {err}")))?
}

fn build_project_bundle(input: PathBuf) -> AppResult<ProjectIngest> {
    if input.is_file() {
        return Err(AppError::message(
            "Drop a project folder, not a single file. Single files can still be pasted into the prompt.",
        ));
    }
    let root = input;
    if !root.is_dir() {
        return Err(AppError::message(format!(
            "{} is not a folder.",
            root.display()
        )));
    }
    let name = root
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "project".into());

    let mut files = Vec::new();
    let mut truncated = false;
    collect_project_files(&root, &root, 0, &mut files, &mut truncated);

    let mut tree_lines = Vec::new();
    let mut bundle = String::new();
    let mut bytes = 0u64;
    let budget: u64 = 1_500_000;
    let per_file_budget: usize = 48 * 1024;

    for file in &files {
        let relative = file
            .strip_prefix(&root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        let depth = relative.matches('/').count();
        tree_lines.push(format!("{}{}", "  ".repeat(depth), relative));
        if bytes >= budget {
            truncated = true;
            continue;
        }
        let Ok(raw) = fs::read(file) else {
            continue;
        };
        let text = String::from_utf8_lossy(&raw);
        let clipped = if text.len() > per_file_budget {
            truncated = true;
            &text[..text
                .char_indices()
                .take_while(|(index, _)| *index < per_file_budget)
                .last()
                .map(|(index, _)| index)
                .unwrap_or(0)]
        } else {
            &text
        };
        if clipped.contains('\u{0}') {
            continue;
        }
        bytes += clipped.len() as u64;
        bundle.push_str(&format!("\n=== {relative} ===\n"));
        bundle.push_str(clipped.trim_end());
        bundle.push('\n');
    }

    let tokens = (bundle.chars().count() as u64 / 4).max(1);
    Ok(ProjectIngest {
        root: root.to_string_lossy().to_string(),
        name,
        files: files.len(),
        bytes,
        tokens,
        tree: tree_lines.join("\n"),
        bundle,
        truncated,
    })
}

// ---------------------------------------------------------------------------
// Web search pipeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct WebPageResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub text: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WebSearchResponse {
    pub query: String,
    pub provider: String,
    pub results: Vec<WebPageResult>,
    pub elapsed_ms: u64,
    pub bytes: u64,
    pub message: String,
}

fn search_client() -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::ACCEPT,
        reqwest::header::HeaderValue::from_static(
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        ),
    );
    headers.insert(
        reqwest::header::ACCEPT_LANGUAGE,
        reqwest::header::HeaderValue::from_static("en-US,en;q=0.9"),
    );
    reqwest::Client::builder()
        .user_agent(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        .default_headers(headers)
        .timeout(Duration::from_secs(12))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn decode_href(raw: &str) -> Option<String> {
    let mut href = raw
        .trim()
        .trim_matches(|character| character == '"' || character == '\'')
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x2F;", "/")
        .replace("&#47;", "/");

    if href.starts_with("//") {
        href = format!("https:{href}");
    }
    if let Some(index) = href.find("uddg=") {
        let encoded = href[index + 5..].split('&').next().unwrap_or_default();
        if let Ok(decoded) = urlencoding::decode(encoded) {
            href = decoded.into_owned();
            if href.starts_with("//") {
                href = format!("https:{href}");
            }
        }
    }

    let parsed = reqwest::Url::parse(href.trim()).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    if parsed.host_str().map(str::is_empty).unwrap_or(true) {
        return None;
    }
    Some(parsed.to_string())
}

fn strip_tags(fragment: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in fragment.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&#x27;", "'")
        .replace("&hellip;", "…")
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Read one attribute from an opening tag. Both quote styles are accepted, and
/// the name has to be a real attribute boundary (`data-href` is not `href`).
fn html_attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = name.to_ascii_lowercase();
    let mut cursor = 0usize;

    while let Some(offset) = lower[cursor..].find(&needle) {
        let start = cursor + offset;
        let after = start + needle.len();
        let boundary = start == 0
            || lower
                .as_bytes()
                .get(start - 1)
                .map(|byte| byte.is_ascii_whitespace() || *byte == b'<')
                .unwrap_or(false);
        if !boundary {
            cursor = after;
            continue;
        }

        let rest = tag[after..].trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            cursor = after;
            continue;
        };
        let rest = rest.trim_start();
        let value = if let Some(rest) = rest.strip_prefix('"') {
            let end = rest.find('"')?;
            &rest[..end]
        } else if let Some(rest) = rest.strip_prefix('\'') {
            let end = rest.find('\'')?;
            &rest[..end]
        } else {
            let end = rest
                .find(|character: char| character.is_whitespace() || character == '>' || character == '/')
                .unwrap_or(rest.len());
            &rest[..end]
        };
        return Some(value.to_string());
    }
    None
}

fn tag_has_class(tag: &str, class_name: &str) -> bool {
    html_attr(tag, "class")
        .map(|value| {
            value
                .split_whitespace()
                .any(|token| token.eq_ignore_ascii_case(class_name))
        })
        .unwrap_or(false)
}

fn find_snippet(html: &str, from: usize) -> String {
    let end = (from + 6_000).min(html.len());
    let Some(window) = html.get(from..end) else {
        return String::new();
    };
    let Some(relative) = window.find("result__snippet") else {
        return String::new();
    };
    let before = &window[..relative];
    let Some(tag_start_relative) = before.rfind('<') else {
        return String::new();
    };
    // A CSS rule starts inside <style> and already has a closing '>'.
    if before[tag_start_relative..].contains('>') {
        return String::new();
    }
    let tag_start = from + tag_start_relative;
    let Some(open_relative) = html[tag_start..].find('>') else {
        return String::new();
    };
    let text_start = tag_start + open_relative + 1;
    let Some(close_relative) = html[text_start..].find("</") else {
        return String::new();
    };
    let snippet = normalize_text(&strip_tags(&html[text_start..text_start + close_relative]));
    snippet.chars().take(300).collect()
}

/// Walk opening `<a>` tags and keep only the ones whose class matches. This
/// avoids the old failure mode where the class name was found in a stylesheet
/// and an unrelated anchor (sometimes labelled "go") became the citation.
fn parse_search_results(html: &str, anchor_class: &str) -> Vec<(String, String, String)> {
    let mut results: Vec<(String, String, String)> = Vec::new();
    let mut cursor = 0usize;

    while results.len() < 6 {
        let Some(relative) = html[cursor..].find("<a") else {
            break;
        };
        let anchor = cursor + relative;
        let after_name = anchor + 2;
        let next = html[after_name..].chars().next();
        let is_anchor = matches!(next, Some(character) if character.is_whitespace() || character == '>');
        if !is_anchor {
            cursor = after_name;
            continue;
        }
        let Some(tag_end_relative) = html[anchor..].find('>') else {
            break;
        };
        let tag_end = anchor + tag_end_relative;
        let tag = &html[anchor..=tag_end];
        let after = tag_end + 1;
        if !tag_has_class(tag, anchor_class) {
            cursor = after;
            continue;
        }
        let Some(href) = html_attr(tag, "href") else {
            cursor = after;
            continue;
        };
        let Some(close_relative) = html[after..].find("</a>") else {
            break;
        };
        let title = normalize_text(&strip_tags(&html[after..after + close_relative]));
        let anchor_end = after + close_relative + 4;
        let snippet = find_snippet(html, anchor_end);

        if !title.is_empty() {
            if let Some(url) = decode_href(&href) {
                if !results
                    .iter()
                    .any(|(_, existing_url, _)| existing_url == &url)
                {
                    results.push((title, url, snippet));
                }
            }
        }
        cursor = anchor_end;
    }

    results
}

fn parse_bing_results(html: &str) -> Vec<(String, String, String)> {
    let mut results: Vec<(String, String, String)> = Vec::new();
    let mut cursor = 0usize;

    while results.len() < 6 {
        let Some(relative) = html[cursor..].find("b_algo") else {
            break;
        };
        let block_start = cursor + relative;
        let block_end = (block_start + 12_000).min(html.len());
        let Some(block) = html.get(block_start..block_end) else {
            break;
        };
        let Some(anchor_relative) = block.find("<a") else {
            cursor = block_start + 6;
            continue;
        };
        let anchor = block_start + anchor_relative;
        let Some(tag_end_relative) = html[anchor..].find('>') else {
            break;
        };
        let tag_end = anchor + tag_end_relative;
        let tag = &html[anchor..=tag_end];
        let Some(href) = html_attr(tag, "href") else {
            cursor = block_start + 6;
            continue;
        };
        let Some(close_relative) = html[tag_end + 1..].find("</a>") else {
            break;
        };
        let title = normalize_text(&strip_tags(
            &html[tag_end + 1..tag_end + 1 + close_relative],
        ));
        let with_title = tag_end + 1 + close_relative + 4;
        let snippet = find_paragraph(html, with_title);

        if !title.is_empty() {
            if let Some(url) = decode_href(&href) {
                if !results
                    .iter()
                    .any(|(_, existing_url, _)| existing_url == &url)
                {
                    results.push((title, url, snippet));
                }
            }
        }
        cursor = block_start + 6;
    }

    results
}

fn find_paragraph(html: &str, from: usize) -> String {
    let end = (from + 4_000).min(html.len());
    let Some(window) = html.get(from..end) else {
        return String::new();
    };
    let Some(relative) = window.find("<p") else {
        return String::new();
    };
    let start = from + relative;
    let Some(open_relative) = html[start..].find('>') else {
        return String::new();
    };
    let text_start = start + open_relative + 1;
    let Some(close_relative) = html[text_start..].find("</p>") else {
        return String::new();
    };
    normalize_text(&strip_tags(&html[text_start..text_start + close_relative]))
        .chars()
        .take(300)
        .collect()
}

/// Turn an HTML page into readable text: scripts, styles and navigation are
/// dropped, block tags become newlines, and whitespace is collapsed.
fn html_to_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut cleaned = String::with_capacity(html.len() / 4);
    let mut index = 0usize;
    while index < html.len() {
        if let Some(offset) = lower[index..].find("<script") {
            let start = index + offset;
            cleaned.push_str(&html[index..start]);
            match lower[start..].find("</script>") {
                Some(end) => index = start + end + 9,
                None => break,
            }
            continue;
        }
        cleaned.push_str(&html[index..]);
        break;
    }

    let lower = cleaned.to_ascii_lowercase();
    let mut without_style = String::with_capacity(cleaned.len());
    let mut index = 0usize;
    while index < cleaned.len() {
        if let Some(offset) = lower[index..].find("<style") {
            let start = index + offset;
            without_style.push_str(&cleaned[index..start]);
            match lower[start..].find("</style>") {
                Some(end) => index = start + end + 8,
                None => break,
            }
            continue;
        }
        without_style.push_str(&cleaned[index..]);
        break;
    }

    let mut text = String::with_capacity(without_style.len());
    let mut in_tag = false;
    for ch in without_style.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                text.push(' ');
            }
            '\n' | '\r' if !in_tag => text.push('\n'),
            _ if !in_tag => text.push(ch),
            _ => {}
        }
    }
    text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&hellip;", "…");
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(14_000)
        .collect()
}

#[tauri::command]
pub async fn web_search(query: String, pages: Option<u32>) -> AppResult<WebSearchResponse> {
    let query = query.trim().to_string();
    if query.is_empty() {
        return Err(AppError::message("A web search needs a query."));
    }
    let started = Instant::now();
    let client = search_client();

    let mut found: Vec<(String, String, String)> = Vec::new();
    let mut provider = "DuckDuckGo".to_string();
    let mut notes: Vec<String> = Vec::new();

    match client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query.as_str())])
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {
            let html = response.text().await.unwrap_or_default();
            found = parse_search_results(&html, "result__a");
        }
        Ok(response) => notes.push(format!("DuckDuckGo answered HTTP {}.", response.status())),
        Err(err) => notes.push(format!("DuckDuckGo could not be reached: {err}")),
    }

    if found.is_empty() {
        match client
            .get("https://lite.duckduckgo.com/lite/")
            .query(&[("q", query.as_str())])
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let html = response.text().await.unwrap_or_default();
                found = parse_search_results(&html, "result-link");
                if found.is_empty() {
                    found = parse_search_results(&html, "result__a");
                }
                if !found.is_empty() {
                    provider = "DuckDuckGo Lite".into();
                }
            }
            Ok(response) => notes.push(format!(
                "DuckDuckGo Lite answered HTTP {}.",
                response.status()
            )),
            Err(err) => notes.push(format!("DuckDuckGo Lite could not be reached: {err}")),
        }
    }

    if found.is_empty() {
        match client
            .get("https://www.bing.com/search")
            .query(&[("q", query.as_str()), ("setlang", "en")])
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let html = response.text().await.unwrap_or_default();
                found = parse_bing_results(&html);
                if !found.is_empty() {
                    provider = "Bing".into();
                }
            }
            Ok(response) => notes.push(format!("Bing answered HTTP {}.", response.status())),
            Err(err) => notes.push(format!("Bing could not be reached: {err}")),
        }
    }

    if found.is_empty() {
        return Err(AppError::message(if notes.is_empty() {
            "The search returned no results. Check the connection and try a different query.".into()
        } else {
            format!("No readable search results. {}", notes.join(" "))
        }));
    }

    let wanted = pages.unwrap_or(3).clamp(1, 4) as usize;
    found.truncate(wanted);
    let mut results = Vec::new();
    let mut bytes = 0u64;

    for (title, url, snippet) in found {
        let mut page_bytes = 0u64;
        let mut text = String::new();

        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                let content_type = response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if content_type.is_empty()
                    || content_type.contains("html")
                    || content_type.contains("text")
                {
                    if let Ok(body) = response.bytes().await {
                        page_bytes = body.len() as u64;
                        let capped = &body[..body.len().min(512 * 1024)];
                        text = html_to_text(&String::from_utf8_lossy(capped));
                    }
                }
            }
            _ => {}
        }

        if text.trim().chars().count() < 40 {
            text = snippet.clone();
        }
        bytes = bytes.saturating_add(page_bytes);
        results.push(WebPageResult {
            title,
            url,
            snippet,
            text,
            bytes: page_bytes,
        });
    }

    let populated = results
        .iter()
        .filter(|result| !result.text.trim().is_empty())
        .count();
    Ok(WebSearchResponse {
        query: query.clone(),
        provider,
        results,
        elapsed_ms: started.elapsed().as_millis() as u64,
        bytes,
        message: format!(
            "Ingested {populated} page{} ({}) into the prefill bundle.",
            if populated == 1 { "" } else { "s" },
            format_bytes(bytes)
        ),
    })
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0usize;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

// ---------------------------------------------------------------------------
// Audio transcription (uses the engine's speech model when one is loaded)
// ---------------------------------------------------------------------------

/// Transcribe a recording entirely on this machine with whisper.cpp.
///
/// `audio_base64` is normally the 16 kHz mono 16-bit WAV produced by the
/// Composer's Web Audio recorder. Raw 16-bit PCM and the compressed formats
/// whisper.cpp decodes itself (MP3, OGG, FLAC) are accepted too. The managed
/// floating text is passed through without envelope wrapping so each caller can
/// hand it to the frontend exactly as spoken.
#[tauri::command]
pub async fn transcribe_audio(
    app: AppHandle,
    store: State<'_, std::sync::Arc<SettingsStore>>,
    audio_base64: String,
    mime: Option<String>,
    file_name: Option<String>,
    model: Option<String>,
    language: Option<String>,
    strip_non_speech: Option<bool>,
) -> AppResult<crate::whisper::TranscriptionResult> {
    crate::whisper::transcribe_audio_payload(
        &app,
        &store.get(),
        &audio_base64,
        mime.as_deref(),
        file_name.as_deref(),
        model.as_deref(),
        language.as_deref(),
        strip_non_speech,
    )
    .await
}

/// Transcribe a dropped or attached audio file directly, without base64
/// crossing the IPC boundary.
#[tauri::command]
pub async fn transcribe_audio_file(
    app: AppHandle,
    store: State<'_, std::sync::Arc<SettingsStore>>,
    path: String,
    model: Option<String>,
    language: Option<String>,
) -> AppResult<crate::whisper::TranscriptionResult> {
    crate::whisper::transcribe_audio_path(
        &app,
        &store.get(),
        &path,
        model.as_deref(),
        language.as_deref(),
    )
    .await
}

// ---------------------------------------------------------------------------
// Model library
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct LibraryModel {
    pub name: String,
    pub path: String,
    pub format: String,
    pub size_bytes: u64,
    pub quant: String,
    pub architecture: String,
    pub parameters: String,
    pub publisher: String,
    pub repo: Option<String>,
    pub downloaded_at: Option<String>,
    pub thinking: bool,
    pub tools: bool,
    pub capabilities_source: String,
    /// Another file with the same model name and byte size exists.
    pub duplicate: bool,
    /// How many local files belong to the same Hugging Face repository.
    pub same_repo_count: usize,
    /// Whether this file fits in VRAM, needs a RAM/GPU split, or must stream.
    pub fit: crate::fit::ModelFitEstimate,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelVerification {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub expected: Option<String>,
    pub ok: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LibraryDisk {
    pub models_dir: String,
    pub total_bytes: u64,
    pub free_bytes: Option<u64>,
}

const QUANTS: &[&str] = &[
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

const FAMILIES: &[(&str, &str)] = &[
    ("qwen3", "Qwen3"),
    ("qwen2", "Qwen2"),
    ("qwen", "Qwen"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("deepseek", "DeepSeek"),
    ("mixtral", "Mixtral"),
    ("mistral", "Mistral"),
    ("phi", "Phi"),
    ("glm", "GLM"),
    ("olmo", "OLMo"),
    ("mamba", "Mamba"),
    ("rwkv", "RWKV"),
    ("falcon", "Falcon"),
    ("gpt-oss", "GPT-OSS"),
];

fn infer_quant(name: &str) -> String {
    let upper = name.to_ascii_uppercase();
    QUANTS
        .iter()
        .find(|token| upper.contains(**token))
        .map(|token| token.to_string())
        .unwrap_or_else(|| "—".into())
}

/// `GGUF` is the file container, not a quantization, so a recorded value that
/// says only `GGUF` must not shadow what the header can prove.
fn is_specific_quant(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed != "—" && !trimmed.eq_ignore_ascii_case("GGUF")
}

fn infer_architecture(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    FAMILIES
        .iter()
        .find(|(needle, _)| lower.contains(needle))
        .map(|(_, label)| label.to_string())
        .unwrap_or_else(|| "Unknown".into())
}

fn infer_parameters(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    for index in 0..chars.len() {
        if chars[index] != 'b' && chars[index] != 'm' {
            continue;
        }
        let mut start = index;
        while start > 0
            && (chars[start - 1].is_ascii_digit() || chars[start - 1] == '.')
        {
            start -= 1;
        }
        if start == index {
            continue;
        }
        let number: String = chars[start..index].iter().collect();
        if number.starts_with('.') || number.matches('.').count() > 1 {
            continue;
        }
        if index + 1 < chars.len() && chars[index + 1].is_ascii_alphabetic() {
            continue;
        }
        let Ok(value) = number.parse::<f64>() else {
            continue;
        };
        if value < 0.1 || value > 1000.0 {
            continue;
        }
        return if chars[index] == 'b' {
            format!("{value}B")
        } else {
            format!("{value}M")
        };
    }
    "—".into()
}

fn template_capabilities(template: &str) -> (bool, bool) {
    let lower = template.to_ascii_lowercase();
    let thinking = [
        "enable_thinking",
        "reasoning_content",
        "reasoning_effort",
        "reasoning",
        "thinking",
        "<|think|>",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let tools = [
        "tool_calls",
        "tool_call",
        "tool_choice",
        "function_call",
        "<tool_call>",
        "tools",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    (thinking, tools)
}

fn inferred_capabilities(name: &str) -> (bool, bool) {
    let lower = name.to_ascii_lowercase();
    let thinking = [
        "qwen3",
        "deepseek-r1",
        "deepseek-v3.1",
        "deepseek-v3.2",
        "reasoning",
        "thinking",
        "magistral",
        "phi-4-reasoning",
        "glm-4.5",
        "glm-4.6",
        "minimax-m2",
        "kimi-k2",
        "gpt-oss",
        "granite-4",
        "seed-oss",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let tools = [
        "qwen",
        "llama-3.1",
        "llama-3.2",
        "llama-3.3",
        "llama-4",
        "mistral",
        "mixtral",
        "hermes",
        "deepseek",
        "command-r",
        "granite",
        "glm-4",
        "phi-4",
        "gpt-oss",
        "devstral",
        "kimi-k2",
        "minimax",
        "seed-oss",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    (thinking, tools)
}

fn detect_capabilities(
    features: Option<&crate::gguf::GgufFeatures>,
    haystack: &str,
) -> (bool, bool, String) {
    if let Some(template) = features
        .and_then(|value| value.chat_template.as_deref())
        .filter(|value| !value.trim().is_empty())
    {
        let (thinking, tools) = template_capabilities(template);
        (thinking, tools, "chat template".into())
    } else {
        let (thinking, tools) = inferred_capabilities(haystack);
        (thinking, tools, "model family".into())
    }
}

fn scan_library_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 4 || out.len() >= 400 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_library_files(&path, depth + 1, out);
            continue;
        }
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_ascii_lowercase();
            if ext == "jns" || ext == "gguf" {
                out.push(path);
            }
        }
    }
}

/// Pick the value shown in the library's Quant column. A recorded download
/// wins, then the file/repository name, then the GGUF header's dominant tensor
/// type; `GGUF` is the container, not a quantization, so it never masks a real
/// value. Non-GGUF files keep the em dash when nothing is known.
fn choose_quant(
    record: Option<&str>,
    inferred: &str,
    header: Option<&str>,
    is_gguf: bool,
) -> String {
    record
        .filter(|value| is_specific_quant(value))
        .map(str::to_string)
        .or_else(|| is_specific_quant(inferred).then(|| inferred.to_string()))
        .or_else(|| {
            if !is_gguf {
                return None;
            }
            header
                .filter(|value| is_specific_quant(value))
                .map(str::to_string)
        })
        .unwrap_or_else(|| if is_gguf { "GGUF".into() } else { "—".into() })
}

/// Reading every GGUF header in the models folder is real disk work, so the
/// library scan runs on the blocking pool instead of the IPC thread.
#[tauri::command]
pub async fn library_models(app: AppHandle) -> Vec<LibraryModel> {
    let store = app.state::<std::sync::Arc<SettingsStore>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || library_models_snapshot(&store))
        .await
        .unwrap_or_default()
}

pub(crate) fn library_models_snapshot(store: &SettingsStore) -> Vec<LibraryModel> {
    let settings = store.get();
    let root = expand_home(&settings.models_dir);
    let mut files = Vec::new();
    scan_library_files(&root, 0, &mut files);

    let downloads = read_download_manifest(&root);
    let hardware = crate::fit::hardware_profile();
    let mut models: Vec<LibraryModel> = files
        .into_iter()
        .map(|path| {
            let name = path
                .file_stem()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".into());
            let record = downloads.iter().find(|record| {
                record
                    .files
                    .iter()
                    .any(|file| path.ends_with(file.replace('/', std::path::MAIN_SEPARATOR_STR)))
            });
            let format = path
                .extension()
                .map(|ext| ext.to_string_lossy().to_ascii_uppercase())
                .unwrap_or_else(|| "—".into());
            let features = if format == "GGUF" {
                crate::gguf::features(path.to_string_lossy().as_ref())
            } else {
                None
            };
            let haystack = record
                .map(|value| value.repo.clone())
                .or_else(|| features.as_ref().and_then(|value| value.name.clone()))
                .unwrap_or_else(|| name.clone());
            let (thinking, tools, capabilities_source) =
                detect_capabilities(features.as_ref(), &haystack);
            let size_bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            // The manifest only names a quantization for files this app
            // downloaded, and the repository name often omits it. A GGUF
            // copied in by hand therefore used to show "—" even though its
            // header names the dominant tensor type. Prefer, in order: the
            // recorded download, the file/repository name, then the header.
            let header_quant = if format == "GGUF" {
                crate::companions::inspect_gguf_metadata(path.to_string_lossy().as_ref())
                    .ok()
                    .map(|info| info.quant)
            } else {
                None
            };
            let inferred = infer_quant(&format!(
                "{} {} {}",
                name,
                haystack,
                path.to_string_lossy()
            ));
            let quant = choose_quant(
                record.map(|value| value.quant.as_str()),
                &inferred,
                header_quant.as_deref(),
                format == "GGUF",
            );
            let fit = crate::fit::estimate_model_fit(
                size_bytes,
                features.as_ref().and_then(|value| value.layer_count),
                if format == "GGUF" {
                    settings.llama_ctx_size
                } else {
                    settings.ctx_size
                },
                &hardware,
            );
            LibraryModel {
                name,
                size_bytes,
                path: path.to_string_lossy().to_string(),
                format,
                quant,
                architecture: infer_architecture(&haystack),
                parameters: infer_parameters(&haystack),
                publisher: record
                    .and_then(|value| value.repo.split('/').next().map(str::to_string))
                    .unwrap_or_else(|| "local".into()),
                repo: record.map(|value| value.repo.clone()),
                downloaded_at: record.map(|value| value.saved_at.clone()),
                thinking,
                tools,
                capabilities_source,
                duplicate: false,
                same_repo_count: 1,
                fit,
            }
        })
        .collect();

    let mut repo_counts: HashMap<String, usize> = HashMap::new();
    let mut copy_counts: HashMap<(String, u64), usize> = HashMap::new();
    for model in &models {
        if let Some(repo) = &model.repo {
            *repo_counts.entry(repo.clone()).or_default() += 1;
        }
        *copy_counts
            .entry((model.name.to_lowercase(), model.size_bytes))
            .or_default() += 1;
    }
    for model in &mut models {
        model.same_repo_count = model
            .repo
            .as_ref()
            .and_then(|repo| repo_counts.get(repo).copied())
            .unwrap_or(1);
        model.duplicate = copy_counts
            .get(&(model.name.to_lowercase(), model.size_bytes))
            .copied()
            .unwrap_or(1)
            > 1;
    }

    models.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    models
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DownloadRecord {
    pub repo: String,
    pub quant: String,
    pub files: Vec<String>,
    pub saved_at: String,
    #[serde(default)]
    pub size_bytes: u64,
    /// Expected SHA-256 for each repo-relative file, when Hugging Face's LFS
    /// metadata provides it.
    #[serde(default)]
    pub checksums: HashMap<String, String>,
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join(".mikrotitan").join("downloads.json")
}

/// Download records written before the rename live in `.janas`. They are still
/// read so existing files keep verifying, and the next record written moves
/// them into the MikroTitan folder.
fn legacy_manifest_path(root: &Path) -> PathBuf {
    root.join(".janas").join("downloads.json")
}

pub fn read_download_manifest(root: &Path) -> Vec<DownloadRecord> {
    let path = manifest_path(root);
    let path = if path.is_file() {
        path
    } else {
        legacy_manifest_path(root)
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Vec<DownloadRecord>>(&raw).ok())
        .unwrap_or_default()
}

pub fn record_download(root: &Path, record: DownloadRecord) {
    let mut records = read_download_manifest(root);
    if let Some(existing) = records
        .iter_mut()
        .find(|existing| existing.repo == record.repo && existing.quant == record.quant)
    {
        *existing = record;
    } else {
        records.push(record);
    }
    let path = manifest_path(root);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&records) {
        let _ = fs::write(path, json);
    }
}

/// Verify a local model against the checksum recorded when it was downloaded.
#[tauri::command]
pub async fn verify_local_model(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    path: String,
) -> AppResult<ModelVerification> {
    let settings = store.get();
    let target = validate_model_file(&settings, &path)?;
    let size_bytes = fs::metadata(&target)?.len();
    let manifest = read_download_manifest(&expand_home(&settings.models_dir));
    let relative = manifest.iter().find_map(|record| {
        record
            .files
            .iter()
            .find(|file| target.ends_with(file.replace('/', std::path::MAIN_SEPARATOR_STR)))
            .map(|file| (record, file.clone()))
    });
    let expected = relative
        .as_ref()
        .and_then(|(record, file)| record.checksums.get(file).cloned());
    let hash_path = target.clone();
    let actual = tauri::async_runtime::spawn_blocking(move || sha256_file(&hash_path))
        .await
        .map_err(|err| AppError::message(format!("Could not hash that model: {err}")))??;
    let (ok, message) = match expected.as_deref() {
        Some(expected) if !expected.eq_ignore_ascii_case(&actual) => (
            false,
            "Checksum mismatch. This file may be corrupt or was not downloaded by MikroTitan."
                .to_string(),
        ),
        Some(_) => (true, "Verified against the download checksum.".to_string()),
        None => (
            true,
            "No upstream checksum is recorded; the SHA-256 is shown for comparison.".to_string(),
        ),
    };
    Ok(ModelVerification {
        path: target.to_string_lossy().to_string(),
        size_bytes,
        sha256: actual,
        expected,
        ok,
        message,
    })
}

#[tauri::command]
pub fn delete_local_model(
    store: State<'_, std::sync::Arc<SettingsStore>>,
    path: String,
) -> AppResult<u64> {
    let settings = store.get();
    let target = validate_model_file(&settings, &path)?;
    let active_paths = [
        settings.model_path.as_str(),
        settings.llama_model_path.as_str(),
        settings.llama_draft_path.as_str(),
        settings.llama_mtp_path.as_str(),
        settings.llama_mmproj_path.as_str(),
        settings.draft_path.as_str(),
    ];
    let active_main = |main: &str| {
        !main.trim().is_empty()
            && expand_home(main).canonicalize().ok().as_deref() == Some(target.as_path())
    };
    let active = active_paths
        .iter()
        .filter(|value| !value.trim().is_empty())
        .filter_map(|value| expand_home(value).canonicalize().ok())
        .any(|active| active == target);
    if active {
        return Err(AppError::message(
            "That model is active. Select another model before deleting it.",
        ));
    }

    // A model can also be bound as a companion by the active bundle while the
    // bundle's main model is the one running. Deleting it would break the next
    // launch, so refuse until the user explicitly unpairs it.
    let bound_to_active_bundle = settings.model_bundles.iter().any(|bundle| {
        let bundle_is_active = active_main(&bundle.main_model_path)
            || (!settings.active_bundle_id.is_empty()
                && bundle.id == settings.active_bundle_id
                && active_main(&settings.llama_model_path));
        bundle_is_active
            && [bundle.mmproj_path.as_str(), bundle.draft_model_path.as_str()]
                .iter()
                .filter(|value| !value.trim().is_empty())
                .filter_map(|value| expand_home(value).canonicalize().ok())
                .any(|bound| bound == target)
    });
    if bound_to_active_bundle {
        return Err(AppError::message(
            "That file is bound to the active model bundle. Unpair the projector or draft model in the Library before deleting it.",
        ));
    }
    let bytes = fs::metadata(&target)?.len();
    fs::remove_file(&target)?;
    Ok(bytes)
}

#[tauri::command]
pub fn library_disk(store: State<'_, std::sync::Arc<SettingsStore>>) -> LibraryDisk {
    let settings = store.get();
    let root = expand_home(&settings.models_dir);
    let mut files = Vec::new();
    scan_library_files(&root, 0, &mut files);
    let total_bytes = files
        .iter()
        .filter_map(|path| fs::metadata(path).ok())
        .map(|meta| meta.len())
        .sum();
    LibraryDisk {
        models_dir: root.to_string_lossy().to_string(),
        total_bytes,
        free_bytes: free_disk_bytes(&root),
    }
}

fn validate_model_file(settings: &AppSettings, path: &str) -> AppResult<PathBuf> {
    let root = expand_home(&settings.models_dir)
        .canonicalize()
        .map_err(|_| AppError::message("Set an existing models folder in Settings first."))?;
    let target = expand_home(path)
        .canonicalize()
        .map_err(|err| AppError::message(format!("Could not find that model: {err}")))?;
    if !target.starts_with(&root) {
        return Err(AppError::message(
            "Only models inside the configured models folder can be managed here.",
        ));
    }
    let extension = target
        .extension()
        .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if extension != "gguf" && extension != "jns" {
        return Err(AppError::message("Only .gguf and .jns model files can be managed."));
    }
    if !target.is_file() {
        return Err(AppError::message("That model path is not a file."));
    }
    Ok(target)
}

fn sha256_file(path: &Path) -> AppResult<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read as _;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(unix)]
fn free_disk_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } == 0 {
        Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
    } else {
        None
    }
}

#[cfg(not(unix))]
fn free_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------
// External source links
// ---------------------------------------------------------------------------

/// Open a citation in the user's real browser. The webview cannot navigate
/// away from the app, and target="_blank" links are silently ignored there.
#[tauri::command]
pub fn open_external(url: String) -> AppResult<()> {
    let target = validate_external_url(&url)?;
    launch_external(&target)
}

fn validate_external_url(input: &str) -> AppResult<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(AppError::message("The source link is empty."));
    }
    if trimmed.chars().count() > 4_096 {
        return Err(AppError::message("The source link is too long to open."));
    }
    let parsed = reqwest::Url::parse(trimmed)
        .map_err(|err| AppError::message(format!("That source link is not a valid URL: {err}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(AppError::message(
            "Only http and https source links can be opened.",
        ));
    }
    if parsed.host_str().map(str::is_empty).unwrap_or(true) {
        return Err(AppError::message("The source link has no host."));
    }
    Ok(parsed.to_string())
}

#[cfg(target_os = "linux")]
fn launch_external(url: &str) -> AppResult<()> {
    for program in ["xdg-open", "gio", "sensible-browser"] {
        let mut command = Command::new(program);
        if program == "gio" {
            command.arg("open");
        }
        command
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if command.spawn().is_ok() {
            return Ok(());
        }
    }
    Err(AppError::message(
        "No system URL handler was found. Install xdg-utils or open the source manually.",
    ))
}

#[cfg(target_os = "macos")]
fn launch_external(url: &str) -> AppResult<()> {
    Command::new("open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|err| AppError::message(format!("Could not open the source: {err}")))
}

#[cfg(target_os = "windows")]
fn launch_external(url: &str) -> AppResult<()> {
    Command::new("cmd")
        .args(["/C", "start", ""])
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|err| AppError::message(format!("Could not open the source: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_memory_across_devices() {
        let (total, free) = parse_nvidia_smi_memory("24576, 512\n24576, 1024\n");
        assert_eq!(total, 2 * 24_576 * 1024 * 1024);
        assert_eq!(free, (24_576 * 2 - 1_536) * 1024 * 1024);
    }

    #[test]
    fn nvidia_smi_memory_skips_unsupported_counters() {
        let (total, free) = parse_nvidia_smi_memory("[N/A], [N/A]\n12288, 4096\n");
        assert_eq!(total, 12_288 * 1024 * 1024);
        assert_eq!(free, (12_288 - 4_096) * 1024 * 1024);

        // A driver that hides both counters must yield "unknown", not "no GPU".
        assert_eq!(parse_nvidia_smi_memory("[Not Supported], [N/A]"), (0, 0));
        // A truncated row must not be counted at all.
        assert_eq!(parse_nvidia_smi_memory("8192"), (0, 0));
    }

    fn vram(vendor: &str, total_mib: u64, used_mib: u64) -> VramDevice {
        const MIB: u64 = 1024 * 1024;
        VramDevice {
            vendor: vendor.into(),
            total_bytes: total_mib * MIB,
            used_bytes: used_mib * MIB,
        }
    }

    #[test]
    fn cuda_preference_picks_the_discrete_card_on_a_hybrid_laptop() {
        const MIB: u64 = 1024 * 1024;
        let devices = [vram("intel", 512, 64), vram("nvidia", 8_188, 1_024)];

        assert_eq!(
            select_vram(&devices, VramPreference::Cuda),
            (8_188 * MIB, (8_188 - 1_024) * MIB)
        );
        // "Any" must not sum the carve-out into the discrete card either.
        assert_eq!(
            select_vram(&devices, VramPreference::Any).0,
            8_188 * MIB
        );
    }

    #[test]
    fn a_backend_preference_never_borrows_another_vendors_memory() {
        let devices = [vram("intel", 512, 0)];
        assert_eq!(select_vram(&devices, VramPreference::Cuda), (0, 0));
        assert_eq!(select_vram(&devices, VramPreference::Rocm), (0, 0));
        assert_eq!(select_vram(&devices, VramPreference::Any).0, 512 * 1024 * 1024);
    }

    #[test]
    fn rocm_sums_matching_cards_only() {
        const MIB: u64 = 1024 * 1024;
        let devices = [
            vram("intel", 512, 0),
            vram("amd", 8_188, 188),
            vram("amd", 8_188, 188),
        ];
        assert_eq!(
            select_vram(&devices, VramPreference::Rocm),
            (2 * 8_188 * MIB, 2 * (8_188 - 188) * MIB)
        );
    }

    #[test]
    fn sysfs_scan_finds_the_discrete_card_on_a_hybrid_laptop() {
        let root = std::env::temp_dir().join(format!("mt-drm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let igpu = root.join("card0/device");
        fs::create_dir_all(&igpu).unwrap();
        fs::write(igpu.join("vendor"), "0x8086\n").unwrap();
        fs::write(igpu.join("mem_info_vram_total"), "536870912\n").unwrap();
        fs::write(igpu.join("mem_info_vram_used"), "5368709\n").unwrap();

        let dgpu = root.join("card1/device");
        fs::create_dir_all(&dgpu).unwrap();
        fs::write(dgpu.join("vendor"), "0x10de\n").unwrap();
        fs::write(dgpu.join("mem_info_vram_total"), "8589934592\n").unwrap();
        fs::write(dgpu.join("mem_info_vram_used"), "1073741824\n").unwrap();

        // Vendor only in uevent, as some drivers expose it.
        let amd = root.join("card2/device");
        fs::create_dir_all(&amd).unwrap();
        fs::write(amd.join("uevent"), "DRIVER=amdgpu\n").unwrap();
        fs::write(amd.join("mem_info_vram_total"), "4294967296\n").unwrap();

        // Connector entries and cards without a counter are not GPUs here.
        fs::create_dir_all(root.join("card1-DP-1")).unwrap();
        fs::create_dir_all(root.join("card3/device")).unwrap();

        let devices = vram_devices_in(&root);
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0].vendor, "intel");
        assert_eq!(devices[1].vendor, "nvidia");
        assert_eq!(devices[2].vendor, "amd");

        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(
            select_vram(&devices, VramPreference::Cuda),
            (8 * GIB, 7 * GIB)
        );
        // The iGPU carve-out is never substituted for a missing NVIDIA card.
        assert_eq!(select_vram(&devices[..1], VramPreference::Cuda), (0, 0));
        assert_eq!(select_vram(&devices, VramPreference::Rocm).0, 4 * GIB);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn library_quant_prefers_a_specific_value_and_falls_back_to_the_header() {
        // A hand-copied GGUF with a neutral name and no manifest record.
        assert_eq!(choose_quant(None, "—", Some("Q4_K"), true), "Q4_K");
        // Older manifests stored the container name; it must not hide the header.
        assert_eq!(choose_quant(Some("GGUF"), "—", Some("Q5_K"), true), "Q5_K");
        // The recorded download is the most exact source.
        assert_eq!(choose_quant(Some("Q4_K_M"), "—", Some("Q4_K"), true), "Q4_K_M");
        // The file name is more specific than the dominant tensor type.
        assert_eq!(choose_quant(None, "Q8_0", Some("Q4_K"), true), "Q8_0");
        // Nothing known at all.
        assert_eq!(choose_quant(None, "—", None, true), "GGUF");
        assert_eq!(choose_quant(None, "—", Some("Q4_K"), false), "—");
    }

    #[test]
    fn detects_thinking_and_tool_capabilities_from_templates_and_names() {
        let qwen_template =
            "{%- if tools %}<tool_call>{% endif %}{%- if enable_thinking %}reasoning{% endif %}";
        assert_eq!(template_capabilities(qwen_template), (true, true));

        let llama_template = "{%- if tools %}<|python_tag|>{% endif %}";
        assert_eq!(template_capabilities(llama_template), (false, true));

        assert_eq!(inferred_capabilities("Qwen3-0.6B-Q4_K_M"), (true, true));
        assert_eq!(inferred_capabilities("llama-2-7b"), (false, false));
    }

    #[test]
    fn decodes_duckduckgo_redirects_and_rejects_unsafe_urls() {
        let url = decode_href(
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1&rut=abc",
        )
        .unwrap();
        assert_eq!(url, "https://example.com/page?a=1");
        assert!(decode_href("javascript:alert(1)").is_none());
        assert!(decode_href("file:///etc/passwd").is_none());
        assert!(decode_href("/relative/path").is_none());
        assert!(decode_href("https://example.com/a?x=1&amp;y=2").is_some());
    }

    #[test]
    fn parses_results_from_anchors_not_css() {
        let html = r#"
          <style>.result__a { color: red } .result__snippet { font: serif }</style>
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&rut=x">Example docs</a>
          <a class="result__snippet" href="https://example.com/docs">A useful snippet.</a>
          <a class="result__a" href="https://rust-lang.org/">Rust</a>
        "#;
        let results = parse_search_results(html, "result__a");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "Example docs");
        assert_eq!(results[0].1, "https://example.com/docs");
        assert_eq!(results[0].2, "A useful snippet.");
        assert_eq!(results[1].1, "https://rust-lang.org/");
    }

    #[test]
    fn parses_bing_blocks() {
        let html = r#"
          <li class="b_algo"><h2><a href="https://example.org/story">Story title</a></h2>
          <div><p>A short summary of the story.</p></div></li>
        "#;
        let results = parse_bing_results(html);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Story title");
        assert_eq!(results[0].1, "https://example.org/story");
        assert_eq!(results[0].2, "A short summary of the story.");
    }

    #[test]
    fn validates_external_source_links() {
        assert!(validate_external_url("https://example.com/page").is_ok());
        assert!(validate_external_url("javascript:alert(1)").is_err());
        assert!(validate_external_url("file:///etc/passwd").is_err());
        assert!(validate_external_url("not a url").is_err());
    }

    #[test]
    fn hashes_model_files_with_sha256() {
        let path = std::env::temp_dir().join(format!(
            "janas-sha-{}.bin",
            uuid::Uuid::new_v4().simple()
        ));
        fs::write(&path, b"abc").expect("write fixture");
        assert_eq!(
            sha256_file(&path).expect("hash"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    #[ignore = "network-dependent smoke test"]
    async fn live_search_returns_openable_sources() {
        let response = web_search("Tauri 2 release notes".into(), Some(1))
            .await
            .expect("live search failed");
        assert!(!response.results.is_empty());
        assert!(response.results.iter().all(|result| {
            result.url.starts_with("http://") || result.url.starts_with("https://")
        }));
        assert!(response
            .results
            .iter()
            .all(|result| !result.title.eq_ignore_ascii_case("go")));
    }
}

