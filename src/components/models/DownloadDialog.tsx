import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Download,
  Eye,
  FolderTree,
  HardDrive,
  Pause,
  Play,
  Sparkles,
  X,
} from "lucide-react";

import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { errorMessage, formatBytes } from "../../lib/format";
import type {
  DownloadProgress,
  HfDraftHint,
  HfMmprojFile,
  HfModel,
  HfQuant,
  HardwareProfile,
  ModelFitEstimate,
} from "../../types";
import { Button } from "../ui/Button";
import { ProgressBar } from "../ui/ProgressBar";
import { FitBadge } from "./FitBadge";

interface DownloadDialogProps {
  model: HfModel;
  modelsDir: string;
  downloads: Record<string, DownloadProgress>;
  /** This machine's RAM/VRAM, fetched by the explorer on mount. */
  hardware: HardwareProfile | null;
  /** The llama.cpp context to estimate the KV cache at, when configured. */
  contextSize: number | null;
  onClose: () => void;
  onDownload: (
    quant: HfQuant,
    mmproj?: HfMmprojFile,
    autoCompile?: boolean,
  ) => Promise<string>;
  onPause: (id: string) => void;
  onResume: (id: string) => void;
  onCancel: (id: string) => void;
}

export function DownloadDialog({
  model,
  modelsDir,
  downloads,
  hardware,
  contextSize,
  onClose,
  onDownload,
  onPause,
  onResume,
  onCancel,
}: DownloadDialogProps) {
  const [quants, setQuants] = useState<HfQuant[]>(model.quants);
  const [selected, setSelected] = useState<string>(
    model.quants[0]?.quant ?? "",
  );
  const [projectors, setProjectors] = useState<HfMmprojFile[]>([]);
  const [selectedMmproj, setSelectedMmproj] = useState<string | null>(null);
  const [includeMmproj, setIncludeMmproj] = useState(false);
  const [draftHint, setDraftHint] = useState<HfDraftHint | null>(null);
  const [loading, setLoading] = useState(model.quants.length === 0);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadId, setDownloadId] = useState<string | null>(null);
  const [freeBytes, setFreeBytes] = useState<number | null>(null);
  const [fits, setFits] = useState<Record<string, ModelFitEstimate>>({});
  const [autoCompile, setAutoCompile] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .libraryDisk()
      .then((disk) => {
        if (!cancelled) setFreeBytes(disk.free_bytes);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .inspectRepoBundles(model.id)
      .then((bundles) => {
        if (cancelled) return;
        setQuants(bundles.quants);
        if (!bundles.quants.some((quant) => quant.quant === selected)) {
          setSelected(bundles.quants[0]?.quant ?? "");
        }
        setProjectors(bundles.available_mmproj);
        setDraftHint(bundles.draft_hint);
        const recommended =
          bundles.recommended_mmproj ?? bundles.available_mmproj[0]?.path ?? null;
        setSelectedMmproj(recommended);
        setIncludeMmproj(Boolean(recommended));
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // Only the repository identity should trigger a refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.id]);

  // Every quantization row gets a hardware verdict. The Hub metadata has no
  // GGUF header, so the backend infers a block count from `parameters`.
  useEffect(() => {
    if (!quants.length) {
      setFits({});
      return;
    }
    let cancelled = false;
    void Promise.all(
      quants.map(async (quant) => {
        try {
          const estimate = await api.estimateQuantFit(
            quant.size_bytes,
            null,
            contextSize,
            model.parameters,
          );
          return [quant.quant, estimate] as const;
        } catch {
          return null;
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      setFits(
        Object.fromEntries(
          entries.filter(
            (entry): entry is readonly [string, ModelFitEstimate] =>
              entry !== null,
          ),
        ),
      );
    });
    return () => {
      cancelled = true;
    };
  }, [quants, model.parameters, contextSize]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !starting) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, starting]);

  const active = useMemo(
    () => quants.find((quant) => quant.quant === selected) ?? quants[0],
    [quants, selected],
  );
  const mmproj = useMemo(
    () => projectors.find((projector) => projector.path === selectedMmproj),
    [projectors, selectedMmproj],
  );
  const chosenMmproj = includeMmproj ? mmproj : undefined;
  const combinedBytes =
    (active?.size_bytes ?? 0) + (chosenMmproj?.size ?? 0);
  const activeFit = active ? fits[active.quant] ?? null : null;
  // Both tiers above the RAM budget need Janas SSD streaming; the slow one is
  // the same situation with a warning attached.
  const exceedsRam =
    activeFit?.tier === "requires_ssd_streaming" ||
    activeFit?.tier === "tight_or_slow";

  const job = downloadId ? downloads[downloadId] : undefined;
  const running = job ? ["preparing", "downloading"].includes(job.status) : false;
  const paused = job?.status === "paused";
  const finished = job?.status === "complete";

  const destination = `${modelsDir.replace(/\/+$/, "")}/${model.id}`;
  const insufficient =
    active != null && freeBytes != null && combinedBytes > freeBytes;

  const start = async () => {
    if (!active) return;
    setStarting(true);
    setError(null);
    try {
      const id = await onDownload(
        active,
        chosenMmproj,
        autoCompile && exceedsRam,
      );
      setDownloadId(id);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setStarting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 p-6 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={`Download ${model.name}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !starting) onClose();
      }}
    >
      <div className="panel w-full max-w-xl overflow-hidden bg-ink-900 shadow-panel animate-fade-in">
        <div className="flex items-start justify-between gap-4 border-b border-white/5 px-5 py-4">
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-[0.14em] text-slate-500">
              {model.author}
            </p>
            <h2 className="mt-1 truncate text-base font-semibold text-white">
              {model.name}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="max-h-[70vh] space-y-4 overflow-y-auto px-5 py-4">
          {job ? (
            <div className="rounded-xl border border-white/10 bg-ink-950/60 p-3.5">
              <div className="flex items-center justify-between gap-2">
                <p className="truncate text-[11px] font-medium text-slate-200">
                  {job.repo}
                </p>
                <span
                  className={cn(
                    "text-[9px] uppercase tracking-wider",
                    paused
                      ? "text-amber-300"
                      : finished
                        ? "text-emerald-300"
                        : job.status === "error"
                          ? "text-rose-300"
                          : "text-acid-300",
                  )}
                >
                  {job.status}
                </span>
              </div>
              <p className="mt-0.5 truncate text-[10px] text-slate-600">
                {job.message}
              </p>
              <div className="mt-2 flex items-center gap-2">
                <ProgressBar
                  value={job.percent}
                  tone={
                    job.status === "error"
                      ? "rose"
                      : paused
                        ? "sky"
                        : "acid"
                  }
                  className="flex-1"
                />
                <span className="w-20 text-right font-mono text-[9px] text-slate-500">
                  {formatBytes(job.downloaded)} / {formatBytes(job.total)}
                </span>
              </div>
              <div className="mt-2.5 flex items-center gap-2">
                {running || paused ? (
                  <>
                    <Button
                      size="sm"
                      variant={paused ? "primary" : "secondary"}
                      onClick={() =>
                        paused ? onResume(job.id) : onPause(job.id)
                      }
                      icon={
                        paused ? (
                          <Play className="h-3 w-3" />
                        ) : (
                          <Pause className="h-3 w-3" />
                        )
                      }
                    >
                      {paused ? "Resume" : "Pause"}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => onCancel(job.id)}
                    >
                      Cancel
                    </Button>
                  </>
                ) : null}
                {finished ? (
                  <p className="text-[10px] leading-4 text-emerald-300/80">
                    {job?.message || "Download complete."}
                  </p>
                ) : null}
                {job.status === "error" ? (
                  <p className="text-[10px] leading-4 text-rose-300/80">
                    {job.message}
                  </p>
                ) : null}
              </div>
            </div>
          ) : null}

          {!job ? (
            <>
              <div className="flex items-center justify-between">
                <span className="label">Quantization</span>
                {loading ? (
                  <span className="text-[10px] text-slate-500">
                    inspecting the repository tree…
                  </span>
                ) : (
                  <span className="text-[10px] text-slate-500">
                    {quants.length} main build{quants.length === 1 ? "" : "s"}
                  </span>
                )}
              </div>

              {error ? (
                <div className="flex items-start gap-2 rounded-xl border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[10px] leading-4 text-amber-200">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{error}</span>
                </div>
              ) : null}

              <div className="max-h-48 space-y-1.5 overflow-y-auto pr-1">
                {quants.length ? (
                  quants.map((quant) => (
                    <button
                      key={quant.quant}
                      type="button"
                      onClick={() => setSelected(quant.quant)}
                      className={cn(
                        "flex w-full items-center justify-between gap-3 rounded-xl border px-3 py-2.5 text-left transition",
                        active?.quant === quant.quant
                          ? "border-acid-400/40 bg-acid-400/5"
                          : "border-white/10 bg-white/[0.02] hover:bg-white/5",
                      )}
                    >
                      <div className="min-w-0">
                        <p className="text-xs font-medium text-slate-200">
                          {quant.quant}
                        </p>
                        <p className="text-[10px] text-slate-500">
                          {quant.files.length} file
                          {quant.files.length === 1 ? "" : "s"}
                        </p>
                      </div>
                      <div className="flex shrink-0 flex-col items-end gap-1">
                        <span className="font-mono text-[10px] text-slate-400">
                          {formatBytes(quant.size_bytes)}
                        </span>
                        <FitBadge fit={fits[quant.quant] ?? null} />
                      </div>
                    </button>
                  ))
                ) : (
                  <p className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-6 text-center text-[11px] text-slate-500">
                    {loading
                      ? "Looking for GGUF files…"
                      : "This repository has no convertible GGUF files."}
                  </p>
                )}
              </div>

              <div className="space-y-2 rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <div className="flex items-start justify-between gap-3">
                  <label
                    className={cn(
                      "flex min-w-0 items-start gap-2",
                      !projectors.length && "opacity-50",
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={includeMmproj && projectors.length > 0}
                      disabled={!projectors.length}
                      onChange={(event) => setIncludeMmproj(event.target.checked)}
                      className="mt-0.5 h-3.5 w-3.5 accent-current text-acid-400"
                    />
                    <span className="min-w-0">
                      <span className="flex items-center gap-1.5 text-[11px] font-medium text-slate-200">
                        <Eye className="h-3 w-3 text-sky-300" />
                        Include Vision Projector (mmproj)
                      </span>
                      <span className="mt-0.5 block text-[9.5px] leading-4 text-slate-500">
                        {projectors.length
                          ? `${projectors.length} projector${projectors.length === 1 ? "" : "s"} in this repo. f16 or bf16 keeps the vision tower at reference quality.`
                          : "This repository has no mmproj file. Image input will stay unavailable for this model."}
                      </span>
                    </span>
                  </label>
                  {chosenMmproj ? (
                    <span className="shrink-0 font-mono text-[10px] text-slate-400">
                      +{formatBytes(chosenMmproj.size)}
                    </span>
                  ) : null}
                </div>

                {projectors.length ? (
                  <select
                    value={selectedMmproj ?? ""}
                    onChange={(event) => setSelectedMmproj(event.target.value)}
                    disabled={!includeMmproj}
                    className="field !py-1.5 font-mono text-[10.5px] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {projectors.map((projector) => (
                      <option key={projector.path} value={projector.path}>
                        {projector.path} · {projector.quant} ·{" "}
                        {formatBytes(projector.size)}
                        {projector.recommended ? " · recommended" : ""}
                      </option>
                    ))}
                  </select>
                ) : null}

                {draftHint ? (
                  <div className="flex items-start gap-2 rounded-lg border border-violet-400/15 bg-violet-400/5 px-2.5 py-2 text-[9.5px] leading-4 text-violet-200/80">
                    <Sparkles className="mt-0.5 h-3 w-3 shrink-0" />
                    <span>
                      <span className="font-medium text-violet-200">
                        {draftHint.size_label} draft hint:
                      </span>{" "}
                      {draftHint.reason} Download{" "}
                      <span className="font-mono text-violet-200/90">
                        {draftHint.repo_id}
                      </span>{" "}
                      separately and pair it in Library.
                    </span>
                  </div>
                ) : null}
              </div>

              {exceedsRam ? (
                <label className="flex items-start gap-2 rounded-xl border border-amber-400/25 bg-amber-400/5 px-3 py-2.5">
                  <input
                    type="checkbox"
                    checked={autoCompile}
                    onChange={(event) => setAutoCompile(event.target.checked)}
                    className="mt-0.5 h-3.5 w-3.5 accent-current text-amber-300"
                  />
                  <span className="min-w-0">
                    <span className="flex items-center gap-1.5 text-[11px] font-medium text-amber-200">
                      <HardDrive className="h-3 w-3" />
                      Auto-compile to .jns after download for Janas SSD streaming
                    </span>
                    <span className="mt-0.5 block text-[9.5px] leading-4 text-slate-500">
                      This build needs {formatBytes(activeFit?.total_needed_bytes ?? 0)} of
                      working memory and exceeds system RAM. When the download
                      finishes it is compiled with the Janas converter so
                      janas-server can stream it from the SSD.
                    </span>
                  </span>
                </label>
              ) : null}

              {active ? (
                <>
                  <div className="rounded-xl border border-white/5 bg-ink-950/50 px-3 py-2.5">
                    <div className="flex items-center gap-2 text-[10px] text-slate-500">
                      <FolderTree className="h-3 w-3" />
                      <span className="truncate">{destination}</span>
                    </div>
                    <div className="mt-2 max-h-20 overflow-y-auto font-mono text-[9px] leading-4 text-slate-600">
                      {active.files.map((file) => (
                        <p key={file.path} className="truncate">
                          {file.path}{" "}
                          <span className="text-slate-700">
                            ({formatBytes(file.size)})
                          </span>
                        </p>
                      ))}
                      {chosenMmproj ? (
                        <p className="truncate text-sky-300/60">
                          {chosenMmproj.path}{" "}
                          <span className="text-slate-700">
                            ({formatBytes(chosenMmproj.size)})
                          </span>
                        </p>
                      ) : null}
                    </div>
                  </div>

                  {insufficient ? (
                    <div className="flex items-start gap-2 rounded-xl border border-rose-400/20 bg-rose-500/5 px-3 py-2 text-[10px] leading-4 text-rose-300">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      <span>
                        This download needs {formatBytes(combinedBytes)} but only{" "}
                        {formatBytes(freeBytes ?? 0)} is free in the models
                        folder. Free space before downloading.
                      </span>
                    </div>
                  ) : freeBytes != null ? (
                    <p className="px-1 text-[9px] text-slate-600">
                      {formatBytes(freeBytes)} free in the models folder.
                    </p>
                  ) : null}

                  <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2.5">
                    <div className="flex items-center justify-between">
                      <p className="label">Memory estimate</p>
                      <span className="font-mono text-[10px] text-slate-400">
                        {formatBytes(combinedBytes)} on disk
                      </span>
                    </div>
                    {activeFit ? (
                      <div className="mt-2 flex items-start gap-2">
                        <FitBadge fit={activeFit} />
                        <p className="text-[9.5px] leading-4 text-slate-500">
                          {activeFit.reason}
                        </p>
                      </div>
                    ) : null}
                    {hardware ? (
                      <p className="mt-1.5 text-[9px] leading-4 text-slate-600">
                        Detected:{" "}
                        {hardware.vram_total_bytes > 0
                          ? `${formatBytes(hardware.vram_total_bytes)} VRAM`
                          : hardware.gpu_backend === "none"
                            ? "no GPU"
                            : "VRAM unknown"}
                        {` · ${formatBytes(hardware.ram_total_bytes)} RAM`}
                        {hardware.storage_mb_s != null
                          ? ` · ${Math.round(hardware.storage_mb_s)} MB/s storage`
                          : ""}
                      </p>
                    ) : null}
                    {activeFit ? null : (
                      <>
                        <div className="mt-2 grid grid-cols-2 gap-2">
                          <MemoryStat
                            label="System RAM"
                            value={formatBytes(
                              combinedBytes * 1.15 + 1.5 * 1024 ** 3,
                            )}
                            hint="working set at load"
                          />
                          <MemoryStat
                            label="VRAM"
                            value={formatBytes(combinedBytes * 1.05)}
                            hint="if fully offloaded"
                          />
                        </div>
                        <p className="mt-1.5 text-[9px] leading-4 text-slate-600">
                          Rough estimate: weights
                          {chosenMmproj ? ", projector" : ""} and a 1.5 GiB
                          margin for the KV cache and runtime. A
                          mixture-of-experts model streams unused experts from
                          disk, so resident RAM can be far lower.
                        </p>
                      </>
                    )}
                  </div>
                </>
              ) : null}

              {starting ? <ProgressBar value={null} /> : null}
            </>
          ) : null}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-white/5 bg-ink-850/60 px-5 py-3.5">
          <p className="text-[10px] text-slate-600">
            Pause keeps the stream open; resume continues at the same byte.
          </p>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={onClose}>
              {job && !finished ? "Run in background" : "Close"}
            </Button>
            {!job || job.status === "error" ? (
              <Button
                size="sm"
                variant="primary"
                busy={starting}
                disabled={!active || insufficient}
                onClick={() => void start()}
                icon={<Download className="h-3.5 w-3.5" />}
              >
                {active
                  ? `Download ${formatBytes(combinedBytes)}`
                  : "Download"}
              </Button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function MemoryStat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-ink-950/60 px-2.5 py-1.5">
      <p className="text-[9px] uppercase tracking-wider text-slate-600">
        {label}
      </p>
      <p className="mt-0.5 font-mono text-[11px] text-slate-300">{value}</p>
      <p className="text-[8px] text-slate-700">{hint}</p>
    </div>
  );
}
