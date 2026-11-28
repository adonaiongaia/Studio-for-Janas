import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Boxes,
  Brain,
  Cpu,
  Eye,
  HardDrive,
  Link2,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Trash2,
  TriangleAlert,
  Wrench,
  type LucideIcon,
} from "lucide-react";

import { api } from "../../lib/api";
import { ask } from "@tauri-apps/plugin-dialog";
import { cn } from "../../lib/cn";
import { errorMessage, formatBytes, formatRelativeDate } from "../../lib/format";
import type {
  CompanionPairing,
  LibraryCompanions,
  LibraryDisk,
  LibraryModel,
  ModelVerification,
} from "../../types";
import { Button } from "../ui/Button";
import { CompanionDrawer } from "./CompanionDrawer";
import { FitBadge } from "./FitBadge";

interface ModelLibraryViewProps {
  onUseModel: (path: string) => Promise<void>;
}

export function ModelLibraryView({ onUseModel }: ModelLibraryViewProps) {
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [using, setUsing] = useState<string | null>(null);
  const [disk, setDisk] = useState<LibraryDisk | null>(null);
  const [verifying, setVerifying] = useState<string | null>(null);
  const [verifications, setVerifications] = useState<
    Record<string, ModelVerification>
  >({});
  const [deleting, setDeleting] = useState<string | null>(null);
  const [companions, setCompanions] = useState<LibraryCompanions | null>(
    null,
  );
  const [pairing, setPairing] = useState<CompanionPairing | null>(null);
  const [quickPairing, setQuickPairing] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [nextModels, nextDisk, nextCompanions] = await Promise.all([
        api.libraryModels(),
        api.libraryDisk(),
        api.libraryCompanions().catch(() => null),
      ]);
      setModels(nextModels);
      setDisk(nextDisk);
      setCompanions(nextCompanions);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const verify = async (model: LibraryModel) => {
    setVerifying(model.path);
    setError(null);
    try {
      const result = await api.verifyLocalModel(model.path);
      setVerifications((previous) => ({
        ...previous,
        [model.path]: result,
      }));
      if (!result.ok) setError(result.message);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setVerifying(null);
    }
  };

  const remove = async (model: LibraryModel) => {
    const confirmed = await ask(
      `Delete “${model.name}”?\n\nThis removes ${formatBytes(
        model.size_bytes,
      )} from ${model.path}. It cannot be undone.`,
      { title: "Delete model", kind: "warning" },
    );
    if (!confirmed) return;
    setDeleting(model.path);
    setError(null);
    try {
      await api.deleteLocalModel(model.path);
      setVerifications((previous) => {
        const next = { ...previous };
        delete next[model.path];
        return next;
      });
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setDeleting(null);
    }
  };

  const companionByPath = useMemo(() => {
    const map = new Map<string, LibraryCompanions["models"][number]>();
    for (const info of companions?.models ?? []) map.set(info.path, info);
    return map;
  }, [companions]);

  const pairingByMain = useMemo(() => {
    const map = new Map<string, CompanionPairing>();
    for (const item of companions?.pairings ?? []) {
      map.set(item.main.path, item);
    }
    return map;
  }, [companions]);

  const quickPair = async (item: CompanionPairing) => {
    const candidate = item.vision.recommended;
    if (!candidate) {
      setPairing(item);
      return;
    }
    setQuickPairing(item.main.path);
    setError(null);
    try {
      await api.pairModelCompanions(
        item.main.path,
        candidate.path,
        item.draft.bound_path ?? item.draft.recommended?.path ?? null,
        8,
      );
      await onUseModel(item.main.path);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setQuickPairing(null);
    }
  };

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return models;
    return models.filter((model) =>
      `${model.name} ${model.architecture} ${model.quant} ${model.publisher} ${
        model.repo ?? ""
      } ${model.fit.label} ${model.thinking ? "thinking reasoning" : ""} ${
        model.tools ? "tools tool calling" : ""
      }`
        .toLowerCase()
        .includes(needle),
    );
  }, [models, query]);

  const totalBytes = models.reduce((sum, model) => sum + model.size_bytes, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 border-b border-white/5 px-6 py-5">
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="relative min-w-[260px] flex-1">
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-600" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter by name, architecture, publisher, quantization or capability…"
              className="field pl-10"
              spellCheck={false}
            />
          </div>
          <Button
            variant="secondary"
            onClick={() => void refresh()}
            icon={<RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />}
          >
            Rescan drives
          </Button>
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <Boxes className="h-3 w-3 text-acid-500/70" />
            {models.length} local model{models.length === 1 ? "" : "s"} ·{" "}
            {formatBytes(totalBytes)} on this machine
            {disk?.free_bytes != null
              ? ` · ${formatBytes(disk.free_bytes)} free`
              : ""}
          </span>
          <span className="inline-flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1 text-violet-300/80">
              <Brain className="h-2.5 w-2.5" /> thinking
            </span>
            <span className="inline-flex items-center gap-1 text-emerald-300/80">
              <Wrench className="h-2.5 w-2.5" /> tools
            </span>
            <span className="text-slate-700">·</span>
            <span>Architecture and capabilities use metadata and filenames.</span>
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
        {error ? (
          <div className="mb-4 rounded-xl border border-rose-400/20 bg-rose-500/5 px-4 py-3 text-[11px] text-rose-300">
            {error}
          </div>
        ) : null}

        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <div
                key={index}
                className="h-16 animate-pulse rounded-xl border border-white/5 bg-ink-900/60"
              />
            ))}
          </div>
        ) : filtered.length ? (
          <div className="overflow-hidden rounded-xl border border-white/5">
            <div className="grid grid-cols-[minmax(200px,2fr)_100px_90px_120px_110px_110px_210px] gap-3 border-b border-white/5 bg-white/[0.02] px-4 py-2 text-[9px] uppercase tracking-wider text-slate-600">
              <span>Model</span>
              <span>Architecture</span>
              <span>Params</span>
              <span>Publisher</span>
              <span>Quant</span>
              <span className="text-right">Size</span>
              <span className="text-right">Actions</span>
            </div>
            <div className="divide-y divide-white/5">
              {filtered.map((model) => (
                <div
                  key={model.path}
                  className="group grid grid-cols-[minmax(200px,2fr)_100px_90px_120px_110px_110px_210px] items-center gap-3 px-4 py-3 transition hover:bg-white/[0.02]"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "rounded border px-1 py-0.5 font-mono text-[8px] uppercase",
                          model.format === "JNS"
                            ? "border-acid-400/30 bg-acid-400/10 text-acid-300"
                            : "border-white/10 bg-white/5 text-slate-400",
                        )}
                      >
                        {model.format}
                      </span>
                      <p className="truncate text-[11.5px] text-slate-300">
                        {model.name}
                      </p>
                    </div>
                    <p className="mt-0.5 truncate text-[9px] text-slate-600">
                      {model.repo ?? model.path}
                      {model.downloaded_at
                        ? ` · ${formatRelativeDate(
                            new Date(
                              Number(model.downloaded_at) || model.downloaded_at,
                            ).toISOString(),
                          )}`
                        : ""}
                    </p>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1">
                      <FitBadge fit={model.fit} />
                    </div>
                    {model.thinking || model.tools ? (
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        {model.thinking ? (
                          <CapabilityBadge
                            icon={Brain}
                            label="Thinking"
                            source={model.capabilities_source}
                            tone="thinking"
                          />
                        ) : null}
                        {model.tools ? (
                          <CapabilityBadge
                            icon={Wrench}
                            label="Tools"
                            source={model.capabilities_source}
                            tone="tools"
                          />
                        ) : null}
                      </div>
                    ) : null}
                    <CompanionRowBadges
                      model={model}
                      info={companionByPath.get(model.path) ?? null}
                      pairing={pairingByMain.get(model.path) ?? null}
                      busy={quickPairing === model.path}
                      onQuickPair={() => {
                        const item = pairingByMain.get(model.path);
                        if (item) void quickPair(item);
                      }}
                      onOpenPairing={() => {
                        const item = pairingByMain.get(model.path);
                        if (item) setPairing(item);
                      }}
                    />
                    {model.duplicate ||
                    model.same_repo_count > 1 ||
                    verifications[model.path] ? (
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        {model.duplicate ? (
                          <span className="rounded border border-amber-400/20 bg-amber-400/5 px-1 py-0.5 text-[8px] text-amber-300/80">
                            duplicate
                          </span>
                        ) : null}
                        {model.same_repo_count > 1 ? (
                          <span className="rounded border border-white/10 bg-white/5 px-1 py-0.5 text-[8px] text-slate-500">
                            {model.same_repo_count} in repo
                          </span>
                        ) : null}
                        {verifications[model.path] ? (
                          <span
                            title={verifications[model.path].message}
                            className={cn(
                              "rounded border px-1 py-0.5 text-[8px]",
                              verifications[model.path].ok
                                ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-300/80"
                                : "border-rose-400/20 bg-rose-500/5 text-rose-300/80",
                            )}
                          >
                            {verifications[model.path].ok
                              ? verifications[model.path].expected
                                ? "checksum ok"
                                : "sha-256 only"
                              : "checksum mismatch"}
                          </span>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                  <span className="truncate text-[10.5px] text-slate-400">
                    {model.architecture}
                  </span>
                  <span className="font-mono text-[10.5px] text-slate-400">
                    {model.parameters}
                  </span>
                  <span className="truncate text-[10.5px] text-slate-500">
                    {model.publisher}
                  </span>
                  <span className="font-mono text-[10px] text-slate-400">
                    {model.quant}
                  </span>
                  <span className="flex items-center justify-end gap-1 font-mono text-[10px] text-slate-500">
                    <HardDrive className="h-3 w-3 text-slate-700" />
                    {formatBytes(model.size_bytes)}
                  </span>
                  <div className="flex items-center justify-end gap-1">
                    <button
                      type="button"
                      title="Verify this file against its download checksum"
                      onClick={() => void verify(model)}
                      disabled={verifying === model.path || deleting === model.path}
                      className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9px] text-slate-300 transition hover:bg-white/10 disabled:opacity-40"
                    >
                      {verifying === model.path ? (
                        <RefreshCw className="h-2.5 w-2.5 animate-spin" />
                      ) : (
                        <ShieldCheck className="h-2.5 w-2.5" />
                      )}
                      Verify
                    </button>
                    <button
                      type="button"
                      title="Delete this model file"
                      onClick={() => void remove(model)}
                      disabled={verifying === model.path || deleting === model.path}
                      className="inline-flex items-center gap-1 rounded-md border border-rose-400/15 bg-rose-500/5 px-1.5 py-0.5 text-[9px] text-rose-300/80 transition hover:bg-rose-500/10 disabled:opacity-40"
                    >
                      {deleting === model.path ? (
                        <RefreshCw className="h-2.5 w-2.5 animate-spin" />
                      ) : (
                        <Trash2 className="h-2.5 w-2.5" />
                      )}
                      Delete
                    </button>
                    {pairingByMain.has(model.path) ? (
                      <button
                        type="button"
                        title="Choose and test the projector and draft model for this bundle"
                        onClick={() => {
                          const item = pairingByMain.get(model.path);
                          if (item) setPairing(item);
                        }}
                        className="inline-flex items-center gap-1 rounded-md border border-acid-400/20 bg-acid-400/5 px-1.5 py-0.5 text-[9px] text-acid-200 transition hover:bg-acid-400/10"
                      >
                        <Link2 className="h-2.5 w-2.5" />
                        Pair
                      </button>
                    ) : null}
                    <button
                      type="button"
                      title={
                        model.format === "GGUF"
                          ? "Load this GGUF with llama.cpp"
                          : "Load this model with Janas"
                      }
                      onClick={async () => {
                        setUsing(model.path);
                        try {
                          await onUseModel(model.path);
                        } finally {
                          setUsing(null);
                        }
                      }}
                      className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9px] text-slate-300 transition hover:bg-white/10"
                    >
                      {using === model.path ? "…" : "Use"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center py-20 text-center">
            <Boxes className="h-7 w-7 text-slate-700" />
            <p className="mt-3 text-sm font-medium text-slate-300">
              No local models yet
            </p>
            <p className="mt-1 max-w-sm text-[11px] leading-5 text-slate-600">
              Download a GGUF from the Model Explorer, then compile it in the
              Compiler. Converted .jns files appear here with their
              architecture, publisher, size and quantization.
            </p>
          </div>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2 border-t border-white/5 bg-ink-900/30 px-6 py-2.5 text-[9.5px] text-slate-600">
        <Cpu className="h-3 w-3" />
        Scanned from your models folder. No manifest is uploaded anywhere.
      </div>

      {pairing ? (
        <CompanionDrawer
          pairing={pairing}
          models={companions?.models ?? []}
          onClose={() => setPairing(null)}
          onUse={onUseModel}
          onChanged={() => void refresh()}
        />
      ) : null}
    </div>
  );
}

/** Vision and speculation state for one library row, driven by header facts. */
function CompanionRowBadges({
  model,
  info,
  pairing,
  busy,
  onQuickPair,
  onOpenPairing,
}: {
  model: LibraryModel;
  info: LibraryCompanions["models"][number] | null;
  pairing: CompanionPairing | null;
  busy: boolean;
  onQuickPair: () => void;
  onOpenPairing: () => void;
}) {
  if (!info) return null;

  if (info.role === "mmproj") {
    return (
      <div className="mt-1 flex flex-wrap items-center gap-1">
        <span className="inline-flex items-center gap-1 rounded border border-sky-400/20 bg-sky-400/5 px-1.5 py-0.5 text-[8px] text-sky-300">
          <Eye className="h-2.5 w-2.5" /> Vision projector
        </span>
        <span className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-[8px] text-slate-500">
          {info.vision_family ?? "family unknown"}
        </span>
      </div>
    );
  }

  if (info.role === "draft_candidate") {
    return (
      <div className="mt-1 flex flex-wrap items-center gap-1">
        <span className="inline-flex items-center gap-1 rounded border border-violet-400/20 bg-violet-400/5 px-1.5 py-0.5 text-[8px] text-violet-300">
          <Sparkles className="h-2.5 w-2.5" /> Draft-sized
        </span>
        <span className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-[8px] text-slate-500">
          {info.architecture ?? model.architecture} · {info.vocab_size ?? "?"} vocab
        </span>
      </div>
    );
  }

  if (!pairing) return null;

  const visionReady = Boolean(pairing.vision.bound_path);
  const draftPaired = Boolean(pairing.draft.bound_path);

  return (
    <div className="mt-1 flex flex-wrap items-center gap-1">
      {info.role === "vision_main" ? (
        visionReady ? (
          <span className="inline-flex items-center gap-1 rounded border border-emerald-400/25 bg-emerald-400/10 px-1.5 py-0.5 text-[8px] text-emerald-300">
            <Eye className="h-2.5 w-2.5" /> Vision ready
          </span>
        ) : (
          <button
            type="button"
            onClick={onQuickPair}
            disabled={busy}
            title={pairing.vision.message}
            className="inline-flex items-center gap-1 rounded border border-amber-400/30 bg-amber-400/10 px-1.5 py-0.5 text-[8px] text-amber-200 transition hover:bg-amber-400/15 disabled:opacity-50"
          >
            {busy ? (
              <RefreshCw className="h-2.5 w-2.5 animate-spin" />
            ) : (
              <TriangleAlert className="h-2.5 w-2.5" />
            )}
            Missing MMPROJ · Pair
          </button>
        )
      ) : null}
      {draftPaired ? (
        <span className="inline-flex items-center gap-1 rounded border border-violet-400/25 bg-violet-400/10 px-1.5 py-0.5 text-[8px] text-violet-200">
          <Sparkles className="h-2.5 w-2.5" /> Draft paired (+Speculative)
        </span>
      ) : pairing.draft.recommended ? (
        <button
          type="button"
          onClick={onOpenPairing}
          title={pairing.draft.message}
          className="inline-flex items-center gap-1 rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[8px] text-slate-400 transition hover:text-violet-200"
        >
          <Sparkles className="h-2.5 w-2.5" /> Draft available
        </button>
      ) : null}
    </div>
  );
}

function CapabilityBadge({
  icon: Icon,
  label,
  source,
  tone,
}: {
  icon: LucideIcon;
  label: string;
  source: string;
  tone: "thinking" | "tools";
}) {
  return (
    <span
      title={`${label} support detected from ${source}.`}
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[8px] font-medium",
        tone === "thinking"
          ? "border-violet-400/20 bg-violet-400/10 text-violet-300"
          : "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
      )}
    >
      <Icon className="h-2.5 w-2.5" />
      {label}
    </span>
  );
}
