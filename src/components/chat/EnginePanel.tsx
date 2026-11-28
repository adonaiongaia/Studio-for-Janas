import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Cpu,
  FolderOpen,
  HardDrive,
  Loader2,
  MemoryStick,
  Play,
  Plus,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  X,
  Zap,
} from "lucide-react";

import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { basename, errorMessage, formatBytes } from "../../lib/format";
import {
  adapterWarningView,
  fallbackLabel,
} from "../../lib/llamaHardening";
import { shortcutLabel } from "../../lib/shortcuts";
import { useBackendCapabilities } from "../../hooks/useBackendCapabilities";
import { SAMPLING_PARAMETERS } from "../../lib/backendCapabilities";
import type {
  AdapterWarningEvent,
  AgentPreferences,
  AppSettings,
  ChatUsage,
  EngineKind,
  EnginePreferences,
  EngineStatus,
  EngineTelemetry,
  Persona,
} from "../../types";
import { Slider } from "../ui/Slider";
import { GpuOffloadControl } from "../ui/GpuOffloadControl";
import { Toggle } from "../ui/Toggle";
import { TokenTicker } from "./TokenTicker";

interface EnginePanelProps {
  collapsed: boolean;
  onToggle: () => void;
  preferences: EnginePreferences;
  onUpdate: <K extends keyof EnginePreferences>(
    key: K,
    value: EnginePreferences[K],
  ) => void;
  settings: AppSettings | null;
  activeEngine: EngineKind | null;
  engineTelemetry: EngineTelemetry | null;
  engineRunning: boolean;
  engineBusy: boolean;
  engineStatus: EngineStatus;
  /** Adapter defaults, GPU fallbacks and auto-rollbacks raised by the backend. */
  adapterWarning: AdapterWarningEvent | null;
  onDismissAdapterWarning: () => void;
  onOpenEngineSetup: () => void;
  onSwitchEngine: (engine: EngineKind) => Promise<void>;
  onApplyEngineSettings: (patch: Partial<AppSettings>) => Promise<void>;
  onRestartEngine: () => Promise<void>;
  onReset: (runtime: Partial<AppSettings>) => Promise<void>;
  agentPreferences: AgentPreferences;
  onAgentUpdate: <K extends keyof AgentPreferences>(
    key: K,
    value: AgentPreferences[K],
  ) => void;
  onPickWorkspace: () => Promise<void>;
  onClearWorkspace: () => void;
  workspaceLocked: boolean;
  systemPrompt: string;
  onSystemPromptChange: (value: string) => void;
  explicitArtifactTags: boolean;
  onExplicitArtifactTagsChange: (value: boolean) => void;
  usingDefaultPrompt: boolean;
  onUseDefaultPrompt: () => void;
  onMakeDefaultPrompt: () => void;
  personas: Persona[];
  onAddPersona: (name: string, prompt: string) => Persona | null;
  onDeletePersona: (id: string) => void;
  workspaceRoot: string;
  approvalRulesVersion: number;
  usage: ChatUsage | null;
  elapsedMs: number | null;
  firstTokenMs: number | null;
  liveBytes: number;
  streaming: boolean;
  startedAtMs: number | null;
}

interface EngineForm {
  ctx_size: number;
  cache_gib: number;
  mtp_mode: string;
  mtp_path: string;
  draft_path: string;
  prompt_on: string;
  reply_on: string;
  drafts_on: string;
  think: string;
  npu_int8: boolean;
  engine_mode: string;
  janas_attention: string;
  janas_kv_cache: string;
  llama_gpu_layers: number;
  llama_ctx_size: number;
  llama_mtp_mode: string;
  llama_mtp_path: string;
  llama_draft_path: string;
  llama_draft_max: number;
  llama_draft_min: number;
  llama_draft_p_min: number;
  llama_drafts_on: string;
  llama_flash_attn: string;
  llama_kv_cache_k: string;
  llama_kv_cache_v: string;
  llama_direct_cpu_moe: boolean;
  llama_cpu_threads: number;
  ssd_streaming: boolean;
  ssd_cache_dir: string;
}

function formFrom(settings: AppSettings | null): EngineForm {
  return {
    ctx_size: settings?.ctx_size ?? 16384,
    cache_gib: settings?.cache_gib ?? 0,
    mtp_mode: settings?.mtp_mode ?? "auto",
    mtp_path: settings?.mtp_path ?? "",
    draft_path: settings?.draft_path ?? "",
    prompt_on: settings?.prompt_on ?? "auto",
    reply_on: settings?.reply_on ?? "auto",
    drafts_on: settings?.drafts_on ?? "auto",
    think: settings?.think ?? "auto",
    npu_int8: settings?.npu_int8 ?? false,
    engine_mode: settings?.engine_mode ?? "auto",
    janas_attention: settings?.janas_attention ?? "auto",
    janas_kv_cache: settings?.janas_kv_cache ?? "auto",
    llama_gpu_layers: settings?.llama_gpu_layers ?? 0,
    llama_ctx_size: settings?.llama_ctx_size ?? 16384,
    llama_mtp_mode: settings?.llama_mtp_mode ?? "auto",
    llama_mtp_path: settings?.llama_mtp_path ?? "",
    llama_draft_path: settings?.llama_draft_path ?? "",
    llama_draft_max: settings?.llama_draft_max ?? 8,
    llama_draft_min: settings?.llama_draft_min ?? 0,
    llama_draft_p_min: settings?.llama_draft_p_min ?? 0.75,
    llama_drafts_on: settings?.llama_drafts_on ?? "auto",
    llama_flash_attn: settings?.llama_flash_attn ?? "auto",
    llama_kv_cache_k: settings?.llama_kv_cache_k ?? "f16",
    llama_kv_cache_v: settings?.llama_kv_cache_v ?? "f16",
    llama_direct_cpu_moe: settings?.llama_direct_cpu_moe ?? false,
    llama_cpu_threads: settings?.llama_cpu_threads ?? 0,
    ssd_streaming: settings?.ssd_streaming ?? false,
    ssd_cache_dir: settings?.ssd_cache_dir ?? "",
  };
}

const DEVICE_OPTIONS = [
  ["auto", "Auto"],
  ["cpu", "CPU"],
  ["gpu", "GPU"],
  ["npu", "NPU"],
  ["npu+gpu", "NPU + GPU"],
];

export function EnginePanel({
  collapsed,
  onToggle,
  preferences,
  onUpdate,
  settings,
  activeEngine,
  engineTelemetry,
  engineRunning,
  engineBusy,
  engineStatus,
  adapterWarning,
  onDismissAdapterWarning,
  onOpenEngineSetup,
  onSwitchEngine,
  onApplyEngineSettings,
  onRestartEngine,
  onReset,
  agentPreferences,
  onAgentUpdate,
  onPickWorkspace,
  onClearWorkspace,
  workspaceLocked,
  systemPrompt,
  onSystemPromptChange,
  explicitArtifactTags,
  onExplicitArtifactTagsChange,
  usingDefaultPrompt,
  onUseDefaultPrompt,
  onMakeDefaultPrompt,
  personas,
  onAddPersona,
  onDeletePersona,
  workspaceRoot,
  approvalRulesVersion,
  usage,
  elapsedMs,
  firstTokenMs,
  liveBytes,
  streaming,
  startedAtMs,
}: EnginePanelProps) {
  const [form, setForm] = useState<EngineForm>(() => formFrom(settings));
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [tuning, setTuning] = useState(false);
  const [tuneNotice, setTuneNotice] = useState<string | null>(null);
  const [tuneSwitch, setTuneSwitch] = useState(false);
  const [personaId, setPersonaId] = useState("");
  const [personaOpen, setPersonaOpen] = useState(false);
  const [personaName, setPersonaName] = useState("");
  const selectedPersona = personas.find(
    (persona) => persona.id === personaId && persona.prompt === systemPrompt,
  );

  useEffect(() => {
    setForm(formFrom(settings));
  }, [settings]);

  const dirty = useMemo(() => {
    if (!settings) return false;
    const current = formFrom(settings);
    return (Object.keys(current) as Array<keyof EngineForm>).some(
      (key) => current[key] !== form[key],
    );
  }, [form, settings]);

  const capabilities = useBackendCapabilities(
    `${activeEngine ?? "none"}:${engineStatus}`,
  );
  const hiddenParameters = SAMPLING_PARAMETERS.filter(
    (parameter) => capabilities[parameter.key] === false,
  );
  const adapterNotice = adapterWarning
    ? adapterWarningView(adapterWarning)
    : null;
  const fallbackStep =
    adapterNotice?.kind === "fallback" ? (adapterWarning?.step ?? null) : null;

  if (collapsed) {
    return (
      <aside className="flex w-10 shrink-0 flex-col items-center gap-3 border-l border-white/5 bg-ink-900/40 py-3">
        <button
          type="button"
          onClick={onToggle}
          className="rounded-lg p-1.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
          title={`Expand engine settings (${shortcutLabel("toggle-engine")})`}
          aria-label="Expand engine settings"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <SlidersHorizontal className="h-4 w-4 text-slate-600" />
        <span className="mt-1 text-[8px] uppercase tracking-[0.2em] text-slate-600 [writing-mode:vertical-rl]">
          engine settings
        </span>
      </aside>
    );
  }

  const update = <K extends keyof EngineForm>(key: K, value: EngineForm[K]) =>
    setForm((previous) => ({ ...previous, [key]: value }));

  const apply = async () => {
    if (!settings) return;
    setSaving(true);
    try {
      await onApplyEngineSettings(form);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 2_000);
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    if (!settings || resetting) return;
    const defaults = formFrom(null);
    setResetting(true);
    setForm(defaults);
    try {
      await onReset(defaults);
    } finally {
      setResetting(false);
    }
  };

  /**
   * One click: measure the active model against this machine and write the
   * GPU layers, context and Flash Attention it can actually sustain. A model
   * above system RAM is left untouched and the notice offers Janas instead.
   */
  const autoTune = async () => {
    if (!settings || tuning) return;
    const modelPath =
      activeEngine === "llama" ? settings.llama_model_path : settings.model_path;
    setTuning(true);
    setTuneNotice(null);
    setTuneSwitch(false);
    try {
      const result = await api.autoTuneForModel(modelPath || null);
      setForm(formFrom(result.settings));
      setTuneNotice(result.notice);
      setTuneSwitch(result.requires_engine_switch);
      await onApplyEngineSettings(result.settings);
    } catch (err) {
      setTuneNotice(errorMessage(err));
    } finally {
      setTuning(false);
    }
  };

  return (
    <aside className="flex w-[320px] shrink-0 flex-col border-l border-white/5 bg-ink-900/40">
      <div className="flex h-12 shrink-0 items-center gap-2 border-b border-white/5 px-4">
        <SlidersHorizontal className="h-3.5 w-3.5 text-slate-500" />
        <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300">
          Engine settings
        </span>
        <button
          type="button"
          onClick={() => void reset()}
          disabled={resetting}
          className="ml-auto rounded-lg p-1 text-slate-500 transition hover:bg-rose-500/10 hover:text-rose-300 disabled:opacity-40"
          aria-label="Reset engine settings and clear the system prompt"
          title="Reset sampling, context and runtime settings, and clear the system prompt"
        >
          {resetting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5" />
          )}
        </button>
        <button
          type="button"
          onClick={onToggle}
          className="rounded-lg p-1 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
          title={`Collapse engine settings (${shortcutLabel("toggle-engine")})`}
          aria-label="Collapse engine settings"
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
        {adapterNotice ? (
          <div
            className={cn(
              "rounded-xl border px-3 py-2.5",
              adapterNotice.kind === "auto-rollback"
                ? "border-rose-400/25 bg-rose-500/5"
                : "border-amber-400/20 bg-amber-400/5",
            )}
            role="status"
          >
            <div className="flex items-start gap-2">
              <AlertTriangle
                className={cn(
                  "mt-0.5 h-3.5 w-3.5 shrink-0",
                  adapterNotice.kind === "auto-rollback"
                    ? "text-rose-300"
                    : "text-amber-300",
                )}
              />
              <div className="min-w-0 flex-1">
                <p
                  className={cn(
                    "text-[10.5px] font-medium",
                    adapterNotice.kind === "auto-rollback"
                      ? "text-rose-200"
                      : "text-amber-200",
                  )}
                >
                  {adapterNotice.title}
                </p>
                <p className="mt-1 text-[9.5px] leading-4 text-slate-400">
                  {adapterNotice.detail}
                </p>
                {adapterNotice.hint ? (
                  <p className="mt-1 text-[9px] leading-4 text-slate-600">
                    {adapterNotice.hint}
                  </p>
                ) : null}
                {adapterNotice.showSettings ? (
                  <button
                    type="button"
                    onClick={onOpenEngineSetup}
                    className="mt-1.5 text-[9px] font-medium text-acid-300 underline-offset-2 transition hover:underline"
                  >
                    Open Automatic engine setup
                  </button>
                ) : null}
              </div>
              {adapterNotice.dismissible ? (
                <button
                  type="button"
                  onClick={onDismissAdapterWarning}
                  aria-label="Dismiss llama.cpp warning"
                  className="rounded p-0.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200"
                >
                  <X className="h-3 w-3" />
                </button>
              ) : null}
            </div>
          </div>
        ) : null}

        <TokenTicker
          usage={usage}
          elapsedMs={elapsedMs}
          firstTokenMs={firstTokenMs}
          liveBytes={liveBytes}
          streaming={streaming}
          startedAtMs={startedAtMs}
        />

        <div className="rounded-xl border border-acid-400/20 bg-acid-400/[0.04] px-3 py-2.5">
          <div className="flex items-center justify-between gap-2">
            <p className="flex items-center gap-1.5 text-[10.5px] font-medium text-acid-200">
              <Zap className="h-3 w-3" /> Hardware fit
            </p>
            <button
              type="button"
              onClick={() => void autoTune()}
              disabled={tuning || !settings}
              title="Set GPU layers, context size and Flash Attention from this machine's RAM and VRAM"
              className="inline-flex items-center gap-1 rounded-lg border border-acid-400/30 bg-acid-400/10 px-2 py-1 text-[9.5px] font-medium text-acid-200 transition hover:bg-acid-400/20 disabled:opacity-40"
            >
              {tuning ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Zap className="h-3 w-3" />
              )}
              Auto-Tune
            </button>
          </div>
          <p className="mt-1 text-[9px] leading-4 text-slate-500">
            Reads RAM, VRAM and the active model header, then sets{" "}
            <span className="font-mono">--n-gpu-layers</span>, context and Flash
            Attention. A model larger than RAM is left alone so Janas can stream
            it.
          </p>
          {tuneNotice ? (
            <div className="mt-2 rounded-lg border border-white/5 bg-ink-950/50 px-2 py-1.5">
              <p className="text-[9.5px] leading-4 text-slate-300">
                {tuneNotice}
              </p>
              {tuneSwitch ? (
                <button
                  type="button"
                  onClick={() => void onSwitchEngine("janas")}
                  className="mt-1 text-[9px] font-medium text-acid-300 underline-offset-2 transition hover:underline"
                >
                  Switch to Janas
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        <section className="space-y-2">
          <SectionTitle>Engine</SectionTitle>
          <div className="grid grid-cols-2 gap-2">
            {(["janas", "llama"] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                disabled={engineBusy || activeEngine === kind}
                onClick={() => void onSwitchEngine(kind)}
                className={cn(
                  "rounded-lg border px-2 py-2 text-left transition",
                  activeEngine === kind
                    ? "border-acid-400/40 bg-acid-400/10"
                    : "border-white/10 bg-white/[0.02] hover:bg-white/[0.05] disabled:opacity-50",
                )}
              >
                <span className="flex items-center gap-1.5 text-[10.5px] font-medium text-slate-200">
                  {kind === "janas" ? (
                    <HardDrive className="h-3 w-3 text-slate-500" />
                  ) : (
                    <Zap className="h-3 w-3 text-slate-500" />
                  )}
                  {kind === "janas" ? "Janas" : "llama.cpp"}
                </span>
                <span className="mt-0.5 block text-[9px] text-slate-600">
                  {activeEngine === kind ? "active" : "switch & restart"}
                </span>
              </button>
            ))}
          </div>
          {fallbackStep ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-amber-400/25 bg-amber-400/5 px-2 py-0.5 text-[9px] text-amber-200">
              <Zap className="h-2.5 w-2.5" />
              {fallbackLabel(fallbackStep)}
            </span>
          ) : null}
          <p className="text-[10px] leading-4 text-slate-600">
            Starting one engine stops the other first, so they never contend for
            the same GPU or model lock.
          </p>
        </section>

        <section className="space-y-3">
          <SectionTitle>Sampling</SectionTitle>
          <div className="sampling-stack">
            <div className="param-slot" data-param="temperature">
              <Slider
                label="Temperature"
                value={preferences.temperature}
                min={0}
                max={2}
                step={0.05}
                onChange={(value) => onUpdate("temperature", value)}
                leftHint="0 · greedy"
                rightHint="2 · wild"
              />
            </div>
            <div className="param-slot" data-param="top_p">
              <Slider
                label="Top-P"
                value={preferences.topP}
                min={0}
                max={1}
                step={0.01}
                onChange={(value) => onUpdate("topP", value)}
                leftHint="0 · narrow"
                rightHint="1 · full"
              />
            </div>
            <div className="param-slot" data-param="top_k">
              <Slider
                label="Top-K"
                value={preferences.topK ?? 40}
                min={1}
                max={100}
                step={1}
                formatValue={(value) => String(Math.round(value))}
                onChange={(value) => onUpdate("topK", Math.round(value))}
                leftHint="1 · narrowest"
                rightHint="100 · widest"
                description="How many of the most likely tokens survive sampling. Sent as the integer top_k field."
              />
            </div>
            <div className="param-slot" data-param="min_p">
              <Slider
                label="Min-P"
                value={preferences.minP}
                min={0}
                max={0.5}
                step={0.005}
                onChange={(value) => onUpdate("minP", value)}
                formatValue={(value) => value.toFixed(3)}
                leftHint="0 · keep all"
                rightHint="0.5 · keep likely"
                description="Drops tokens below this fraction of the top token's probability."
              />
            </div>
            <div className="param-slot" data-param="repeat_penalty">
              <Slider
                label="Repeat penalty"
                value={preferences.repeatPenalty}
                min={0.8}
                max={1.8}
                step={0.01}
                onChange={(value) => onUpdate("repeatPenalty", value)}
                leftHint="0.8 · permissive"
                rightHint="1.8 · strict"
              />
            </div>
            <div
              className="param-slot flex items-center justify-between gap-2"
              data-param="max_tokens"
            >
              <span className="text-xs font-medium text-slate-300">
                Max reply
              </span>
              <select
                value={preferences.maxTokens ?? 0}
                onChange={(event) =>
                  onUpdate(
                    "maxTokens",
                    Number(event.target.value) > 0
                      ? Number(event.target.value)
                      : null,
                  )
                }
                className="field !w-28 !py-1.5 text-[11px]"
              >
                <option value={0}>Unlimited</option>
                {[256, 512, 1024, 2048, 4096, 8192].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
          </div>
          {hiddenParameters.length ? (
            <p
              className="rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-2 text-[10px] leading-4 text-slate-600"
              role="status"
              aria-live="polite"
            >
              Not exposed by the active engine:{" "}
              <span className="text-slate-400">
                {hiddenParameters
                  .map((parameter) => parameter.label)
                  .join(", ")}
              </span>
              .
            </p>
          ) : null}
          <Toggle
            checked={preferences.thinking}
            onChange={(value) => onUpdate("thinking", value)}
            label="Thinking mode"
            description="Ask reasoning models to think before replying. Sent as chat_template_kwargs."
          />
          <Toggle
            checked={preferences.structuredOutput}
            onChange={(value) => onUpdate("structuredOutput", value)}
            label="Structured output"
            description="Request response_format: json_object so the reply is valid JSON."
          />
        </section>

        <AgentConfiguration
          preferences={agentPreferences}
          onUpdate={onAgentUpdate}
          onPickWorkspace={onPickWorkspace}
          onClearWorkspace={onClearWorkspace}
          workspaceLocked={workspaceLocked}
          workspaceRoot={workspaceRoot}
          approvalRulesVersion={approvalRulesVersion}
        />

        <section className="space-y-3">
          <SectionTitle>Context</SectionTitle>
          <Slider
            label="Prefetch window"
            value={preferences.contextPrefetch}
            min={1}
            max={40}
            step={1}
            onChange={(value) => onUpdate("contextPrefetch", value)}
            formatValue={(value) => `${value} turns`}
            leftHint="last turn only"
            rightHint="40 turns"
            description="How many earlier turns are bundled and re-read with each prompt."
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-slate-300">
              Cache eviction
            </span>
            <select
              value={preferences.eviction}
              onChange={(event) =>
                onUpdate("eviction", event.target.value as "lru" | "lfu")
              }
              className="field !w-28 !py-1.5 text-[11px]"
            >
              <option value="lru">LRU</option>
              <option value="lfu">LFU</option>
            </select>
          </div>
          <p className="text-[10px] leading-4 text-slate-600">
            Eviction applies to the studio's local memory index (recalled facts).
            janas-server keeps its own expert cache below.
          </p>
        </section>

        {activeEngine !== "llama" ? (
        <section className="space-y-3">
          <SectionTitle>Server runtime</SectionTitle>
          <label className="block space-y-1.5">
            <span className="flex items-center justify-between text-xs text-slate-300">
              Context window
              <span className="font-mono text-[10px] text-slate-500">
                --ctx {form.ctx_size.toLocaleString()}
              </span>
            </span>
            <input
              type="number"
              min={512}
              step={512}
              value={form.ctx_size}
              onChange={(event) =>
                update("ctx_size", Number(event.target.value) || 512)
              }
              className="field !py-1.5 text-[11px]"
            />
          </label>
          <label className="block space-y-1.5">
            <span className="flex items-center justify-between text-xs text-slate-300">
              Expert cache
              <span className="font-mono text-[10px] text-slate-500">
                --cache
              </span>
            </span>
            <input
              type="number"
              min={0}
              step={0.5}
              value={form.cache_gib}
              onChange={(event) =>
                update("cache_gib", Number(event.target.value) || 0)
              }
              className="field !py-1.5 text-[11px]"
            />
            <span className="block text-[10px] text-slate-600">
              0 lets the engine choose based on free memory.
            </span>
          </label>

          <div className="grid grid-cols-2 gap-2">
            <SelectField
              label="MTP / speculation"
              value={form.mtp_mode}
              onChange={(value) => update("mtp_mode", value)}
              options={[
                ["auto", "Auto"],
                ["off", "Off"],
                ["file", "Custom file"],
              ]}
            />
            <SelectField
              label="Engine mode"
              value={form.engine_mode}
              onChange={(value) => update("engine_mode", value)}
              options={[
                ["auto", "Auto"],
                ["eco", "Eco"],
                ["max", "Max"],
              ]}
            />
          </div>
          <SelectField
            label="Flash attention"
            value={form.janas_attention}
            onChange={(value) => update("janas_attention", value)}
            options={[
              ["auto", "Auto · let Janas decide"],
              ["exact", "Exact · float attention"],
              ["fast", "Fast · 16-bit query"],
            ]}
          />
          <SelectField
            label="KV cache quantization"
            value={form.janas_kv_cache}
            onChange={(value) => update("janas_kv_cache", value)}
            options={[
              ["auto", "Auto · model default"],
              ["8", "8-bit · lower memory"],
              ["16", "16-bit · higher quality"],
            ]}
          />
          <p className="text-[10px] leading-4 text-slate-600">
            Janas reads these as JANAS_ATTN and JANAS_KV when the server starts;
            restart the engine to apply them.
          </p>
          {form.mtp_mode === "file" ? (
            <TextField
              label="MTP block file"
              value={form.mtp_path}
              onChange={(value) => update("mtp_path", value)}
              placeholder="/models/model-mtp.jns"
            />
          ) : null}
          <TextField
            label="Draft model"
            value={form.draft_path}
            onChange={(value) => update("draft_path", value)}
            placeholder="optional small same-family .jns"
          />

          <div className="grid grid-cols-2 gap-2">
            <SelectField
              label="Prompt on"
              value={form.prompt_on}
              onChange={(value) => update("prompt_on", value)}
              options={DEVICE_OPTIONS}
            />
            <SelectField
              label="Reply on"
              value={form.reply_on}
              onChange={(value) => update("reply_on", value)}
              options={DEVICE_OPTIONS}
            />
            <SelectField
              label="Drafts on"
              value={form.drafts_on}
              onChange={(value) => update("drafts_on", value)}
              options={DEVICE_OPTIONS}
            />
            <SelectField
              label="Think default"
              value={form.think}
              onChange={(value) => update("think", value)}
              options={[
                ["auto", "Auto"],
                ["on", "On"],
                ["off", "Off"],
              ]}
            />
          </div>
          <Toggle
            checked={form.npu_int8}
            onChange={(value) => update("npu_int8", value)}
            label="NPU int8 activations"
            description="Faster NPU prefill on dense Qwen3 models, slightly less exact."
          />
          <Toggle
            checked={form.ssd_streaming}
            onChange={(value) => update("ssd_streaming", value)}
            label="SSD streaming"
            description="Passed only when the probed Janas build advertises --ssd-streaming; otherwise the adapter keeps it out of the command."
          />
          <TextField
            label="SSD cache directory"
            value={form.ssd_cache_dir}
            onChange={(value) => update("ssd_cache_dir", value)}
            placeholder="optional cache path for streamed experts"
          />
          {engineTelemetry ? (
            <div className="grid grid-cols-3 gap-2 rounded-lg border border-white/5 bg-white/[0.02] p-2 text-center">
              <Metric
                label="SSD read"
                value={`${formatBytes(engineTelemetry.read_bytes_per_sec)}/s`}
              />
              <Metric
                label="IOPS"
                value={engineTelemetry.read_ops_per_sec.toLocaleString()}
              />
              <Metric
                label="KV cache"
                value={`${Math.round(engineTelemetry.kv_cache_usage * 100)}%`}
              />
            </div>
          ) : null}
        </section>
        ) : (
          <LlamaRuntimeSection
            form={form}
            update={update}
            telemetry={engineTelemetry}
            modelPath={settings?.llama_model_path ?? ""}
          />
        )}

        <section className="space-y-2">
          <div className="flex items-center gap-2">
            <SectionTitle>System prompt</SectionTitle>
            <span
              className={cn(
                "ml-auto rounded-full border px-1.5 py-0.5 text-[8px] uppercase tracking-wider",
                usingDefaultPrompt
                  ? "border-white/10 bg-white/[0.03] text-slate-500"
                  : "border-acid-400/25 bg-acid-400/10 text-acid-300/90",
              )}
              title={
                usingDefaultPrompt
                  ? "This chat follows the saved default prompt."
                  : "This chat has its own system prompt."
              }
            >
              {usingDefaultPrompt ? "default" : "this chat"}
            </span>
          </div>
          <textarea
            rows={4}
            value={systemPrompt}
            onChange={(event) => onSystemPromptChange(event.target.value)}
            placeholder="Persistent instructions sent before every conversation…"
            className="field resize-none text-[11px] leading-5"
          />
          <div className="flex items-center gap-1.5">
            <select
              value={selectedPersona?.id ?? ""}
              onChange={(event) => {
                const id = event.target.value;
                setPersonaId(id);
                const persona = personas.find((entry) => entry.id === id);
                if (persona) onSystemPromptChange(persona.prompt);
              }}
              className="field min-w-0 flex-1 !px-2 !py-1.5 text-[10px]"
            >
              <option value="">Insert a persona…</option>
              {personas.map((persona) => (
                <option key={persona.id} value={persona.id}>
                  {persona.name}
                </option>
              ))}
            </select>
            {selectedPersona ? (
              <button
                type="button"
                onClick={() => {
                  onDeletePersona(selectedPersona.id);
                  setPersonaId("");
                }}
                className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-500 transition hover:text-rose-300"
                aria-label="Delete the selected persona"
                title="Delete this persona"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            ) : null}
            <button
              type="button"
              onClick={() => setPersonaOpen((value) => !value)}
              disabled={!systemPrompt.trim()}
              className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 transition hover:text-slate-200 disabled:opacity-40"
              aria-label="Save the current prompt as a persona"
              title="Save this prompt as a persona"
            >
              <Plus className="h-3 w-3" />
            </button>
          </div>
          {personaOpen ? (
            <div className="flex items-center gap-1.5">
              <input
                value={personaName}
                onChange={(event) => setPersonaName(event.target.value)}
                placeholder="Persona name"
                className="field min-w-0 flex-1 !px-2 !py-1.5 text-[10px]"
              />
              <button
                type="button"
                onClick={() => {
                  const persona = onAddPersona(personaName, systemPrompt);
                  if (persona) setPersonaId(persona.id);
                  setPersonaName("");
                  setPersonaOpen(false);
                }}
                disabled={!systemPrompt.trim()}
                className="inline-flex h-7 shrink-0 items-center rounded-lg bg-acid-400 px-2 text-[10px] font-medium text-ink-950 transition hover:bg-acid-300 disabled:opacity-40"
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setPersonaOpen(false)}
                className="inline-flex h-7 shrink-0 items-center rounded-lg border border-white/10 bg-white/5 px-2 text-[10px] text-slate-400 transition hover:text-slate-200"
              >
                Cancel
              </button>
            </div>
          ) : null}
          <div className="flex items-center gap-2">
            {!usingDefaultPrompt ? (
              <>
                <button
                  type="button"
                  onClick={onUseDefaultPrompt}
                  className="text-[9.5px] text-slate-500 underline-offset-2 transition hover:text-slate-300 hover:underline"
                >
                  Use default
                </button>
                <button
                  type="button"
                  onClick={onMakeDefaultPrompt}
                  disabled={!systemPrompt.trim()}
                  className="text-[9.5px] text-slate-500 underline-offset-2 transition hover:text-slate-300 hover:underline disabled:opacity-40"
                >
                  Make default
                </button>
              </>
            ) : null}
          </div>
          <p className="text-[10px] leading-4 text-slate-600">
            {usingDefaultPrompt
              ? "This chat follows the saved default. Editing here gives it its own prompt; new chats keep the default."
              : "This chat has its own prompt. Other chats and new chats keep the saved default."}
          </p>
          <div className="rounded-lg border border-white/5 bg-white/[0.02] p-2.5">
            <Toggle
              checked={explicitArtifactTags}
              onChange={onExplicitArtifactTagsChange}
              label="Explicit artifact tags"
              description="Append an instruction asking the model to wrap complete HTML, SVG, Mermaid, Markdown, JSON and CSV documents in <artifact> tags. It does not edit your prompt; fenced blocks still preview automatically."
            />
          </div>
        </section>

        <div className="space-y-2 rounded-xl border border-white/5 bg-white/[0.02] p-2.5">
          <p className="flex items-center gap-1.5 text-[10px] leading-4 text-slate-500">
            <Sparkles className="h-3 w-3 text-acid-500/70" />
            Runtime flags require a restart; sampling applies to the next
            message.
          </p>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => void apply()}
              disabled={!dirty || saving}
              className={cn(
                "inline-flex h-7 flex-1 items-center justify-center gap-1.5 rounded-lg px-2 text-[10px] font-medium transition",
                dirty
                  ? "bg-acid-400 text-ink-950 hover:bg-acid-300"
                  : "border border-white/10 bg-white/5 text-slate-500",
              )}
            >
              {saving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : saved ? (
                <Save className="h-3 w-3" />
              ) : (
                <MemoryStick className="h-3 w-3" />
              )}
              {saved ? "Saved" : "Apply runtime"}
            </button>
            {engineRunning ? (
              <button
                type="button"
                onClick={() => void onRestartEngine()}
                disabled={engineBusy}
                className="inline-flex h-7 items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2 text-[10px] text-slate-300 transition hover:bg-white/10 disabled:opacity-40"
                title="Restart the engine with the saved runtime flags"
              >
                {engineBusy ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Play className="h-3 w-3" />
                )}
                Restart
              </button>
            ) : null}
          </div>
          {settings ? (
            <p className="flex items-center gap-1.5 truncate font-mono text-[9px] text-slate-600">
              <Cpu className="h-2.5 w-2.5" />
              {activeEngine === "llama"
                ? `${form.llama_ctx_size.toLocaleString()} ctx · ${form.llama_gpu_layers} GPU layers`
                : `${formatBytes(settings.cache_gib > 0 ? settings.cache_gib * 1024 ** 3 : 0)} cache`}
            </p>
          ) : null}
        </div>
      </div>
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-mono text-[10px] text-slate-300">{value}</p>
      <p className="mt-0.5 text-[8px] uppercase tracking-wider text-slate-600">
        {label}
      </p>
    </div>
  );
}

const KV_CACHE_OPTIONS = [
  ["f16", "f16 · balanced"],
  ["q8_0", "q8_0 · 8-bit"],
  ["q5_1", "q5_1 · 5-bit"],
  ["q5_0", "q5_0 · 5-bit"],
  ["q4_1", "q4_1 · 4-bit"],
  ["q4_0", "q4_0 · 4-bit"],
  ["iq4_nl", "iq4_nl · 4-bit"],
  ["f32", "f32 · full"],
];

function LlamaRuntimeSection({
  form,
  update,
  telemetry,
  modelPath,
}: {
  form: EngineForm;
  update: <K extends keyof EngineForm>(key: K, value: EngineForm[K]) => void;
  telemetry: EngineTelemetry | null;
  modelPath: string;
}) {
  return (
    <section className="space-y-3">
      <SectionTitle>llama.cpp runtime</SectionTitle>
      <div className="space-y-3">
        <div className="space-y-1.5">
          <span className="flex items-center justify-between text-[10px] text-slate-500">
            GPU offload
            <span className="font-mono text-[9px]">--n-gpu-layers</span>
          </span>
          <GpuOffloadControl
            modelPath={modelPath}
            value={form.llama_gpu_layers}
            onChange={(value) => update("llama_gpu_layers", value)}
          />
        </div>
        <label className="block space-y-1.5">
          <span className="flex items-center justify-between text-[10px] text-slate-500">
            Context size
            <span className="font-mono text-[9px]">-c</span>
          </span>
          <input
            type="number"
            min={512}
            step={512}
            value={form.llama_ctx_size}
            onChange={(event) =>
              update("llama_ctx_size", Number(event.target.value) || 512)
            }
            className="field !py-1.5 font-mono text-[11px]"
          />
        </label>
      </div>
      <SelectField
        label="Flash attention"
        value={form.llama_flash_attn}
        onChange={(value) => update("llama_flash_attn", value)}
        options={[
          ["auto", "Auto · let llama.cpp decide"],
          ["on", "On · faster attention"],
          ["off", "Off · maximum compatibility"],
        ]}
      />
      <div className="grid grid-cols-2 gap-2">
        <SelectField
          label="KV cache K"
          value={form.llama_kv_cache_k}
          onChange={(value) => update("llama_kv_cache_k", value)}
          options={KV_CACHE_OPTIONS}
        />
        <SelectField
          label="KV cache V"
          value={form.llama_kv_cache_v}
          onChange={(value) => update("llama_kv_cache_v", value)}
          options={KV_CACHE_OPTIONS}
        />
      </div>
      <p className="text-[10px] leading-4 text-slate-600">
        KV quantization cuts cache memory, and some models only accept f16 for
        V.
      </p>
      <div className="space-y-2.5 rounded-xl border border-white/5 bg-white/[0.02] p-2.5">
        <SelectField
          label="MTP / speculation"
          value={form.llama_mtp_mode}
          onChange={(value) => update("llama_mtp_mode", value)}
          options={[
            ["auto", "Auto · llama.cpp default"],
            ["off", "Off · --spec-type none"],
            ["mtp", "MTP · --spec-type draft-mtp"],
          ]}
        />
        {form.llama_mtp_mode === "mtp" ? (
          <TextField
            label="MTP head file"
            value={form.llama_mtp_path}
            onChange={(value) => update("llama_mtp_path", value)}
            placeholder="optional mtp-*.gguf"
          />
        ) : null}
        <TextField
          label="Draft model"
          value={form.llama_draft_path}
          onChange={(value) => update("llama_draft_path", value)}
          placeholder="optional small same-family .gguf"
        />
        <div className="grid grid-cols-3 gap-2">
          <label className="space-y-1">
            <span className="text-[10px] text-slate-500">Draft max</span>
            <input
              type="number"
              min={1}
              max={64}
              step={1}
              value={form.llama_draft_max}
              onChange={(event) =>
                update("llama_draft_max", Number(event.target.value) || 8)
              }
              className="field !py-1.5 font-mono text-[11px]"
              title="--draft-max: tokens proposed per speculative step"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[10px] text-slate-500">Draft min</span>
            <input
              type="number"
              min={0}
              max={64}
              step={1}
              value={form.llama_draft_min}
              onChange={(event) =>
                update("llama_draft_min", Number(event.target.value) || 0)
              }
              className="field !py-1.5 font-mono text-[11px]"
              title="--draft-min: shortest accepted draft run (0 uses llama.cpp's default)"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[10px] text-slate-500">Draft p-min</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={form.llama_draft_p_min}
              onChange={(event) =>
                update("llama_draft_p_min", Number(event.target.value) || 0)
              }
              className="field !py-1.5 font-mono text-[11px]"
              title="--draft-p-min: minimum draft probability kept"
            />
          </label>
        </div>
        <SelectField
          label="Drafts on"
          value={form.llama_drafts_on}
          onChange={(value) => update("llama_drafts_on", value)}
          options={[
            ["auto", "Auto · llama.cpp decides"],
            ["cpu", "CPU · draft device none"],
            ["gpu", "GPU · all draft layers"],
          ]}
        />
        <p className="text-[10px] leading-4 text-slate-600">
          Auto passes the draft model and lets llama.cpp detect the type. MTP
          uses the heads inside the main model; the head file is only needed
          when they ship in a separate GGUF, and placement only applies once a
          draft model or head file is loaded. The two paths share one slot, so
          set just one.
        </p>
      </div>
      <div className="space-y-2.5 rounded-xl border border-white/5 bg-white/[0.02] p-2.5">
        <Toggle
          checked={form.llama_direct_cpu_moe}
          onChange={(value) => update("llama_direct_cpu_moe", value)}
          label="Direct CPU MoE Compute"
          description="Runs MoE cache misses from host RAM on the CPU while the GPU computes cache hits. llama.cpp only."
        />
        <label
          className={cn(
            "flex items-center justify-between gap-2",
            !form.llama_direct_cpu_moe && "opacity-50",
          )}
        >
          <span className="text-[10px] text-slate-500">
            Max CPU compute threads
          </span>
          <input
            type="number"
            min={0}
            max={256}
            step={1}
            disabled={!form.llama_direct_cpu_moe}
            value={form.llama_cpu_threads}
            onChange={(event) =>
              update("llama_cpu_threads", Number(event.target.value) || 0)
            }
            className="field !w-20 !py-1.5 font-mono text-[11px] disabled:cursor-not-allowed"
          />
        </label>
        <p className="text-[10px] leading-4 text-slate-600">
          Sets a 2 GiB MoE cache and keeps expert weights in system RAM. Use 0
          for llama.cpp's thread default; override the cache with{" "}
          <span className="font-mono text-slate-500">--moe-cache-mib</span> in
          Extra llama arguments.
        </p>
      </div>
      {telemetry ? (
        <div className="grid grid-cols-3 gap-2 rounded-lg border border-white/5 bg-white/[0.02] p-2 text-center">
          <Metric
            label="tokens/s"
            value={telemetry.tokens_per_second.toFixed(1)}
          />
          <Metric
            label="KV cache"
            value={`${Math.round(telemetry.kv_cache_usage * 100)}%`}
          />
          <Metric label="RSS" value={formatBytes(telemetry.rss_bytes)} />
        </div>
      ) : null}
    </section>
  );
}

function AgentConfiguration({
  preferences,
  onUpdate,
  onPickWorkspace,
  onClearWorkspace,
  workspaceLocked,
  workspaceRoot,
  approvalRulesVersion,
}: {
  preferences: AgentPreferences;
  onUpdate: <K extends keyof AgentPreferences>(
    key: K,
    value: AgentPreferences[K],
  ) => void;
  onPickWorkspace: () => Promise<void>;
  onClearWorkspace: () => void;
  workspaceLocked: boolean;
  workspaceRoot: string;
  approvalRulesVersion: number;
}) {
  const [open, setOpen] = useState(true);
  const [remembered, setRemembered] = useState<string[]>([]);
  const [rulesError, setRulesError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!workspaceRoot.trim()) {
      setRemembered([]);
      setRulesError(null);
      return;
    }
    void api
      .agentApprovalRules(workspaceRoot)
      .then((rules) => {
        if (!cancelled) {
          setRemembered(rules);
          setRulesError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setRemembered([]);
          setRulesError(errorMessage(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [workspaceRoot, approvalRulesVersion]);

  const forgetRule = async (command: string) => {
    try {
      setRemembered(await api.forgetApprovalRule(workspaceRoot, command));
      setRulesError(null);
    } catch (err) {
      setRulesError(errorMessage(err));
    }
  };

  const clearRules = async () => {
    try {
      setRemembered(await api.clearApprovalRules(workspaceRoot));
      setRulesError(null);
    } catch (err) {
      setRulesError(errorMessage(err));
    }
  };

  return (
    <section className="space-y-3">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 text-left"
        aria-expanded={open}
      >
        <SectionTitle>Agent configuration</SectionTitle>
        <Bot className="h-3 w-3 shrink-0 text-acid-500/60" />
        <ChevronDown
          className={cn(
            "h-3 w-3 shrink-0 text-slate-600 transition",
            !open && "-rotate-90",
          )}
        />
      </button>

      {open ? (
        <div className="space-y-3">
          <Slider
            label="Max iterations"
            value={preferences.maxIterations}
            min={1}
            max={25}
            step={1}
            onChange={(value) => onUpdate("maxIterations", Math.round(value))}
            formatValue={(value) => `${Math.round(value)} steps`}
            leftHint="1 · single step"
            rightHint="25 · long tasks"
            description="Hard stop for the Think → Act → Observe loop."
          />

          <label className="block space-y-1">
            <span className="block text-[10px] text-slate-500">Approval policy</span>
            <select
              value={preferences.approvalPolicy}
              onChange={(event) =>
                onUpdate(
                  "approvalPolicy",
                  event.target.value as AgentPreferences["approvalPolicy"],
                )
              }
              className="field !px-2 !py-1.5 text-[10.5px]"
            >
              <option value="destructive">Ask for destructive actions</option>
              <option value="all">Ask for everything except reads</option>
              <option value="auto">Full auto</option>
            </select>
            <span className="block text-[10px] leading-4 text-slate-600">
              {preferences.approvalPolicy === "auto"
                ? "Every normal tool runs without pausing. Irreversible commands and sensitive paths still ask. Use only in a workspace you trust."
                : preferences.approvalPolicy === "all"
                  ? "Writes, shell and MCP tools pause for approval, except shell commands you remembered for this workspace. Read-only tools run automatically."
                  : "Read-only tools run automatically. Writes, shell and MCP tools pause unless the shell command is remembered for this workspace."}
            </span>
            {preferences.approvalPolicy !== "auto" ? (
              <span className="block text-[10px] leading-4 text-slate-600">
                “Allow file edits for this run” covers only write_file and
                edit_file. Shell commands and MCP tools still pause for
                approval.
              </span>
            ) : null}
            <span className="block text-[10px] leading-4 text-slate-600">
              Sensitive paths (.env, .git, SSH keys, cloud credentials) always
              ask when named in a tool path or command. Irreversible commands
              (force-push, reset --hard, rm -rf, publishing, curl | sh) always
              ask, even under Full auto. Tool output is scrubbed for likely
              secrets before it reaches the trace or model.
            </span>
          </label>

          <div className="space-y-1.5 rounded-lg border border-white/5 bg-white/[0.02] p-2">
            <div className="flex items-center gap-1.5">
              <ShieldCheck className="h-3 w-3 shrink-0 text-slate-600" />
              <span className="text-[10px] text-slate-500">
                Remembered commands
              </span>
              {remembered.length ? (
                <button
                  type="button"
                  onClick={() => void clearRules()}
                  className="ml-auto text-[9px] text-slate-500 transition hover:text-rose-300"
                >
                  Clear all
                </button>
              ) : null}
            </div>
            {workspaceRoot.trim() ? (
              <>
                <p
                  className="truncate font-mono text-[8.5px] text-slate-600"
                  title={workspaceRoot}
                >
                  {workspaceRoot}
                </p>
                {remembered.length ? (
                  <div className="space-y-1">
                    {remembered.map((command) => (
                      <div
                        key={command}
                        className="flex items-center gap-1.5 rounded-md border border-white/5 bg-white/[0.02] px-2 py-1"
                      >
                        <code
                          className="min-w-0 flex-1 truncate font-mono text-[9.5px] text-slate-300"
                          title={command}
                        >
                          {command}
                        </code>
                        <button
                          type="button"
                          onClick={() => void forgetRule(command)}
                          className="shrink-0 text-slate-600 transition hover:text-rose-300"
                          aria-label={`Forget ${command}`}
                          title="Forget this command"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-[9px] leading-4 text-slate-600">
                    Nothing remembered yet. Approving a shell command with
                    “Always allow this command here” adds it to this workspace.
                    Rules match the same command with extra arguments; compound
                    commands and commands containing likely secrets never match.
                  </p>
                )}
              </>
            ) : (
              <p className="text-[9px] leading-4 text-slate-600">
                Choose a workspace folder to see the commands remembered there.
              </p>
            )}
            {rulesError ? (
              <p className="text-[9px] leading-4 text-rose-300/80">
                {rulesError}
              </p>
            ) : null}
          </div>

          <label className="block space-y-1">
            <span className="block text-[10px] text-slate-500">
              Verification command
            </span>
            <input
              value={preferences.verifyCommand ?? ""}
              onChange={(event) =>
                onUpdate("verifyCommand", event.target.value)
              }
              placeholder="npm test"
              spellCheck={false}
              className="field !px-2 !py-1.5 font-mono text-[10.5px]"
            />
            <span className="block text-[10px] leading-4 text-slate-600">
              Runs in the workspace after file edits, before the agent can
              finish. Leave empty to disable. It is pre-authorised, so it runs
              without an approval prompt.
            </span>
          </label>

          <div className="space-y-1.5">
            <span className="block text-[10px] text-slate-500">Workspace root</span>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => void onPickWorkspace()}
                disabled={workspaceLocked}
                className={cn(
                  "inline-flex h-8 min-w-0 flex-1 items-center gap-1.5 rounded-lg border px-2 text-[10px] transition disabled:cursor-not-allowed disabled:opacity-50",
                  workspaceRoot
                    ? "border-white/10 bg-white/[0.03] text-slate-300 hover:bg-white/[0.06]"
                    : "border-dashed border-white/15 bg-transparent text-slate-500 hover:text-slate-300",
                )}
                title={
                  workspaceLocked
                    ? "The agent workspace is pinned while a run or pending undo exists."
                    : workspaceRoot || "Choose a workspace folder"
                }
              >
                <FolderOpen className="h-3 w-3 shrink-0" />
                <span className="truncate">
                  {workspaceRoot
                    ? basename(workspaceRoot)
                    : "Choose a workspace folder…"}
                </span>
              </button>
              {workspaceRoot ? (
                <button
                  type="button"
                  onClick={() => onClearWorkspace()}
                  disabled={workspaceLocked}
                  className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-500 transition hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-40"
                  aria-label="Clear the agent workspace root"
                  title={
                    workspaceLocked
                      ? "The agent workspace is pinned while a run or pending undo exists."
                      : "Clear the workspace root"
                  }
                >
                  <X className="h-3 w-3" />
                </button>
              ) : null}
            </div>
            <p className="break-all font-mono text-[9px] leading-4 text-slate-600">
              {workspaceRoot ||
                "File, search and shell tools stay disabled until a folder is chosen. MCP tools still work."}
            </p>
            {workspaceLocked ? (
              <p className="text-[9px] leading-4 text-amber-300/70">
                Locked while a run or pending undo exists. Undo or keep its
                changes before switching workspaces.
              </p>
            ) : null}
          </div>

          <Toggle
            checked={preferences.planningStep}
            onChange={(value) => onUpdate("planningStep", value)}
            label="Planning step"
            description="Ask for a numbered <plan> checklist before the first tool call."
          />

          <Toggle
            checked={preferences.autoCompact}
            onChange={(value) => onUpdate("autoCompact", value)}
            label="Auto-compact context"
            description="Summarise older steps into <agent_memory> when the window fills up."
          />

          <Slider
            label="Compaction threshold"
            value={Math.round(preferences.compactionThreshold * 100)}
            min={50}
            max={90}
            step={1}
            disabled={!preferences.autoCompact}
            onChange={(value) => onUpdate("compactionThreshold", value / 100)}
            formatValue={(value) => `${Math.round(value)}%`}
            leftHint="50% · early"
            rightHint="90% · late"
            description="Share of the active model's n_ctx that triggers compaction."
          />
        </div>
      ) : null}
    </section>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-600">
        {children}
      </span>
      <span className="h-px flex-1 bg-white/5" />
    </div>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[][];
}) {
  return (
    <label className="block space-y-1">
      <span className="block truncate text-[10px] text-slate-500">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="field !px-2 !py-1.5 text-[10.5px]"
      >
        {options.map(([id, text]) => (
          <option key={id} value={id}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block space-y-1">
      <span className="block text-[10px] text-slate-500">{label}</span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        className="field !px-2 !py-1.5 font-mono text-[10px]"
      />
    </label>
  );
}
