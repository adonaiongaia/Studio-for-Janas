import {
  Cpu,
  HardDrive,
  TriangleAlert,
  Zap,
  type LucideIcon,
} from "lucide-react";

import { cn } from "../../lib/cn";
import type { FitTier, ModelFitEstimate } from "../../types";

const TONES: Record<
  FitTier,
  { className: string; icon: LucideIcon; hint: string }
> = {
  fits_vram: {
    className: "border-emerald-400/25 bg-emerald-400/10 text-emerald-300",
    icon: Zap,
    hint: "Runs fully on the GPU.",
  },
  ram_gpu_split: {
    className: "border-sky-400/25 bg-sky-400/10 text-sky-300",
    icon: Cpu,
    hint: "Offloads the layers that fit and keeps the rest in RAM.",
  },
  requires_ssd_streaming: {
    className: "border-amber-400/25 bg-amber-400/10 text-amber-300",
    icon: HardDrive,
    hint: "Exceeds RAM: stream the .jns from the SSD with janas-server.",
  },
  tight_or_slow: {
    className: "border-rose-400/25 bg-rose-500/10 text-rose-300",
    icon: TriangleAlert,
    hint: "Exceeds RAM and the drive is too slow to stream comfortably.",
  },
};

/** The four-tier fit verdict, shared by the Hub dialog and the local library. */
export function FitBadge({
  fit,
}: {
  fit: ModelFitEstimate | null | undefined;
}) {
  if (!fit) return null;
  const tone = TONES[fit.tier];
  const Icon = tone.icon;
  return (
    <span
      title={`${fit.label} — ${fit.reason} ${tone.hint}`}
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[8px] font-medium",
        tone.className,
      )}
    >
      <Icon className="h-2.5 w-2.5" />
      {fit.label}
    </span>
  );
}
