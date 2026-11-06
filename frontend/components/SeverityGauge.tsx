"use client";

import { useLanguage } from "@/context/LanguageContext";
import { AlertTriangle, CheckCircle, Info, ShieldAlert } from "lucide-react";

interface SeverityGaugeProps {
  damagePercentage: number | null;
  leafPixels: number | null;
  minedPixels: number | null;
  severityTier: string;
  uncertaintyStatus?: string;
  isModelPredicted?: boolean;
}

export default function SeverityGauge({
  damagePercentage,
  leafPixels,
  minedPixels,
  severityTier,
  uncertaintyStatus,
  isModelPredicted = true,
}: SeverityGaugeProps) {
  const { t } = useLanguage();
  const pct =
    damagePercentage !== null && damagePercentage !== undefined
      ? damagePercentage
      : null;

  const tierBadge = {
    label: pct == null ? t("severity.unavailable") : pct === 0 ? t("severity.no_regions") : t("severity.estimated"),
    bg: pct == null ? "bg-[#252014]" : "bg-[#10231B]",
    text: pct == null ? "text-[#F1BA65]" : "text-[#6AE6A0]",
    border: "border-[#294B3C]",
    barColor: "bg-[#6AE6A0]",
  };

  return (
    <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE]">
            {t("severity.title")}
          </span>
          <h3 className="text-base font-bold text-[#F2FAF5] mt-0.5">
            {t("severity.subtitle")}
          </h3>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${tierBadge.bg} ${tierBadge.text} ${tierBadge.border}`}
        >
          {tierBadge.label}
        </span>
      </div>

      {/* Percentage Display */}
      <div className="my-4">
        <div className="flex items-baseline gap-2">
          {pct !== null ? (
            <>
              <span className="text-4xl sm:text-5xl font-black text-[#F2FAF5] tracking-tight">
                {pct.toFixed(2)}%
              </span>
              <span className="text-xs sm:text-sm font-semibold text-[#A6B9AE]">
                of estimated leaf surface in accepted regions
              </span>
            </>
          ) : (
            <span className="text-2xl font-bold text-[#768D80]">
              {t("severity.unavailable")}
            </span>
          )}
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-[#10231B] h-3.5 rounded-full mt-3 overflow-hidden p-0.5 border border-[#294B3C]">
          <div
            className={`h-full rounded-full transition-all duration-700 ${tierBadge.barColor}`}
            style={{ width: `${Math.min(100, Math.max(0, pct ?? 0))}%` }}
          />
        </div>
      </div>

      {/* Pixel Metrics Breakdown */}
      <div className="grid grid-cols-2 gap-3 pt-3 border-t border-[#294B3C] text-xs">
        <div className="bg-[#10231B] p-3 rounded-xl border border-[#294B3C]">
          <span className="text-[#768D80] block text-[11px]">{t("severity.canopy_px")}</span>
          <span className="font-mono font-bold text-[#F2FAF5] text-sm mt-0.5 block">
            {leafPixels ? leafPixels.toLocaleString() : "N/A"} px
          </span>
        </div>
        <div className="bg-[#10231B] p-3 rounded-xl border border-[#294B3C]">
          <span className="text-[#768D80] block text-[11px]">{t("severity.mined_px")}</span>
          <span className="font-mono font-bold text-[#F17878] text-sm mt-0.5 block">
            {minedPixels?.toLocaleString() ?? "N/A"} px
          </span>
        </div>
      </div>

      {/* Scientific & Agronomic Context */}
      <div className="mt-4 pt-3 border-t border-[#294B3C] text-[11px] text-[#A6B9AE] space-y-2">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-[#6AE6A0] shrink-0 mt-0.5" />
          <span>
            {isModelPredicted ? "Estimated Area:" : "Measured Annotation Area:"} Calculated via{" "}
            <code className="text-[#6AE6A0] font-mono text-[11px] bg-[#10231B] px-1.5 py-0.5 rounded border border-[#294B3C]">
              100 × (Mines ∩ Leaf) / Leaf
            </code>
            . Overlapping mine polygons are automatically merged to prevent double-counting.
          </span>
        </div>
        <div className="flex items-start gap-2 text-[#F1BA65] bg-[#252014] p-2.5 rounded-xl border border-[#F1BA65]/30">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>
            Visible surface injury does not model whole-plant systemic physiological stress or yield reduction. Always correlate with on-site agronomist field scouting.
          </span>
        </div>
      </div>
    </div>
  );
}
