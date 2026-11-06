import { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon: LucideIcon;
  trend?: {
    value: string;
    isPositive?: boolean;
    isNeutral?: boolean;
  };
  highlight?: boolean;
}

export default function StatCard({
  label,
  value,
  subtext,
  icon: Icon,
  trend,
  highlight = false,
}: StatCardProps) {
  return (
    <div
      className={`p-5 rounded-2xl border transition-all duration-200 group ${
        highlight
          ? "bg-[#1D4032] border-[#6AE6A0]/40 shadow-lg shadow-[#6AE6A0]/10"
          : "bg-[#152C22] border-[#294B3C] hover:border-[#6AE6A0]/40 hover:bg-[#1D4032]"
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-[#A6B9AE] tracking-wide uppercase">
          {label}
        </span>
        <div className="w-9 h-9 rounded-xl bg-[#10231B] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0] group-hover:scale-105 group-hover:border-[#6AE6A0]/50 transition-all">
          <Icon className="w-4 h-4 text-[#6AE6A0]" />
        </div>
      </div>

      <div className="flex items-baseline gap-2 mb-1">
        <span className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight">
          {value}
        </span>
        {trend && (
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${
              trend.isNeutral
                ? "bg-[#10231B] text-[#A6B9AE] border-[#294B3C]"
                : trend.isPositive
                ? "bg-[#10231B] text-[#6AE6A0] border-[#6AE6A0]/30"
                : "bg-[#10231B] text-[#F17878] border-[#F17878]/30"
            }`}
          >
            {trend.value}
          </span>
        )}
      </div>

      {subtext && (
        <p className="text-xs text-[#768D80] leading-relaxed truncate">
          {subtext}
        </p>
      )}
    </div>
  );
}

