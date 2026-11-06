import { LucideIcon, Sprout } from "lucide-react";
import Link from "next/link";

interface EmptyStateProps {
  title: string;
  description: string;
  icon?: LucideIcon;
  actionText?: string;
  actionHref?: string;
  onActionClick?: () => void;
}

export default function EmptyState({
  title,
  description,
  icon: Icon = Sprout,
  actionText,
  actionHref,
  onActionClick,
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center p-8 sm:p-12 text-center rounded-2xl bg-[#152C22]/50 border border-[#294B3C] border-dashed">
      <div className="w-14 h-14 rounded-2xl bg-[#10231B] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0] mb-4 shadow-inner">
        <Icon className="w-7 h-7 text-[#6AE6A0]" />
      </div>
      <h3 className="text-base font-bold text-[#F2FAF5] mb-1.5">{title}</h3>
      <p className="text-xs sm:text-sm text-[#A6B9AE] max-w-md leading-relaxed mb-6">
        {description}
      </p>

      {actionText && actionHref && (
        <Link
          href={actionHref}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-semibold text-xs shadow-md shadow-[#6AE6A0]/15 transition-all hover:scale-102"
        >
          {actionText}
        </Link>
      )}

      {actionText && !actionHref && onActionClick && (
        <button
          onClick={onActionClick}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-semibold text-xs shadow-md shadow-[#6AE6A0]/15 transition-all hover:scale-102"
        >
          {actionText}
        </button>
      )}
    </div>
  );
}

