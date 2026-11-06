interface StatusIndicatorProps {
  status: string;
  size?: "sm" | "md" | "lg";
  showDot?: boolean;
}

export default function StatusIndicator({
  status,
  size = "md",
  showDot = true,
}: StatusIndicatorProps) {
  const normalized = (status || "").toLowerCase().trim();

  let bgClass = "bg-[#10231B] text-[#A6B9AE] border-[#294B3C]";
  let dotClass = "bg-[#A6B9AE]";
  let label = status;

  if (
    normalized.includes("unsupported") ||
    normalized.includes("not_detected") ||
    normalized.includes("untrusted") ||
    normalized.includes("unavailable")
  ) {
    bgClass = "bg-[#271515] text-[#F17878] border-[#F17878]/30";
    dotClass = "bg-[#F17878]";
    label = status ? status.replace(/_/g, " ") : "Unavailable";
  } else if (
    normalized.includes("healthy") ||
    normalized.includes("none") ||
    normalized.includes("zero") ||
    normalized.includes("ready") ||
    normalized.includes("pass") ||
    normalized.includes("clean") ||
    normalized.includes("online")
  ) {
    bgClass = "bg-[#152C22] text-[#6AE6A0] border-[#6AE6A0]/30";
    dotClass = "bg-[#6AE6A0]";
    label = status || "Healthy";
  } else if (
    normalized.includes("low") ||
    normalized.includes("warning") ||
    normalized.includes("moderate") ||
    normalized.includes("review") ||
    normalized.includes("uncertain")
  ) {
    bgClass = "bg-[#252014] text-[#F1BA65] border-[#F1BA65]/30";
    dotClass = "bg-[#F1BA65]";
    label = status || "Moderate";
  } else if (
    normalized.includes("severe") ||
    normalized.includes("danger") ||
    normalized.includes("fail") ||
    normalized.includes("alert") ||
    normalized.includes("infestation") ||
    normalized.includes("high")
  ) {
    bgClass = "bg-[#271515] text-[#F17878] border-[#F17878]/30";
    dotClass = "bg-[#F17878]";
    label = status || "Severe";
  } else if (
    normalized.includes("processing") ||
    normalized.includes("running") ||
    normalized.includes("evaluating")
  ) {
    bgClass = "bg-[#1D4032] text-[#8EF0AF] border-[#8EF0AF]/30 animate-pulse";
    dotClass = "bg-[#8EF0AF]";
    label = status || "Processing";
  }

  const sizeClasses = {
    sm: "text-[10px] px-2 py-0.5 gap-1",
    md: "text-xs px-2.5 py-1 gap-1.5",
    lg: "text-sm px-3.5 py-1.5 gap-2 font-semibold",
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border font-medium uppercase tracking-wider ${bgClass} ${sizeClasses[size]}`}
    >
      {showDot && (
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotClass}`} />
      )}
      <span className="truncate">{label}</span>
    </span>
  );
}

