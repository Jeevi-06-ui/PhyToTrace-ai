import { Cpu, CheckCircle2, XCircle, AlertCircle, Sparkles } from "lucide-react";

interface ModelInfo {
  name: string;
  role: string;
  architecture: string;
  checkpoint: string;
  status: "ready" | "unavailable" | "warning";
  metrics: { [key: string]: string | number };
  device?: string;
}

interface ModelStatusCardProps {
  model: ModelInfo;
}

export default function ModelStatusCard({ model }: ModelStatusCardProps) {
  const isReady = model.status === "ready";

  return (
    <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 hover:border-[#6AE6A0]/40 transition-all duration-200">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-xl border flex items-center justify-center shrink-0 ${
              isReady
                ? "bg-[#10231B] border-[#6AE6A0]/40 text-[#6AE6A0]"
                : "bg-[#271515] border-[#F17878]/40 text-[#F17878]"
            }`}
          >
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-bold text-sm text-[#F2FAF5] flex items-center gap-2">
              {model.name}
            </h4>
            <span className="text-xs text-[#A6B9AE]">{model.role}</span>
          </div>
        </div>

        <span
          className={`px-2.5 py-1 rounded-full text-[10px] font-bold border uppercase tracking-wider flex items-center gap-1.5 ${
            isReady
              ? "bg-[#10231B] text-[#6AE6A0] border-[#6AE6A0]/30"
              : "bg-[#271515] text-[#F17878] border-[#F17878]/30"
          }`}
        >
          {isReady ? (
            <>
              <CheckCircle2 className="w-3 h-3 text-[#6AE6A0]" />
              Loaded & Active
            </>
          ) : (
            <>
              <XCircle className="w-3 h-3 text-[#F17878]" />
              Unavailable
            </>
          )}
        </span>
      </div>

      <div className="space-y-2 pt-2 border-t border-[#294B3C] text-xs">
        <div className="flex justify-between items-center text-[#A6B9AE]">
          <span>Architecture:</span>
          <span className="font-mono text-[#F2FAF5] text-[11px]">
            {model.architecture}
          </span>
        </div>

        <div className="flex justify-between items-center text-[#A6B9AE]">
          <span>Weights Checkpoint:</span>
          <span className="font-mono text-[#6AE6A0] text-[10px] truncate max-w-[200px]" title={model.checkpoint}>
            {model.checkpoint.split("/").pop()}
          </span>
        </div>

        {model.device && (
          <div className="flex justify-between items-center text-[#A6B9AE]">
            <span>Acceleration:</span>
            <span className="font-mono text-[#8EF0AF] text-[11px] font-semibold">
              {model.device}
            </span>
          </div>
        )}

        {/* Dynamic Metric Rows */}
        {Object.entries(model.metrics).map(([key, val]) => (
          <div key={key} className="flex justify-between items-center text-[#A6B9AE]">
            <span>{key}:</span>
            <span className="font-mono font-bold text-[#F2FAF5] text-[11px]">
              {val}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

