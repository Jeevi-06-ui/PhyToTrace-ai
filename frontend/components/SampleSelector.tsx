"use client";

import { useEffect, useState } from "react";
import { Check, Sparkles, Image as ImageIcon } from "lucide-react";

interface SampleImage {
  filename: string;
  label: string;
  category: string;
  description: string;
  url: string;
}

interface SampleSelectorProps {
  onSelectSample: (sample: SampleImage, file: File) => void;
  isLoading?: boolean;
}

export default function SampleSelector({
  onSelectSample,
  isLoading,
}: SampleSelectorProps) {
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/samples")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setSamples(data))
      .catch((err) => console.error("Could not load samples:", err));
  }, []);

  const handleSelect = async (sample: SampleImage) => {
    setSelectedFilename(sample.filename);
    try {
      const resp = await fetch(sample.url);
      if (!resp.ok) throw new Error("Sample image is unavailable.");
      const blob = await resp.blob();
      const file = new File([blob], sample.filename, {
        type: blob.type || "image/jpeg",
      });
      onSelectSample(sample, file);
    } catch (e) {
      console.error("Failed to load sample blob:", e);
    }
  };

  if (samples.length === 0) return null;

  return (
    <div className="bg-[#152C22] border border-[#294B3C] rounded-2xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-xs font-bold text-[#F2FAF5] uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5 text-[#6AE6A0]" />
          <span>Quick Benchmark Samples (1-Click Test)</span>
        </div>
        <span className="text-[11px] text-[#768D80] font-mono">
          Authentic Dataset Samples
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {samples.map((s) => {
          const isSelected = selectedFilename === s.filename;
          const isHealthy = s.category === "healthy";
          return (
            <button
              key={s.filename}
              onClick={() => handleSelect(s)}
              disabled={isLoading}
              className={`p-3 rounded-xl border text-left transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? "bg-[#1D4032] border-[#6AE6A0] ring-2 ring-[#6AE6A0]/25 shadow-md shadow-[#6AE6A0]/10"
                  : "bg-[#10231B] border-[#294B3C] hover:border-[#6AE6A0]/40 hover:bg-[#152C22]"
              } ${isLoading ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <div>
                <div className="flex items-center justify-between gap-1 mb-1.5">
                  <span className="font-semibold text-xs text-[#F2FAF5] truncate">
                    {s.label}
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase tracking-wider ${
                      isHealthy
                        ? "bg-[#152C22] text-[#6AE6A0] border-[#6AE6A0]/30"
                        : "bg-[#271515] text-[#F17878] border-[#F17878]/30"
                    }`}
                  >
                    {isHealthy ? "Healthy" : "Leaf Miner"}
                  </span>
                </div>
                <p className="text-[11px] text-[#A6B9AE] line-clamp-2 leading-relaxed">
                  {s.description}
                </p>
              </div>

              <div className="mt-3 pt-2 border-t border-[#294B3C] flex items-center justify-between text-[11px] text-[#6AE6A0] font-semibold">
                <span className="flex items-center gap-1">
                  <ImageIcon className="w-3 h-3" />
                  Load & Analyze
                </span>
                {isSelected && <Check className="w-3.5 h-3.5 text-[#6AE6A0]" />}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
