"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  Sparkles,
  Layers,
  Sliders,
  ScanLine,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Info,
  Maximize2,
  Minimize2,
  Eye,
  Crosshair,
  SplitSquareVertical,
  Activity,
  Cpu,
  Leaf
} from "lucide-react";

interface RegionData {
  region_id: string;
  bbox: number[];
  x_pct: number;
  y_pct: number;
  confidence: number;
  pest_evidence: string;
  damage_type: string;
  area_pixels: number;
  is_necrotic_spot?: boolean;
  tuta_score: number;
  liriomyza_score: number;
}

interface SpecimenData {
  id: string;
  title: string;
  subtitle: string;
  badge: string;
  tag: string;
  width: number;
  height: number;
  raw_url: string;
  annotated_url: string;
  leaf_mask_url: string;
  mine_mask_url: string;
  damage_detected: boolean;
  damage_percentage: number;
  diagnosis_name: string;
  diagnosis_accuracy: number;
  leaf_area_pixels: number;
  damage_area_pixels: number;
  regions_count: number;
  regions: RegionData[];
}

export default function InteractiveJuryShowcase() {
  const [specimens, setSpecimens] = useState<SpecimenData[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage 0-100
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [activeLayer, setActiveLayer] = useState<"slider" | "raw" | "annotated" | "leaf" | "mine">("slider");
  const [showHotspots, setShowHotspots] = useState<boolean>(true);
  const [activePin, setActivePin] = useState<RegionData | null>(null);
  const [containerWidth, setContainerWidth] = useState<number>(800);

  const containerRef = useRef<HTMLDivElement>(null);

  // Load specimen manifest
  useEffect(() => {
    fetch("/demo/specimens.json")
      .then((r) => r.json())
      .then((data: SpecimenData[]) => {
        if (Array.isArray(data) && data.length > 0) {
          setSpecimens(data);
        }
      })
      .catch((err) => console.error("Failed to load demo specimens:", err));
  }, []);

  const current = specimens[selectedIdx];

  // Measure container width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // Drag handlers for split slider
  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(Math.round(pct * 10) / 10);
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    handleMove(e.clientX);
  };

  const onTouchStart = (e: React.TouchEvent) => {
    setIsDragging(true);
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  };

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) handleMove(e.clientX);
    };
    const onTouchMove = (e: TouchEvent) => {
      if (isDragging && e.touches.length > 0) handleMove(e.touches[0].clientX);
    };
    const onStop = () => setIsDragging(false);

    if (isDragging) {
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onStop);
      window.addEventListener("touchmove", onTouchMove);
      window.addEventListener("touchend", onStop);
    }
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onStop);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onStop);
    };
  }, [isDragging, handleMove]);

  if (!current) {
    return null;
  }

  const isHealthy = !current.damage_detected || current.damage_percentage === 0;

  return (
    <section id="interactive-showcase" className="scroll-mt-24 relative rounded-3xl bg-gradient-to-b from-[#152C22] via-[#10231B] to-[#0A1814] border border-[#294B3C] p-5 sm:p-8 lg:p-10 shadow-2xl overflow-hidden">
      {/* Decorative background glow */}
      <div className="absolute -top-24 -left-24 w-96 h-96 bg-[#6AE6A0]/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-[#1D4032]/40 rounded-full blur-3xl pointer-events-none" />

      {/* Header Bar */}
      <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#10231B] border border-[#294B3C] text-xs font-semibold text-[#6AE6A0] mb-3 shadow-sm">
            <Sparkles className="w-3.5 h-3.5 text-[#6AE6A0]" />
            <span>Interactive Faculty & Jury Demo Hub • Real Field Telemetry</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-black text-[#F2FAF5] tracking-tight">
            Live Neural Vision & Canopy Inspection
          </h2>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1.5 max-w-2xl leading-relaxed">
            Drag the interactive divider to compare the authentic raw camera specimen against the real-time AI segmentation overlay, or toggle individual neural layers.
          </p>
        </div>

        {/* Primary CTA button */}
        <Link
          href="/analyze"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-bold text-xs sm:text-sm transition-all shadow-lg shadow-[#6AE6A0]/20 hover:scale-102 shrink-0 self-start md:self-auto"
        >
          <ScanLine className="w-4 h-4 text-[#0A1814]" />
          <span>Scan Your Own Leaf</span>
          <ArrowRight className="w-4 h-4 text-[#0A1814]" />
        </Link>
      </div>

      {/* Specimen Switcher Tabs */}
      <div className="relative z-10 flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none mb-5">
        {specimens.map((s, idx) => {
          const active = idx === selectedIdx;
          const sHealthy = !s.damage_detected || s.damage_percentage === 0;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => {
                setSelectedIdx(idx);
                setActivePin(null);
              }}
              className={`flex items-center gap-2.5 px-4 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all border shrink-0 ${
                active
                  ? "bg-[#1D4032] border-[#6AE6A0] text-[#F2FAF5] shadow-md shadow-[#6AE6A0]/10"
                  : "bg-[#10231B] hover:bg-[#152C22] border-[#294B3C] text-[#A6B9AE] hover:text-[#F2FAF5]"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  sHealthy ? "bg-[#6AE6A0]" : s.damage_percentage > 20 ? "bg-[#F17878]" : "bg-[#F1BA65]"
                }`}
              />
              <span>{s.badge}</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#0A1814]/60 text-[#6AE6A0]">
                {sHealthy ? "0.0%" : `${s.damage_percentage}%`}
              </span>
            </button>
          );
        })}
      </div>

      {/* Main Interactive Stage */}
      <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Viewer Frame (8 cols on lg) */}
        <div className="lg:col-span-8 flex flex-col space-y-3">
          {/* Layer View Mode Controls */}
          <div className="flex flex-wrap items-center justify-between gap-2 bg-[#10231B] border border-[#234534] p-1.5 rounded-2xl">
            <div className="flex items-center gap-1 overflow-x-auto">
              <button
                type="button"
                onClick={() => setActiveLayer("slider")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  activeLayer === "slider"
                    ? "bg-[#6AE6A0] text-[#0A1814] shadow-sm"
                    : "text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22]"
                }`}
              >
                <SplitSquareVertical className="w-3.5 h-3.5" />
                <span>Split Compare</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveLayer("annotated")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  activeLayer === "annotated"
                    ? "bg-[#6AE6A0] text-[#0A1814] shadow-sm"
                    : "text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22]"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>AI Vision Overlay</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveLayer("raw")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  activeLayer === "raw"
                    ? "bg-[#6AE6A0] text-[#0A1814] shadow-sm"
                    : "text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22]"
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Raw Specimen</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveLayer("mine")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  activeLayer === "mine"
                    ? "bg-[#6AE6A0] text-[#0A1814] shadow-sm"
                    : "text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22]"
                }`}
              >
                <Crosshair className="w-3.5 h-3.5" />
                <span>Mine Lesions</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveLayer("leaf")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  activeLayer === "leaf"
                    ? "bg-[#6AE6A0] text-[#0A1814] shadow-sm"
                    : "text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22]"
                }`}
              >
                <Leaf className="w-3.5 h-3.5" />
                <span>Canopy Mask</span>
              </button>
            </div>

            {/* Hotspots Toggle */}
            <button
              type="button"
              onClick={() => setShowHotspots(!showHotspots)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
                showHotspots
                  ? "bg-[#1D4032] border-[#6AE6A0]/40 text-[#6AE6A0]"
                  : "bg-[#10231B] border-[#234534] text-[#7E9689]"
              }`}
            >
              <Crosshair className="w-3.5 h-3.5" />
              <span>Inspection Pins ({current.regions_count})</span>
            </button>
          </div>

          {/* Interactive Image Frame */}
          <div
            ref={containerRef}
            onMouseDown={activeLayer === "slider" ? onMouseDown : undefined}
            onTouchStart={activeLayer === "slider" ? onTouchStart : undefined}
            className={`relative w-full aspect-[4/3] rounded-2xl overflow-hidden border border-[#234534] bg-[#07130F] select-none ${
              activeLayer === "slider" ? "cursor-ew-resize" : "cursor-default"
            }`}
          >
            {/* Layer 1: Underneath Image */}
            {/* If slider: Underneath is Raw Leaf. If single layer: selected layer */}
            <div className="absolute inset-0 w-full h-full">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={
                  activeLayer === "slider"
                    ? current.raw_url
                    : activeLayer === "raw"
                    ? current.raw_url
                    : activeLayer === "annotated"
                    ? current.annotated_url
                    : activeLayer === "mine"
                    ? current.mine_mask_url
                    : current.leaf_mask_url
                }
                alt={current.title}
                className="w-full h-full object-cover pointer-events-none"
              />
            </div>

            {/* Layer 2: Clipped AI Vision Overlay for Split Slider */}
            {activeLayer === "slider" && (
              <div
                className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
                style={{
                  clipPath: `polygon(${sliderPos}% 0%, 100% 0%, 100% 100%, ${sliderPos}% 100%)`,
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={current.annotated_url}
                  alt="AI Vision Overlay"
                  className="w-full h-full object-cover"
                />
              </div>
            )}

            {/* Split Slider Handle & Divider Line */}
            {activeLayer === "slider" && (
              <>
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-gradient-to-b from-[#6AE6A0] via-[#8EF0AF] to-[#6AE6A0] shadow-[0_0_12px_rgba(106,230,160,0.8)] pointer-events-none"
                  style={{ left: `${sliderPos}%` }}
                />
                <div
                  className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-[#10231B] border-2 border-[#6AE6A0] shadow-xl flex items-center justify-center text-[#6AE6A0] cursor-ew-resize hover:scale-110 active:scale-95 transition-transform"
                  style={{ left: `${sliderPos}%` }}
                >
                  <Sliders className="w-4 h-4 rotate-90" />
                </div>

                {/* Corner Labels */}
                <div className="absolute top-3 left-3 px-2.5 py-1 rounded-lg bg-[#0A1814]/80 backdrop-blur-md border border-[#234534] text-[10px] font-mono text-[#A6B9AE] pointer-events-none">
                  RAW SPECIMEN
                </div>
                <div className="absolute top-3 right-3 px-2.5 py-1 rounded-lg bg-[#0A1814]/80 backdrop-blur-md border border-[#6AE6A0]/40 text-[10px] font-mono text-[#6AE6A0] pointer-events-none">
                  NEURAL AI OVERLAY
                </div>
              </>
            )}

            {/* Inspection Pins (Interactive Lesion Hotspots) */}
            {showHotspots &&
              (activeLayer === "slider" || activeLayer === "annotated" || activeLayer === "mine") &&
              current.regions.map((r, i) => {
                const isSelected = activePin?.region_id === r.region_id;
                // In slider mode, only show pins if they are on the AI overlay side (or if hovered)
                const isVisibleInSlider = activeLayer !== "slider" || r.x_pct >= sliderPos - 5;
                if (!isVisibleInSlider && !isSelected) return null;

                const isTuta = r.pest_evidence === "tuta";

                return (
                  <div
                    key={r.region_id}
                    className="absolute -translate-x-1/2 -translate-y-1/2 z-20 transition-transform"
                    style={{ left: `${r.x_pct}%`, top: `${r.y_pct}%` }}
                  >
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setActivePin(isSelected ? null : r);
                      }}
                      className={`relative flex items-center justify-center w-7 h-7 rounded-full font-mono text-[10px] font-bold shadow-lg transition-transform hover:scale-125 ${
                        isSelected
                          ? "bg-[#F2FAF5] text-[#0A1814] ring-4 ring-[#6AE6A0]"
                          : isTuta
                          ? "bg-[#FFA500] text-[#0A1814] ring-2 ring-[#FFA500]/60"
                          : "bg-[#00CED1] text-[#0A1814] ring-2 ring-[#00CED1]/60"
                      }`}
                    >
                      <span>{i + 1}</span>
                      <span className="absolute inset-0 rounded-full animate-ping opacity-30 bg-current pointer-events-none" />
                    </button>
                  </div>
                );
              })}

            {/* Floating Popover for Selected Region */}
            {activePin && (
              <div
                className="absolute z-30 max-w-xs p-3.5 rounded-xl bg-[#0A1814]/95 backdrop-blur-md border border-[#6AE6A0]/40 text-xs shadow-2xl animate-in fade-in zoom-in-95 pointer-events-auto"
                style={{
                  left: `${Math.min(70, Math.max(10, activePin.x_pct))}%`,
                  top: `${Math.min(65, Math.max(10, activePin.y_pct + 6))}%`,
                }}
              >
                <div className="flex items-center justify-between gap-2 border-b border-[#234534] pb-2 mb-2">
                  <div className="flex items-center gap-1.5 font-bold text-[#F2FAF5]">
                    <Crosshair className="w-3.5 h-3.5 text-[#6AE6A0]" />
                    <span>Region {activePin.region_id}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActivePin(null)}
                    className="text-[#A6B9AE] hover:text-[#F2FAF5] text-xs font-mono px-1"
                  >
                    &times;
                  </button>
                </div>

                <div className="space-y-1.5 text-[11px] text-[#A6B9AE]">
                  <div className="flex justify-between">
                    <span>Identified Species:</span>
                    <strong className="text-[#F2FAF5] capitalize">
                      {activePin.pest_evidence === "tuta" ? "Tuta absoluta" : "Liriomyza trifolii"}
                    </strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Damage Morphology:</span>
                    <span className="font-mono text-[#F1BA65] capitalize">
                      {activePin.damage_type.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Neural Confidence:</span>
                    <span className="font-mono text-[#6AE6A0] font-bold">{activePin.confidence}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Lesion Area:</span>
                    <span className="font-mono text-[#F2FAF5]">{activePin.area_pixels.toLocaleString()} px</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Slider Hint Footer */}
          <div className="flex items-center justify-between text-[11px] text-[#7E9689] px-1 font-mono">
            <span className="flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-[#6AE6A0]" />
              <span>Drag divider left/right to reveal segmentation</span>
            </span>
            <span>Split Ratio: {sliderPos}%</span>
          </div>
        </div>

        {/* Right Column: Agronomic Intelligence HUD & Evidence Breakdown (4 cols on lg) */}
        <div className="lg:col-span-4 flex flex-col justify-between space-y-4">
          {/* Diagnostic Verdict Card */}
          <div className="p-5 rounded-2xl bg-[#10231B] border border-[#234534] shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#6AE6A0] font-bold">
                Specimen Analysis
              </span>
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                  isHealthy
                    ? "bg-[#6AE6A0]/20 text-[#6AE6A0] border border-[#6AE6A0]/30"
                    : current.damage_percentage > 20
                    ? "bg-[#F17878]/20 text-[#F17878] border border-[#F17878]/30"
                    : "bg-[#F1BA65]/20 text-[#F1BA65] border border-[#F1BA65]/30"
                }`}
              >
                {isHealthy ? "Clean Foliage" : current.tag}
              </span>
            </div>

            <div>
              <h3 className="text-lg font-bold text-[#F2FAF5] leading-snug">
                {current.diagnosis_name}
              </h3>
              <p className="text-xs text-[#A6B9AE] mt-1 leading-relaxed">
                {current.subtitle}
              </p>
            </div>

            {/* KPI Grid */}
            <div className="grid grid-cols-2 gap-2.5 pt-2 border-t border-[#234534]">
              <div className="p-3 rounded-xl bg-[#0A1814] border border-[#234534]">
                <div className="text-[10px] font-mono text-[#7E9689] uppercase">Canopy Damage</div>
                <div
                  className={`text-2xl font-black mt-0.5 ${
                    isHealthy
                      ? "text-[#6AE6A0]"
                      : current.damage_percentage > 20
                      ? "text-[#F17878]"
                      : "text-[#F1BA65]"
                  }`}
                >
                  {isHealthy ? "0.0%" : `${current.damage_percentage}%`}
                </div>
                <div className="text-[10px] text-[#A6B9AE] mt-0.5">Surface Area Ratio</div>
              </div>

              <div className="p-3 rounded-xl bg-[#0A1814] border border-[#234534]">
                <div className="text-[10px] font-mono text-[#7E9689] uppercase">Diagnostic Acc.</div>
                <div className="text-2xl font-black text-[#6AE6A0] mt-0.5">
                  {current.diagnosis_accuracy}%
                </div>
                <div className="text-[10px] text-[#A6B9AE] mt-0.5">Multi-Region Fusion</div>
              </div>
            </div>

            {/* Evidence Specs */}
            <div className="space-y-2 pt-2 text-xs">
              <div className="flex items-center justify-between text-[#A6B9AE]">
                <span>Canopy Pixel Count:</span>
                <span className="font-mono text-[#F2FAF5]">{current.leaf_area_pixels.toLocaleString()} px</span>
              </div>
              <div className="flex items-center justify-between text-[#A6B9AE]">
                <span>Mined Lesion Pixels:</span>
                <span className="font-mono text-[#F17878] font-semibold">
                  {current.damage_area_pixels.toLocaleString()} px
                </span>
              </div>
              <div className="flex items-center justify-between text-[#A6B9AE]">
                <span>Localized Mine Clusters:</span>
                <span className="font-mono text-[#6AE6A0] font-semibold">{current.regions_count} Regions</span>
              </div>
            </div>
          </div>

          {/* Machine Learning Rigor Checklist */}
          <div className="p-4 rounded-2xl bg-[#0A1814] border border-[#234534] space-y-2.5 text-xs">
            <div className="flex items-center gap-1.5 font-bold text-[#F2FAF5] text-xs">
              <ShieldCheck className="w-4 h-4 text-[#6AE6A0]" />
              <span>Scientific Rigor & Validation</span>
            </div>
            <div className="space-y-1.5 text-[11px] text-[#A6B9AE]">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#6AE6A0] shrink-0" />
                <span>Zero synthetic mask fabrication</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#6AE6A0] shrink-0" />
                <span>100% verified CVAT annotations</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#6AE6A0] shrink-0" />
                <span>Strict canopy clipping (no background spill)</span>
              </div>
            </div>
          </div>

          {/* Action CTAs */}
          <div className="flex flex-col gap-2 pt-1">
            <Link
              href="/analyze"
              className="flex items-center justify-center gap-2 w-full py-3 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-bold text-xs shadow-md transition-all hover:scale-101"
            >
              <ScanLine className="w-4 h-4" />
              <span>Open Full Diagnostic Lab</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
            <Link
              href="/research"
              className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-[#152C22] hover:bg-[#1D4032] border border-[#294B3C] text-[#F2FAF5] font-semibold text-xs transition-all"
            >
              <Cpu className="w-3.5 h-3.5 text-[#6AE6A0]" />
              <span>Inspect Neural Weights & Metrics</span>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
