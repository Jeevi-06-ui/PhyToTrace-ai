"use client";

import { useState } from "react";
import {
  Eye,
  Layers,
  Maximize2,
  Minimize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  SplitSquareVertical,
  Crosshair,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Info,
  Info as InfoIcon,
  CheckCircle2,
  Sparkles
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";

export interface EvidenceRegion {
  label?: string;
  region_id: string;
  bbox: number[];
  centroid?: number[];
  area_pixels: number;
  leaf_area_pct?: number | null;
  confidence: number;
  polygon?: number[][];
  polygons?: number[][][] | null;
  uncertainty?: string;
  damage_type?: string;
  pest_evidence?: string;
  description?: string;
  recommended_observation?: string;
}

interface LayerToggleViewerProps {
  originalUrl: string;
  annotatedUrl?: string;
  leafMaskUrl?: string;
  leafOverlayUrl?: string;
  mineMaskUrl?: string;
  mineOverlayUrl?: string;
  sideBySideUrl?: string;
  regions?: EvidenceRegion[];
  leafMethod?: string;
  status?: string;
  failureClassification?: string;
}

export default function LayerToggleViewer({
  originalUrl,
  annotatedUrl,
  leafOverlayUrl,
  mineOverlayUrl,
  leafMaskUrl,
  mineMaskUrl,
  sideBySideUrl,
  regions = [],
  leafMethod,
  status,
  failureClassification,
}: LayerToggleViewerProps) {
  const { t } = useLanguage();
  const [activeLayer, setActiveLayer] = useState<
    "composite" | "original" | "leaf" | "mine" | "sbs"
  >("composite");
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);
  const [opacity, setOpacity] = useState<number>(85);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [selectedRegion, setSelectedRegion] = useState<EvidenceRegion | null>(
    regions.length > 0 ? regions[0] : null
  );

  // Zoom handlers
  const handleZoomIn = () => setZoomLevel((z) => Math.min(3, +(z + 0.25).toFixed(2)));
  const handleZoomOut = () => setZoomLevel((z) => Math.max(0.75, +(z - 0.25).toFixed(2)));
  const handleZoomReset = () => setZoomLevel(1);

  // Determine active displayed image URLs
  let overlayImageUrl: string | null = null;
  if (activeLayer === "composite") {
    overlayImageUrl = annotatedUrl || null;
  } else if (activeLayer === "leaf") {
    overlayImageUrl = leafOverlayUrl || leafMaskUrl || null;
  } else if (activeLayer === "mine") {
    overlayImageUrl = mineOverlayUrl || mineMaskUrl || null;
  }

  const leafLabel = leafMethod === "unet" ? "Leaf canopy boundary" : leafMethod ? "Colour heuristic preview" : "Leaf mask preview";
  const isSideBySide = activeLayer === "sbs";

  return (
    <div
      className={`rounded-2xl border border-[#294B3C] bg-[#152C22] overflow-hidden transition-all duration-300 flex flex-col ${isFullscreen
          ? "fixed inset-4 z-50 shadow-2xl bg-[#0A1814] border-[#6AE6A0]"
          : "relative shadow-lg"
        }`}
    >
      {/* Top Toolbar */}
      <div className="p-3 bg-[#10231B] border-b border-[#294B3C] flex flex-wrap items-center justify-between gap-3">
        {/* Layer Selection Pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="flex items-center gap-1.5 mr-2 text-xs font-bold text-[#F2FAF5] uppercase tracking-wider">
            <Layers className="w-4 h-4 text-[#6AE6A0]" />
            <span className="hidden sm:inline">{t("evidence.layers_title", "Layers")}</span>
          </div>

          <button
            onClick={() => setActiveLayer("composite")}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${activeLayer === "composite"
                ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/50 shadow-sm shadow-[#6AE6A0]/20"
                : "bg-[#10231B] text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22] border border-[#294B3C]"
              }`}
          >
            {t("evidence.all_overlays", "All Overlays")}
          </button>

          <button
            onClick={() => setActiveLayer("original")}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${activeLayer === "original"
                ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/50 shadow-sm shadow-[#6AE6A0]/20"
                : "bg-[#10231B] text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22] border border-[#294B3C]"
              }`}
          >
            {t("evidence.raw_leaf", "Raw Leaf")}
          </button>

          <button
            onClick={() => setActiveLayer("leaf")}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${activeLayer === "leaf"
                ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/50 shadow-sm shadow-[#6AE6A0]/20"
                : "bg-[#10231B] text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22] border border-[#294B3C]"
              }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#6AE6A0]" />
            {t("evidence.leaf_canopy", "Leaf Canopy")}
          </button>

          <button
            onClick={() => setActiveLayer("mine")}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${activeLayer === "mine"
                ? "bg-[#271515] text-[#F17878] border border-[#F17878]/50 shadow-sm shadow-[#F17878]/20"
                : "bg-[#10231B] text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22] border border-[#294B3C]"
              }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#F17878]" />
            {t("evidence.mine_damage", "Mine Damage")}
          </button>

          {sideBySideUrl && (
            <button
              onClick={() => setActiveLayer("sbs")}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${activeLayer === "sbs"
                  ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/50 shadow-sm shadow-[#6AE6A0]/20"
                  : "bg-[#10231B] text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22] border border-[#294B3C]"
                }`}
            >
              <SplitSquareVertical className="w-3.5 h-3.5" />
              {t("evidence.side_by_side", "Side-by-Side")}
            </button>
          )}
        </div>

        {/* Zoom & Opacity Controls */}
        <div className="flex items-center gap-3">
          {activeLayer !== "original" && activeLayer !== "sbs" && overlayImageUrl && (
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-xl bg-[#152C22] border border-[#294B3C]">
              <Sliders className="w-3.5 h-3.5 text-[#6AE6A0]" />
              <span className="text-[11px] text-[#A6B9AE] hidden md:inline">
                {t("evidence.opacity", "Opacity")}
              </span>
              <input
                type="range"
                min="10"
                max="100"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="w-16 sm:w-20 accent-[#6AE6A0] h-1.5 bg-[#10231B] rounded-lg cursor-pointer"
                title={`Overlay Opacity: ${opacity}%`}
              />
              <span className="text-[10px] font-mono text-[#F2FAF5] w-7 text-right">
                {opacity}%
              </span>
            </div>
          )}

          {/* Zoom Buttons */}
          <div className="flex items-center gap-1 bg-[#152C22] p-1 rounded-xl border border-[#294B3C]">
            <button
              onClick={handleZoomOut}
              disabled={zoomLevel <= 0.75}
              className="p-1.5 rounded-lg text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#1D4032] disabled:opacity-40 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] font-mono text-[#6AE6A0] px-1 font-semibold">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              disabled={zoomLevel >= 3}
              className="p-1.5 rounded-lg text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#1D4032] disabled:opacity-40 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleZoomReset}
              className="p-1.5 rounded-lg text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#1D4032] transition-colors"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Fullscreen Toggle */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 rounded-xl bg-[#152C22] border border-[#294B3C] text-[#A6B9AE] hover:text-[#6AE6A0] hover:bg-[#1D4032] transition-colors"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen View"}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Main Viewport */}
      <div
        className={`relative bg-[#0A1814] flex overflow-auto p-4 select-none ${isFullscreen ? "flex-1 min-h-[600px]" : "min-h-[420px] max-h-[620px]"
          }`}
      >
        <div
          className="relative m-auto inline-block shrink-0"
          style={{ zoom: zoomLevel }}
        >
          {isSideBySide && sideBySideUrl ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={sideBySideUrl}
              alt="PhytoTrace Side-by-Side Comparison"
              className="max-h-[540px] w-auto rounded-xl shadow-2xl object-contain border border-[#294B3C]"
            />
          ) : (
            <div className="relative">
              {/* Base Raw Photograph */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={originalUrl}
                alt="PhytoTrace Original Tomato Leaf"
                onLoad={(event) => {
                  const { naturalWidth, naturalHeight } = event.currentTarget;
                  setImageSize([naturalWidth, naturalHeight]);
                }}
                className="max-h-[540px] w-auto rounded-xl shadow-2xl object-contain block outline outline-[#294B3C]"
              />

              {/* Pixel-aligned Overlay Layer */}
              {activeLayer !== "original" && overlayImageUrl && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={overlayImageUrl}
                  alt="PhytoTrace Explainability Overlay"
                  className="absolute inset-0 w-full h-full object-contain rounded-xl pointer-events-none transition-opacity duration-200"
                  style={{ opacity: opacity / 100 }}
                />
              )}
              {imageSize && regions.length > 0 && (activeLayer === "composite" || activeLayer === "mine") && (
                <svg
                  viewBox={`0 0 ${imageSize[0]} ${imageSize[1]}`}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  aria-label="Model evidence outlines"
                >
                  {regions.map((region) => (
                    <g key={region.region_id}>
                      {(region.polygons ?? (region.polygon ? [region.polygon] : [])).map((polygon, index) => (
                        <polygon
                          key={index}
                          points={polygon.map((point) => point.join(",")).join(" ")}
                          fill={selectedRegion?.region_id === region.region_id ? "#6AE6A055" : "#F1BA6533"}
                          stroke={selectedRegion?.region_id === region.region_id ? "#6AE6A0" : "#F1BA65"}
                          strokeWidth={2}
                          vectorEffect="non-scaling-stroke"
                          className="pointer-events-auto cursor-pointer"
                          onClick={() => setSelectedRegion(region)}
                        />
                      ))}
                    </g>
                  ))}
                </svg>
              )}
            </div>
          )}
        </div>

        {/* Floating Active Layer Badge */}
        <div className="absolute bottom-4 left-4 bg-[#10231B]/90 backdrop-blur-md border border-[#294B3C] px-3.5 py-1.5 rounded-xl text-xs font-medium text-[#F2FAF5] flex items-center gap-2 shadow-lg">
          <Eye className="w-3.5 h-3.5 text-[#6AE6A0]" />
          <span>
            {activeLayer === "composite" &&
              "Model evidence overlay"}
            {activeLayer === "original" && "Original Unprocessed Tomato Leaf"}
            {activeLayer === "leaf" && leafLabel}
            {activeLayer === "mine" && "Detected damage regions"}
            {activeLayer === "sbs" && "Side-by-Side Diagnostic Comparison"}
          </span>
        </div>
      </div>

      {/* Empty State Banner when no regions are detected */}
      {regions.length === 0 && (
        <div className="border-t border-[#294B3C] bg-[#10231B] p-4 flex items-center justify-between">
          {(status === "LEAF_NOT_DETECTED" || failureClassification === "LEAF_NOT_DETECTED") ? (
            <>
              <div className="flex items-center gap-2.5">
                <ShieldAlert className="w-4 h-4 text-[#F17878]" />
                <span className="text-xs font-bold text-[#F17878]">
                  Tomato Leaf Canopy Not Detected
                </span>
              </div>
              <span className="text-[11px] text-[#A6B9AE]">
                Foliage could not be isolated from background. Use Manual Leaf Focus or provide a closer photograph.
              </span>
            </>
          ) : (status === "LEAF_MASK_UNRELIABLE" || failureClassification === "LEAF_MASK_UNRELIABLE") ? (
            <>
              <div className="flex items-center gap-2.5">
                <InfoIcon className="w-4 h-4 text-[#F1BA65]" />
                <span className="text-xs font-bold text-[#F1BA65]">
                  Canopy Boundary Uncalibrated — Health Status Uncertain
                </span>
              </div>
              <span className="text-[11px] text-[#A6B9AE]">
                Damage area is unavailable. Outline the target leaflet to inspect candidates within it.
              </span>
            </>
          ) : (status === "IMAGE_QUALITY_FAILURE" || failureClassification === "IMAGE_QUALITY_FAILURE") ? (
            <>
              <div className="flex items-center gap-2.5">
                <ShieldAlert className="w-4 h-4 text-[#F17878]" />
                <span className="text-xs font-bold text-[#F17878]">
                  Image Quality Assessment Failed
                </span>
              </div>
              <span className="text-[11px] text-[#A6B9AE]">
                Photograph blur or illumination outside acceptable parameters.
              </span>
            </>
          ) : (status === "NO_QUALIFYING_MINE_DETECTION" || failureClassification === "NO_QUALIFYING_MINE_DETECTION") ? (
            <>
              <div className="flex items-center gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#6AE6A0]" />
                <span className="text-xs font-bold text-[#F2FAF5]">
                  No Qualifying Leaf-Miner Lesions Localized
                </span>
              </div>
              <span className="text-[11px] text-[#A6B9AE]">
                The model returned no qualifying regions. Visible markings still need field review.
              </span>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2.5">
                <InfoIcon className="w-4 h-4 text-[#F1BA65]" />
                <span className="text-xs font-bold text-[#F2FAF5]">
                  No Accepted Evidence Regions
                </span>
              </div>
              <span className="text-[11px] text-[#A6B9AE]">
                See scan telemetry for the model candidates and their review decisions.
              </span>
            </>
          )}
        </div>
      )}

      {/* Interactive Evidence Region Selector (A, B, C...) */}
      {regions.length > 0 && (
        <div className="border-t border-[#294B3C] bg-[#10231B] p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-[#6AE6A0]" />
              <span className="text-xs font-bold text-[#F2FAF5] uppercase tracking-wider">
                {t("evidence.detected_regions", "Detected Evidence Regions")} ({regions.length})
              </span>
            </div>
            <span className="text-[11px] text-[#A6B9AE]">
              Click a region (A, B, C...) for deep evidence details
            </span>
          </div>

          {/* Region Tabs / Pills */}
          <div className="flex flex-wrap items-center gap-2">
            {regions.map((reg) => {
              const label = reg.label || reg.region_id;
              const isSelected = selectedRegion?.region_id === reg.region_id;
              return (
                <button
                  key={reg.region_id}
                  onClick={() => setSelectedRegion(reg)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${isSelected
                      ? "bg-[#6AE6A0] text-[#0A1814] shadow-md shadow-[#6AE6A0]/30 scale-105"
                      : "bg-[#152C22] text-[#F2FAF5] hover:bg-[#1D4032] border border-[#294B3C]"
                    }`}
                >
                  <span className="w-5 h-5 rounded-full bg-[#10231B] text-[#6AE6A0] flex items-center justify-center text-[11px] font-black">
                    {label}
                  </span>
                  <span>{reg.area_pixels} px</span>
                  {reg.leaf_area_pct != null && (
                    <span className="text-[10px] opacity-80">({reg.leaf_area_pct}%)</span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Selected Region Detailed Evidence Card */}
          {selectedRegion && (
            <div className="p-4 rounded-xl bg-[#152C22] border border-[#6AE6A0]/40 shadow-inner space-y-2 mt-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-[#6AE6A0] text-[#0A1814] flex items-center justify-center font-black text-xs">
                    {selectedRegion.label || "R"}
                  </span>
                  <span className="font-bold text-sm text-[#F2FAF5]">
                    {t("evidence.region", "Region")} {selectedRegion.label || selectedRegion.region_id}
                  </span>
                  <span className="text-xs font-mono text-[#A6B9AE]">
                    ({selectedRegion.region_id})
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {selectedRegion.pest_evidence && (
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#10231B] text-[#6AE6A0] border border-[#6AE6A0]/40 capitalize">
                      {selectedRegion.pest_evidence === "liriomyza" ? "Liriomyza (Serpentine)" : "Tuta absoluta (Blotch)"}
                    </span>
                  )}
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#271515] text-[#F17878] border border-[#F17878]/30">
                    Confidence: {(selectedRegion.confidence * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#10231B] text-[#6AE6A0] border border-[#6AE6A0]/30">
                    {selectedRegion.leaf_area_pct == null ? "—" : `${selectedRegion.leaf_area_pct}%`} {t("evidence.leaf_coverage", "Canopy")}
                  </span>
                </div>
              </div>

              <p className="text-[11px] text-[#A6B9AE]">
                {selectedRegion.area_pixels.toLocaleString()} px · Bounding box [{selectedRegion.bbox.join(", ")}]
              </p>
              <p className="text-xs text-[#F2FAF5] leading-relaxed pt-1">
                {selectedRegion.description ||
                  "Experimental region for field review."}
              </p>

              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pt-2 border-t border-[#294B3C] text-[11px] text-[#A6B9AE]">
                <div className="flex items-center gap-1 text-[#F1BA65]">
                  <Info className="w-3.5 h-3.5 shrink-0" />
                  <span>{selectedRegion.uncertainty ?? t("evidence.uncertainty_note", "Visual pattern alone cannot confirm species without larval microscopic confirmation.")}</span>
                </div>
                <div className="font-semibold text-[#6AE6A0]">
                  {selectedRegion.recommended_observation || "Inspect lower leaf surface with 10x hand lens."}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
