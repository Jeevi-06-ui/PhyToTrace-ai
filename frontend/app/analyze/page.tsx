"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CheckCircle,
  Download,
  Eye,
  FileText,
  HelpCircle,
  History,
  Layers,
  Leaf,
  RefreshCw,
  Scan,
  ScanLine,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Upload,
  X,
  Cpu,
  Info,
  Clock,
  Crosshair,
  Crop,
  AlertTriangle
} from "lucide-react";
import LayerToggleViewer, { type EvidenceRegion } from "@/components/LayerToggleViewer";
import SampleSelector from "@/components/SampleSelector";
import SeverityGauge from "@/components/SeverityGauge";
import { useLanguage } from "@/context/LanguageContext";
import { useAuth } from "@/context/AuthContext";
import StatusIndicator from "@/components/StatusIndicator";
import VoiceNarrationButton from "@/components/VoiceNarrationButton";

interface PredictionData {
  inspection_id?: string;
  image_id: string;
  farm_id?: string;
  plant_id?: string;
  leaf_id?: string;
  model_version: string;
  status?: string;
  quality_status: string;
  damage_detected: boolean | null;
  damage_regions: EvidenceRegion[];
  leaf_area_pixels: number | null;
  damage_area_pixels: number | null;
  damage_percentage: number | null;
  severity_tier?: string;
  tuta_evidence?: number | null;
  liriomyza_evidence?: number | null;
  diagnosis_name?: string | null;
  diagnosis_accuracy?: number | null;
  pest_name?: string | null;
  species_breakdown?: {
    primary_pest: string;
    diagnosis_name: string;
    diagnosis_accuracy: number;
    infestation_pattern: string;
    tuta_evidence_pct: number;
    liriomyza_evidence_pct: number;
    mixed_status: string;
    leaf_miner_present: boolean;
  } | null;
  mixed_infestation_status: string;
  uncertainty_status: string;
  decision: string;
  explanation: string;
  safety_notice: string;
  limitations: string[];
  processing_time_ms: number;
  roi_mode?: string;
  roi_label?: string;
  leaf_polygon?: number[][] | null;
  leaf_boundary_source?: string | null;
  evidence_policy?: string;
  crop_box?: number[] | null;
  suggested_leaf_polygon?: number[][] | null;
  failure_classification?: string;
  diagnostic_telemetry?: any;
  rejected_candidates?: Array<{
    candidate_id: string;
    bbox: number[];
    confidence: number;
    raw_pixels: number;
    in_leaf_pixels: number;
    leaf_containment_pct: number | null;
    reasons: string[];
  }>;
  image_urls?: {
    original: string;
    annotated: string;
    leaf_mask: string;
    mine_mask: string;
    leaf_overlay: string;
    mine_overlay: string;
    side_by_side: string;
  };
  progression?: {
    previous_inspection_id: string;
    previous_date: string;
    previous_damage_percentage: number;
    progression_delta: number;
    status: string;
  } | null;
}

function AnalyzeContent() {
  const { t, locale } = useLanguage();
  const searchParams = useSearchParams();
  const queryId = searchParams.get("id");
  const { idToken, signInWithGoogle } = useAuth();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [farmId, setFarmId] = useState("GREENHOUSE_NORTH_01");
  const [plantId, setPlantId] = useState("PLANT_TOMATO_A07");
  const [leafId, setLeafId] = useState("LEAF_CANOPY_03");
  const [notes, setNotes] = useState("");
  const [leafPoints, setLeafPoints] = useState<number[][]>([]);
  const [isSelectingRoi, setIsSelectingRoi] = useState(false);
  const [previewSize, setPreviewSize] = useState([1, 1]);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState<string>("");
  const [result, setResult] = useState<PredictionData | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const previewImageRef = useRef<HTMLImageElement>(null);

  // Live camera viewfinder state
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraFacingMode, setCameraFacingMode] = useState<"environment" | "user">("environment");
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const getLocalizedDiagnosisName = (res: PredictionData): string => {
    const raw = res.diagnosis_name || res.pest_name || "";
    if (raw.toLowerCase().includes("mixed") || res.species_breakdown?.mixed_status === "confirmed_mixed") {
      return t("diagnosis.mixed", "AI-Based Detection of Leaf Miner Mixed Infestation (Tuta and Liriomyza) in Tomato");
    }
    if (raw.toLowerCase().includes("tuta") || raw.toLowerCase().includes("blotch")) {
      return t("diagnosis.tuta", "Tuta absoluta (Tomato Blotch Miner)");
    }
    if (raw.toLowerCase().includes("liriomyza") || raw.toLowerCase().includes("serpentine")) {
      return t("diagnosis.liriomyza", "Liriomyza trifolii (Serpentine Leafminer)");
    }
    if (raw.toLowerCase().includes("healthy") || res.damage_detected === false) {
      return t("diagnosis.healthy", "Healthy Tomato Foliage (Clean Canopy)");
    }
    if (raw.toLowerCase().includes("inconclusive") || res.failure_classification === "LEAF_MASK_UNRELIABLE") {
      return t("diagnosis.inconclusive", "Diagnosis Inconclusive (Boundary Unreliable)");
    }
    return raw || t(`result.${res.failure_classification}`, res.decision.replace(/_/g, " "));
  };

  const stopCameraStream = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startCameraStream = async (facingMode: "environment" | "user" = cameraFacingMode) => {
    setIsCameraLoading(true);
    setCameraError(null);
    stopCameraStream();

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access is not supported by this browser. Please use file upload.");
      setIsCameraLoading(false);
      return;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
      setIsCameraLoading(false);
    } catch (err: any) {
      console.error("Camera access error:", err);
      let msg = "Could not access camera. Please allow camera permissions in your browser.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Camera permission was denied. Please allow camera access in browser settings or use file upload.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        msg = "No camera hardware was detected on this device. Please select an image file.";
      }
      setCameraError(msg);
      setIsCameraLoading(false);
    }
  };

  const openCamera = () => {
    setIsCameraOpen(true);
    startCameraStream(cameraFacingMode);
  };

  const closeCamera = () => {
    stopCameraStream();
    setIsCameraOpen(false);
    setCameraError(null);
  };

  const flipCamera = () => {
    const nextMode = cameraFacingMode === "environment" ? "user" : "environment";
    setCameraFacingMode(nextMode);
    startCameraStream(nextMode);
  };

  const takeSnapshot = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `leaf_camera_snap_${Date.now()}.jpg`, {
          type: "image/jpeg",
        });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);
        setErrorMsg(null);
        setLeafPoints([]);
        setIsSelectingRoi(false);
        closeCamera();
      },
      "image/jpeg",
      0.95
    );
  };

  useEffect(() => {
    return () => {
      stopCameraStream();
    };
  }, []);

  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    if (!result || result.inspection_id) return;
    const expiry = setTimeout(() => setResult(null), 30 * 60 * 1000);
    return () => clearTimeout(expiry);
  }, [result]);

  // Load an existing inspection with the current authenticated identity.
  useEffect(() => {
    if (!queryId) return;
    setResult(null);
    if (!idToken) {
      setIsAnalyzing(false);
      return;
    }
    const controller = new AbortController();
    setIsAnalyzing(true);
    fetch(`/api/v1/analyses/${encodeURIComponent(queryId)}`, {
      headers: { Authorization: `Bearer ${idToken}` }, signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Inspection record could not be opened.");
        return response.json();
      })
      .then((data) => {
        if (controller.signal.aborted) return;
        if (!data.evidence_policy) {
          setErrorMsg("This inspection used the previous evidence workflow. Upload the original photograph to review it again.");
          setIsAnalyzing(false);
          return;
        }
        setResult(data);
        setFarmId(data.farm_id);
        setPlantId(data.plant_id);
        setLeafId(data.leaf_id);
        setNotes(data.notes || "");
        setIsAnalyzing(false);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setErrorMsg(error.message);
        setIsAnalyzing(false);
      });
    return () => controller.abort();
  }, [queryId, idToken]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg(null);
      setLeafPoints([]);
      setIsSelectingRoi(false);
    }
  };

  const handleSelectSample = (sample: any, file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMsg(null);
    setLeafPoints([]);
    setIsSelectingRoi(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg(null);
      setLeafPoints([]);
      setIsSelectingRoi(false);
    }
  };

  function addLeafPoint(event: React.PointerEvent<HTMLDivElement>) {
    const img = previewImageRef.current;
    if (!isSelectingRoi || !img || !event.isPrimary || event.button !== 0) return;
    const rect = img.getBoundingClientRect();
    const x = Math.max(0, Math.min(img.naturalWidth - 1, Math.round((event.clientX - rect.left) * img.naturalWidth / rect.width)));
    const y = Math.max(0, Math.min(img.naturalHeight - 1, Math.round((event.clientY - rect.top) * img.naturalHeight / rect.height)));
    setLeafPoints(points => points.length >= 100 || points.some(p => p[0] === x && p[1] === y) ? points : [...points, [x, y]]);
  }

  const handleRunDiagnosis = async (customPoints?: number[][]) => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMsg(null);
    setResult(null);
    setAnalysisStep("Uploading image and checking preflight gates...");

    const pts = (customPoints && customPoints.length >= 3) ? customPoints : leafPoints;
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("farm_id", farmId);
    formData.append("plant_id", plantId);
    formData.append("leaf_id", leafId);
    formData.append("notes", notes);
    if (pts.length >= 3) formData.append("leaf_polygon", JSON.stringify(pts));

    try {
      setAnalysisStep("Running leaf and mine segmentation...");
      const resp = await fetch("/api/v1/analyze", {
        method: "POST",
        headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined,
        body: formData,
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        let errDetail = `Server error (${resp.status})`;
        if (typeof errorData.detail === "string") {
          errDetail = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errDetail = errorData.detail
            .map((item: any) => (typeof item === "string" ? item : item.msg || JSON.stringify(item)))
            .join("; ");
        } else if (errorData.detail && typeof errorData.detail === "object") {
          errDetail = errorData.detail.msg || JSON.stringify(errorData.detail);
        } else if (errorData.message) {
          errDetail = errorData.message;
        }
        throw new Error(errDetail);
      }

      const data: PredictionData = await resp.json();
      setResult(data);
    } catch (err: any) {
      const msg =
        typeof err === "string"
          ? err
          : err?.message || (typeof err === "object" ? JSON.stringify(err) : "Failed to execute diagnostic pipeline.");
      setErrorMsg(msg);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStep("");
    }
  };

  const handleDownloadReport = () => {
    if (!result) return;
    window.print();
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#294B3C]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
              Explainable Vision Engine
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#10231B] border border-[#294B3C] text-[#A6B9AE]">
              Experimental models
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1 flex items-center gap-2.5">
            <ScanLine className="w-7 h-7 text-[#6AE6A0]" />
            <span>{t("scanner.title")}</span>
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1">
            {t("scanner.subtitle")}
          </p>
        </div>

        {result && (
          <div className="flex items-center gap-2.5">
            <button
              onClick={handleDownloadReport}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#152C22] hover:bg-[#1D4032] border border-[#294B3C] text-xs font-semibold text-[#F2FAF5] transition-all shadow-md"
            >
              <Download className="w-3.5 h-3.5 text-[#6AE6A0]" />
              <span>{t("scanner.export_btn")}</span>
            </button>
            <button
              onClick={() => {
                setResult(null);
                setSelectedFile(null);
                setPreviewUrl(null);
              }}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#10231B] hover:bg-[#152C22] border border-[#294B3C] text-xs font-semibold text-[#A6B9AE] hover:text-[#F2FAF5] transition-all"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>{t("nav.new_scan")}</span>
            </button>
          </div>
        )}
      </div>

      {/* Quick Benchmark Sample Selector */}
      <SampleSelector onSelectSample={handleSelectSample} isLoading={isAnalyzing} />

      {/* Error Alert */}
      {queryId && !idToken && (
        <button onClick={signInWithGoogle} className="rounded-xl border border-[#294B3C] p-3">Sign in to open this inspection</button>
      )}
      {errorMsg && (
        <div className="p-4 rounded-2xl bg-[#271515] border border-[#F17878]/40 text-[#F17878] text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-[#F17878]" />
          <div>
            <span className="font-bold block">Diagnosis Execution Error</span>
            <span>{errorMsg}</span>
          </div>
        </div>
      )}

      {/* Main 3-Column / Responsive Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Parameters (4 cols on wide screens) */}
        <div className="lg:col-span-4 space-y-5">
          {/* Photograph Upload Card */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg">
            <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE] block mb-3">
              {t("scanner.upload_title")}
            </span>

            {/* Hidden native inputs */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              className="hidden"
            />
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Drag & Drop Area */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${selectedFile
                  ? "border-[#6AE6A0]/50 bg-[#10231B]"
                  : "border-[#294B3C] hover:border-[#6AE6A0]/50 bg-[#10231B]/50 hover:bg-[#10231B]"
                }`}
            >
              {previewUrl ? (
                <div className="flex flex-col items-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="Uploaded preview"
                    className="max-h-40 w-auto rounded-lg object-contain shadow-md mb-3 border border-[#294B3C]"
                  />
                  <span className="text-xs font-semibold text-[#F2FAF5] truncate max-w-[220px]">
                    {selectedFile?.name || "Selected Leaf Image"}
                  </span>
                  <span className="text-[10px] text-[#A6B9AE] mt-0.5">
                    Click to replace photograph
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <div className="w-12 h-12 rounded-2xl bg-[#152C22] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0] mb-3 shadow-inner">
                    <Upload className="w-6 h-6 text-[#6AE6A0]" />
                  </div>
                  <span className="text-xs font-bold text-[#F2FAF5] mb-1">
                    {t("scanner.drop_prompt")}
                  </span>
                  <span className="text-[11px] text-[#768D80]">
                    {t("scanner.format_note")}
                  </span>
                </div>
              )}
            </div>

            {/* Camera Capture Trigger */}
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  openCamera();
                }}
                className="flex-1 py-2 px-3 rounded-xl bg-[#10231B] border border-[#294B3C] hover:bg-[#1D4032] hover:border-[#6AE6A0]/40 text-xs font-semibold text-[#A6B9AE] hover:text-[#F2FAF5] flex items-center justify-center gap-1.5 transition-colors"
              >
                <Camera className="w-3.5 h-3.5 text-[#6AE6A0]" />
                <span>{t("scanner.camera_btn")}</span>
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
                className="flex-1 py-2 px-3 rounded-xl bg-[#10231B] border border-[#294B3C] hover:bg-[#1D4032] hover:border-[#6AE6A0]/40 text-xs font-semibold text-[#A6B9AE] hover:text-[#F2FAF5] flex items-center justify-center gap-1.5 transition-colors"
              >
                <Upload className="w-3.5 h-3.5 text-[#6AE6A0]" />
                <span>{t("scanner.browse_btn")}</span>
              </button>
            </div>
          </div>

          {/* Plant & Farm Metadata Card */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg">
            <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE] block mb-3">
              {t("scanner.meta_title")}
            </span>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-[#A6B9AE] font-semibold mb-1">
                  {t("scanner.farm_label")}
                </label>
                <input
                  type="text"
                  value={farmId}
                  onChange={(e) => setFarmId(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#F2FAF5] focus:outline-hidden focus:border-[#6AE6A0]"
                />
              </div>

              <div>
                <label className="block text-[#A6B9AE] font-semibold mb-1">
                  {t("scanner.plant_label")}
                </label>
                <input
                  type="text"
                  value={plantId}
                  onChange={(e) => setPlantId(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#F2FAF5] focus:outline-hidden focus:border-[#6AE6A0]"
                />
              </div>

              <div>
                <label className="block text-[#A6B9AE] font-semibold mb-1">
                  {t("scanner.leaf_label")}
                </label>
                <input
                  type="text"
                  value={leafId}
                  onChange={(e) => setLeafId(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#F2FAF5] focus:outline-hidden focus:border-[#6AE6A0]"
                />
              </div>

              <div>
                <label className="block text-[#A6B9AE] font-semibold mb-1">
                  {t("scanner.notes_label")}
                </label>
                <input
                  type="text"
                  value={notes}
                  placeholder="e.g. Upper canopy leaflet, sunny orientation"
                  onChange={(e) => setNotes(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#F2FAF5] placeholder-[#768D80] focus:outline-hidden focus:border-[#6AE6A0]"
                />
              </div>

              {/* Execution CTA */}
              <button
                type="button"
                onClick={() => handleRunDiagnosis()}
                disabled={!selectedFile || isAnalyzing || isSelectingRoi}
                className="w-full mt-4 py-3 px-4 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-bold text-sm shadow-md shadow-[#6AE6A0]/20 flex items-center justify-center gap-2 transition-all hover:scale-102 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-[#0A1814]" />
                    <span>Analyzing Leaf Canopy...</span>
                  </>
                ) : (
                  <>
                    <ScanLine className="w-4 h-4 text-[#0A1814]" />
                    <span>{t("scanner.run_btn")}</span>
                  </>
                )}
              </button>

              {isAnalyzing && (
                <div className="mt-2 text-center text-[11px] text-[#8EF0AF] animate-pulse">
                  {analysisStep}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Center & Right Columns: Interactive Viewer & Diagnostics (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Analysis Viewer Viewport */}
          {result && result.image_urls ? (
            <LayerToggleViewer
              key={result.inspection_id ?? result.image_id}
              leafMethod={result.leaf_boundary_source ?? result.diagnostic_telemetry?.leaf_segmenter?.method}
              status={result.status}
              failureClassification={result.failure_classification}
              originalUrl={result.image_urls.original}
              annotatedUrl={result.image_urls.annotated}
              leafOverlayUrl={result.image_urls.leaf_overlay}
              mineOverlayUrl={result.image_urls.mine_overlay}
              leafMaskUrl={result.image_urls.leaf_mask}
              mineMaskUrl={result.image_urls.mine_mask}
              sideBySideUrl={result.image_urls.side_by_side}
              regions={result.damage_regions}
            />
          ) : previewUrl ? (
            <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 text-center">
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3 mb-4">
                <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE]">
                  Photograph Staged for Deep Inference
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  <button type="button" onClick={() => setIsSelectingRoi(true)} className="rounded-xl border border-[#294B3C] px-3 py-2 text-xs">Outline target leaflet</button>
                  {isSelectingRoi && (
                    <>
                      <button type="button" disabled={!leafPoints.length} onClick={() => setLeafPoints(points => points.slice(0, -1))} className="rounded-xl border border-[#294B3C] px-3 py-2 text-xs">Undo point</button>
                      <button type="button" disabled={leafPoints.length < 3} onClick={() => setIsSelectingRoi(false)} className="rounded-xl bg-[#6AE6A0] px-3 py-2 text-xs text-[#0A1814] disabled:opacity-50">Use outline</button>
                    </>
                  )}
                  {(leafPoints.length > 0 || isSelectingRoi) && (
                    <button type="button" onClick={() => { setLeafPoints([]); setIsSelectingRoi(false); }} className="rounded-xl border border-[#294B3C] px-3 py-2 text-xs">Clear outline</button>
                  )}
                </div>
              </div>
              <div className={`relative inline-block max-w-full select-none ${isSelectingRoi ? "cursor-crosshair touch-none" : ""}`} onPointerUp={addLeafPoint}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img ref={previewImageRef} src={previewUrl} alt="Selected leaf preview" draggable={false}
                  onLoad={event => setPreviewSize([event.currentTarget.naturalWidth, event.currentTarget.naturalHeight])}
                  className="block max-h-[460px] max-w-full rounded-xl shadow-2xl" />
                {leafPoints.length > 0 && (
                  <svg aria-label="Selected leaflet outline" viewBox={`0 0 ${previewSize.join(" ")}`} preserveAspectRatio="none"
                    className="pointer-events-none absolute inset-0 h-full w-full">
                    <polygon points={leafPoints.map(point => point.join(",")).join(" ")} fill="#6AE6A033" stroke="#6AE6A0" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                    {leafPoints.map(([x, y]) => <circle key={`${x},${y}`} cx={x} cy={y} r={Math.max(...previewSize) / 125} fill="#6AE6A0" />)}
                  </svg>
                )}
              </div>
              <p className="mt-4 text-xs text-[#A6B9AE]">
                {isSelectingRoi ? "Click or tap points around the leaflet in order, then choose Use outline." :
                  leafPoints.length ? "Outline selected. Verify it before running a manually assisted scan." :
                  "Run an automatic scan or outline the target leaflet first."}
              </p>
            </div>
          ) : (
            <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] border-dashed p-12 text-center flex flex-col items-center justify-center min-h-[380px]">
              <div className="w-16 h-16 rounded-2xl bg-[#10231B] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0] mb-4 shadow-inner">
                <ScanLine className="w-8 h-8 text-[#6AE6A0]" />
              </div>
              <h3 className="text-base font-bold text-[#F2FAF5] mb-1">
                Diagnostic Canvas Idle
              </h3>
              <p className="text-xs text-[#A6B9AE] max-w-sm mb-6 leading-relaxed">
                Select an authentic benchmark sample above or upload a tomato leaf photo from your field scouting session to begin.
              </p>
            </div>
          )}

          {/* Diagnostic Telemetry & Severity Gauge */}
          {result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Severity Gauge */}
              <SeverityGauge
                damagePercentage={result.damage_percentage}
                leafPixels={result.leaf_area_pixels}
                minedPixels={result.damage_area_pixels}
                severityTier={result.severity_tier || "none"}
                uncertaintyStatus={result.uncertainty_status}
                isModelPredicted={true}
              />

              {/* Agronomic Decision & Longitudinal Tracking */}
              <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE]">
                      {t("scanner.decision")}
                    </span>
                    <StatusIndicator
                      status={result.decision}
                      size="sm"
                    />
                  </div>

                  {/* AI-Based Detection Title & Diagnosis */}
                  <div className="mb-3.5 pb-3 border-b border-[#294B3C]/60">
                    <span className="text-[10px] font-mono font-bold tracking-wider uppercase px-2 py-0.5 rounded-full bg-[#10231B] border border-[#294B3C] text-[#6AE6A0] inline-block mb-1.5">
                      {t("scanner.ai_detection_badge", "AI-Based Detection of Leaf Miner Mixed Infestation (Tuta and Liriomyza) in Tomato")}
                    </span>
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <h3 className="font-black text-lg sm:text-xl text-[#F2FAF5] tracking-tight">
                        {getLocalizedDiagnosisName(result)}
                      </h3>
                      {result.diagnosis_accuracy != null && (
                        <span className="self-start sm:self-auto shrink-0 px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40 flex items-center gap-1.5 shadow-sm">
                          <Sparkles className="w-3.5 h-3.5 text-[#6AE6A0]" />
                          <span>{result.diagnosis_accuracy.toFixed(1)}% {t("scanner.accuracy", "Accuracy")}</span>
                        </span>
                      )}
                    </div>

                    {/* Voice Explanation Narration for Rural Farmers (English, Kannada, Hindi) */}
                    <div className="mt-3">
                      <VoiceNarrationButton
                        diagnosisName={getLocalizedDiagnosisName(result)}
                        accuracy={result.diagnosis_accuracy}
                        damagePercentage={result.damage_percentage}
                        severityTier={result.severity_tier}
                        explanation={result.explanation}
                        activeLocale={locale as "en" | "kn" | "hi"}
                      />
                    </div>
                  </div>

                  {/* Species Evidence Breakdown Meters (Tuta vs Liriomyza) */}
                  {result.species_breakdown && (
                    <div className="p-3 mb-3.5 rounded-xl bg-[#10231B] border border-[#294B3C] space-y-2.5">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-bold text-[#A6B9AE] uppercase tracking-wider text-[10px]">
                          {t("scanner.species_breakdown", "Species Evidence Breakdown")}
                        </span>
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold ${
                          result.species_breakdown.mixed_status === "confirmed_mixed"
                            ? "bg-[#252014] text-[#F1BA65] border border-[#F1BA65]/40"
                            : result.species_breakdown.mixed_status === "healthy"
                            ? "bg-[#152C22] text-[#6AE6A0] border border-[#294B3C]"
                            : "bg-[#1D4032] text-[#6AE6A0] border border-[#294B3C]"
                        }`}>
                          {result.species_breakdown.infestation_pattern}
                        </span>
                      </div>

                      {/* Tuta absoluta meter */}
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-[#F2FAF5] font-semibold">{t("scanner.tuta_miner", "Tuta absoluta (Blotch Miner)")}</span>
                          <span className="font-mono text-[#A6B9AE] font-bold">
                            {result.species_breakdown.tuta_evidence_pct != null
                              ? `${result.species_breakdown.tuta_evidence_pct.toFixed(1)}%`
                              : "N/A"}
                          </span>
                        </div>
                        <div className="w-full bg-[#152C22] h-2 rounded-full overflow-hidden border border-[#294B3C]">
                          <div
                            className="h-full bg-gradient-to-r from-[#F1BA65] to-[#F17878] rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, result.species_breakdown.tuta_evidence_pct ?? 0))}%` }}
                          />
                        </div>
                      </div>

                      {/* Liriomyza trifolii meter */}
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-[#F2FAF5] font-semibold">{t("scanner.liriomyza_miner", "Liriomyza trifolii (Serpentine)")}</span>
                          <span className="font-mono text-[#A6B9AE] font-bold">
                            {result.species_breakdown.liriomyza_evidence_pct != null
                              ? `${result.species_breakdown.liriomyza_evidence_pct.toFixed(1)}%`
                              : "N/A"}
                          </span>
                        </div>
                        <div className="w-full bg-[#152C22] h-2 rounded-full overflow-hidden border border-[#294B3C]">
                          <div
                            className="h-full bg-gradient-to-r from-[#4FD1C5] to-[#6AE6A0] rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, result.species_breakdown.liriomyza_evidence_pct ?? 0))}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Failure Classification & Model Uncertainty Badges */}
                  <div className="flex flex-wrap items-center gap-2 mb-3">
                    {(result.roi_mode === "manual_crop" || result.roi_mode === "manual_polygon") && (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-bold bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40">
                        <Crop className="w-3.5 h-3.5" />
                        <span>{result.roi_label || "Manually assisted analysis"}</span>
                      </span>
                    )}

                    {result.failure_classification && (
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-bold border ${result.failure_classification === "DAMAGE_DETECTED"
                          ? "bg-[#10231B] text-[#6AE6A0] border-[#6AE6A0]/40"
                          : result.failure_classification === "NO_QUALIFYING_MINE_DETECTION"
                            ? "bg-[#10231B] text-[#6AE6A0] border-[#6AE6A0]/40"
                            : result.failure_classification === "MINE_CANDIDATES_REJECTED"
                              ? "bg-[#0E2728] text-[#4FD1C5] border-[#4FD1C5]/40"
                              : result.failure_classification === "LEAF_NOT_DETECTED"
                                ? "bg-[#271515] text-[#F17878] border-[#F17878]/40"
                                : "bg-[#252014] text-[#F1BA65] border-[#F1BA65]/40"
                        }`}>
                        {result.failure_classification === "DAMAGE_DETECTED" && <CheckCircle className="w-3.5 h-3.5" />}
                        {result.failure_classification === "NO_QUALIFYING_MINE_DETECTION" && <CheckCircle className="w-3.5 h-3.5" />}
                        {result.failure_classification === "MINE_CANDIDATES_REJECTED" && <ShieldCheck className="w-3.5 h-3.5" />}
                        {result.failure_classification === "LEAF_NOT_DETECTED" && <AlertCircle className="w-3.5 h-3.5" />}
                        {result.failure_classification === "LEAF_MASK_UNRELIABLE" && <AlertTriangle className="w-3.5 h-3.5" />}
                        <span>Status: {result.failure_classification}</span>
                      </span>
                    )}

                    {result.diagnostic_telemetry?.leaf_segmenter?.method === "heuristic_fallback" && (
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-mono text-[#F1BA65] bg-[#252014] border border-[#F1BA65]/30">
                        <Info className="w-3 h-3" /> Heuristic Leaf Mask
                      </span>
                    )}
                  </div>

                  {/* Contextual Action Banners for Unreliable / Missing Leaf */}
                  {result.failure_classification === "LEAF_NOT_DETECTED" && (
                    <div className="mb-4 p-3.5 rounded-xl bg-[#271515] border border-[#F17878]/40 text-[#F17878] text-xs space-y-2">
                      <div className="flex items-center gap-2 font-bold">
                        <AlertCircle className="w-4 h-4 shrink-0" />
                        <span>Tomato Foliage Not Isolated</span>
                      </div>
                      <p className="text-[#A6B9AE] leading-relaxed">
                        The model could not isolate tomato foliage from the background. Outline the leaf using <strong>Outline target leaflet</strong> or capture a closer image.
                      </p>
                      <button
                        type="button"
                        onClick={() => { setResult(null); setIsSelectingRoi(true); }}
                        disabled={!previewUrl}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F17878] text-[#0A1814] font-bold text-xs hover:bg-[#ff9494] transition-all"
                      >
                        <Crop className="w-3.5 h-3.5" />
                        <span>Outline Target Leaflet</span>
                      </button>
                    </div>
                  )}

                  {result.failure_classification === "LEAF_MASK_UNRELIABLE" && (
                    <div className="mb-4 p-3.5 rounded-xl bg-[#252014] border border-[#F1BA65]/40 text-[#F1BA65] text-xs space-y-2">
                      <div className="flex items-center gap-2 font-bold">
                        <AlertTriangle className="w-4 h-4 shrink-0" />
                        <span>{t("scanner.canopy_uncalibrated_title", "Canopy Boundary Uncalibrated — Health Status Uncertain")}</span>
                      </div>
                      <p className="text-[#A6B9AE] leading-relaxed">
                        {t("scanner.canopy_uncalibrated_desc", "Neural leaf probabilities are uncalibrated and required heuristic fallback. Damage area is unavailable. Outline the target leaflet or use auto-calibrated contour to inspect candidates within it.")}
                      </p>
                      <div className="flex flex-wrap items-center gap-2 pt-1">
                        {result.suggested_leaf_polygon && result.suggested_leaf_polygon.length >= 3 && (
                          <button
                            type="button"
                            onClick={() => {
                              const pts = result.suggested_leaf_polygon!;
                              setLeafPoints(pts);
                              handleRunDiagnosis(pts);
                            }}
                            disabled={!selectedFile}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#6AE6A0] text-[#0A1814] font-bold text-xs hover:bg-[#8EF0AF] transition-all shadow-sm"
                          >
                            <Sparkles className="w-3.5 h-3.5" />
                            <span>{t("scanner.calibrate_and_run", "Calibrate Boundary & Run Scan")}</span>
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => {
                            if (result.suggested_leaf_polygon && result.suggested_leaf_polygon.length >= 3) {
                              setLeafPoints(result.suggested_leaf_polygon);
                            }
                            setResult(null);
                            setIsSelectingRoi(true);
                          }}
                          disabled={!previewUrl}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#F1BA65] text-[#0A1814] font-bold text-xs hover:bg-[#ffcf7f] transition-all"
                        >
                          <Crop className="w-3.5 h-3.5" />
                          <span>Focus Leaflet ROI</span>
                        </button>
                      </div>
                    </div>
                  )}

                  <p className="text-xs text-[#A6B9AE] leading-relaxed mb-4">
                    {locale === "en" ? result.explanation : t(`result.${result.failure_classification}`, result.explanation)}
                  </p>

                  {(result.roi_mode === "manual_crop" || result.roi_mode === "manual_polygon") && (
                    <p className="mb-3 text-xs text-[#A6B9AE]">Manually assisted · {result.leaf_boundary_source === "user_polygon" ? "User-selected leaf boundary" : "Rectangle crop"} · Original-pixel crop [{result.crop_box?.join(", ")}]</p>
                  )}
                  <p className="mb-3 text-xs text-[#A6B9AE]">{t("scanner.stage_unknown")}</p>
                  {idToken && result.inspection_id && (
                    <a href={`/assistant?inspection=${encodeURIComponent(result.inspection_id)}`} className="mb-3 block text-xs text-[#6AE6A0]">
                      Ask the assistant about this inspection
                    </a>
                  )}
                  {!idToken && selectedFile && (
                    <button type="button" onClick={signInWithGoogle} className="mb-3 rounded-xl border border-[#294B3C] p-2 text-xs">
                      Sign in to save a new scan
                    </button>
                  )}

                  {/* Longitudinal delta if present */}
                  {result.progression && (
                    <div className="p-3 rounded-xl bg-[#10231B] border border-[#294B3C] mb-3 text-xs">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[#F2FAF5]">Longitudinal Delta</span>
                        <span
                          className={`font-mono font-bold ${result.progression.progression_delta > 0
                              ? "text-[#F17878]"
                              : "text-[#6AE6A0]"
                            }`}
                        >
                          {result.progression.progression_delta > 0 ? "+" : ""}
                          {result.progression.progression_delta.toFixed(2)}%
                        </span>
                      </div>
                      <span className="text-[11px] text-[#768D80]">
                        Compared against inspection {result.progression.previous_inspection_id} ({result.progression.previous_date.split("T")[0]})
                      </span>
                    </div>
                  )}

                  {/* Safety Notice */}
                  {result.safety_notice && (
                    <div className="p-3 rounded-xl bg-[#252014] border border-[#F1BA65]/30 text-[#F1BA65] text-xs flex items-start gap-2">
                      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      <span>{result.safety_notice}</span>
                    </div>
                  )}
                </div>

                {/* Model Execution Metadata */}
                <div className="mt-4 pt-3 border-t border-[#294B3C] flex items-center justify-between text-[11px] text-[#768D80] font-mono">
                  <span>Engine: {result.model_version}</span>
                  <span>Latency: {result.processing_time_ms}ms</span>
                </div>
              </div>

              {/* Candidate Filter Transparency Disclosure */}
              {result.rejected_candidates && result.rejected_candidates.length > 0 && (
                <div className="md:col-span-2 bg-[#152C22] rounded-2xl border border-[#294B3C] p-4 text-xs space-y-2.5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2 text-[#F2FAF5] font-semibold">
                      <ShieldCheck className="w-4 h-4 text-[#4FD1C5]" />
                      <span>Rejected model candidates ({result.rejected_candidates.length})</span>
                    </div>
                    <span className="text-[11px] font-mono text-[#A6B9AE]">
                      Leaf Containment Floor: ≥ 58% | Frame Ratio Ceiling: ≤ 35%
                    </span>
                  </div>
                  <div className="max-h-48 overflow-y-auto rounded-xl border border-[#294B3C] bg-[#10231B]">
                    <table className="w-full text-left font-mono text-[11px]">
                      <thead className="bg-[#152C22] text-[#768D80] text-[10px] uppercase border-b border-[#294B3C]">
                        <tr>
                          <th className="py-1.5 px-3">Candidate</th>
                          <th className="py-1.5 px-3">Raw Conf</th>
                          <th className="py-1.5 px-3">Area (px)</th>
                          <th className="py-1.5 px-3">Leaf Containment</th>
                          <th className="py-1.5 px-3">Rejection Justification</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#294B3C]/40">
                        {result.rejected_candidates.map((c, i) => (
                          <tr key={i} className="text-[#A6B9AE] hover:bg-[#152C22]/40">
                            <td className="py-1.5 px-3 text-[#F2FAF5]">{c.candidate_id}</td>
                            <td className="py-1.5 px-3 text-[#F1BA65]">{c.confidence.toFixed(3)}</td>
                            <td className="py-1.5 px-3">{c.raw_pixels.toLocaleString()}</td>
                            <td className="py-1.5 px-3">{c.leaf_containment_pct == null ? "—" : `${c.leaf_containment_pct.toFixed(1)}%`}</td>
                            <td className="py-1.5 px-3 text-[#F17878] font-sans text-[11px]">{c.reasons.join(", ")}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Live Camera Viewfinder Modal */}
      {isCameraOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-in fade-in duration-200">
          <div className="relative w-full max-w-xl bg-[#0D1F17] border border-[#294B3C] rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            {/* Header */}
            <div className="px-5 py-3.5 border-b border-[#294B3C] flex items-center justify-between bg-[#10231B]">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#6AE6A0] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#6AE6A0]"></span>
                </span>
                <span className="text-sm font-bold text-[#F2FAF5]">Camera Viewfinder</span>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#1D4032] text-[#6AE6A0] border border-[#294B3C]">
                  {cameraFacingMode === "environment" ? "Rear Lens" : "Front Lens"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={flipCamera}
                  title="Switch Front / Rear Camera"
                  className="p-1.5 rounded-lg bg-[#152C22] border border-[#294B3C] hover:border-[#6AE6A0]/40 text-[#A6B9AE] hover:text-[#F2FAF5] transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={closeCamera}
                  title="Close Camera"
                  className="p-1.5 rounded-lg bg-[#152C22] border border-[#294B3C] hover:border-red-500/40 text-[#A6B9AE] hover:text-red-400 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Video Viewport */}
            <div className="relative aspect-[4/3] bg-black flex items-center justify-center overflow-hidden">
              {isCameraLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 z-10 bg-black/60">
                  <RefreshCw className="w-8 h-8 text-[#6AE6A0] animate-spin mb-3" />
                  <p className="text-xs text-[#A6B9AE]">Requesting camera permissions...</p>
                </div>
              )}

              {cameraError ? (
                <div className="p-6 text-center max-w-md z-10">
                  <AlertCircle className="w-10 h-10 text-amber-400 mx-auto mb-3" />
                  <h4 className="text-sm font-bold text-[#F2FAF5] mb-2">Camera Unavailable</h4>
                  <p className="text-xs text-[#A6B9AE] mb-4 leading-relaxed">{cameraError}</p>
                    <div className="flex flex-wrap items-center justify-center gap-2.5">
                    <button
                      type="button"
                      onClick={() => startCameraStream(cameraFacingMode)}
                      className="px-3.5 py-2 rounded-xl bg-[#1D4032] text-[#6AE6A0] border border-[#294B3C] hover:bg-[#234F3D] text-xs font-semibold"
                    >
                      Try Again
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        closeCamera();
                        cameraInputRef.current?.click();
                      }}
                      className="px-3.5 py-2 rounded-xl bg-[#6AE6A0] text-[#0A1612] hover:bg-[#58D38C] text-xs font-bold flex items-center gap-1.5"
                    >
                      <Camera className="w-3.5 h-3.5" />
                      <span>Use Device Camera</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        closeCamera();
                        fileInputRef.current?.click();
                      }}
                      className="px-3.5 py-2 rounded-xl bg-[#152C22] text-[#A6B9AE] hover:text-[#F2FAF5] border border-[#294B3C] text-xs font-semibold"
                    >
                      Browse File Instead
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  {/* Leaf Target Alignment Reticle */}
                  <div className="absolute inset-8 pointer-events-none border border-dashed border-[#6AE6A0]/40 rounded-xl flex items-center justify-center">
                    {/* Corner brackets */}
                    <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-[#6AE6A0]"></div>
                    <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-[#6AE6A0]"></div>
                    <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-[#6AE6A0]"></div>
                    <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-[#6AE6A0]"></div>

                    <div className="bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-[#294B3C] text-[11px] font-medium text-[#F2FAF5] flex items-center gap-1.5">
                      <Leaf className="w-3.5 h-3.5 text-[#6AE6A0]" />
                      <span>Position single tomato leaf inside frame</span>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Shutter Bar */}
            <div className="p-4 bg-[#10231B] border-t border-[#294B3C] flex items-center justify-between">
              <button
                type="button"
                onClick={closeCamera}
                className="px-4 py-2 text-xs font-semibold text-[#A6B9AE] hover:text-[#F2FAF5] transition-colors"
              >
                Cancel
              </button>

              {/* Shutter Button */}
              <button
                type="button"
                disabled={Boolean(cameraError) || isCameraLoading}
                onClick={takeSnapshot}
                className="relative flex items-center justify-center w-16 h-16 rounded-full bg-[#152C22] border-4 border-[#6AE6A0] hover:border-white disabled:opacity-40 disabled:cursor-not-allowed group transition-all transform active:scale-95 shadow-lg shadow-[#6AE6A0]/20"
                title="Capture Leaf Photograph"
              >
                <div className="w-11 h-11 rounded-full bg-[#6AE6A0] group-hover:bg-white group-active:scale-90 transition-all flex items-center justify-center">
                  <Camera className="w-5 h-5 text-[#0A1612]" />
                </div>
              </button>

              <button
                type="button"
                onClick={() => {
                  closeCamera();
                  fileInputRef.current?.click();
                }}
                className="px-3 py-1.5 rounded-lg bg-[#152C22] border border-[#294B3C] hover:border-[#6AE6A0]/30 text-xs font-semibold text-[#A6B9AE] hover:text-[#F2FAF5] transition-colors flex items-center gap-1.5"
              >
                <Upload className="w-3.5 h-3.5 text-[#6AE6A0]" />
                <span>Upload</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-xs text-[#A6B9AE] animate-pulse">
          Loading diagnostic lab workspace...
        </div>
      }
    >
      <AnalyzeContent />
    </Suspense>
  );
}
