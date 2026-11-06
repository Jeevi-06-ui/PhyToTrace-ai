"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { 
  ArrowRight, 
  Calendar, 
  CheckCircle2, 
  Clock, 
  Layers, 
  Leaf, 
  Sprout, 
  TrendingDown, 
  TrendingUp,
  Info,
  AlertTriangle
} from "lucide-react";
import StatusIndicator from "@/components/StatusIndicator";
import TrendChart from "@/components/TrendChart";
import { useAuth } from "@/context/AuthContext";
import EmptyState from "@/components/EmptyState";

interface TimelineRecord {
  id: string;
  farm_id: string;
  plant_id: string;
  leaf_id: string;
  created_at: string;
  damage_percentage: number | null;
  severity_tier: string;
  decision: string;
  progression_delta?: number;
  progression_status?: string;
  image_urls?: {
    original: string;
    annotated: string;
  };
}

export default function TimelinePage() {
  const { idToken } = useAuth();
  const [plantId, setPlantId] = useState("");
  const [timelineData, setTimelineData] = useState<TimelineRecord[]>([]);
  const [availablePlants, setAvailablePlants] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch unique plants from real inspections list
    fetch("/api/v1/inspections?limit=100", { headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined })
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => {
        const plants = Array.from(
          new Set((data.items || []).map((i: any) => i.plant_id))
        ).filter(Boolean) as string[];
        setAvailablePlants(plants);
        if (plants.length > 0) {
          setPlantId(plants[0]);
        }
      })
      .catch((e) => console.error("Could not fetch plants:", e));
  }, [idToken]);

  useEffect(() => {
    if (!plantId) return;
    setLoading(true);
    fetch(`/api/v1/inspections?plant_id=${encodeURIComponent(plantId)}&limit=100`, { headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined })
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => {
        setTimelineData((data.items || []).slice().reverse());
        setLoading(false);
      })
      .catch((e) => {
        console.error("Failed to load timeline:", e);
        setLoading(false);
      });
  }, [plantId, idToken]);

  // Chart points for this plant
  const chartPoints = timelineData
    .filter((d) => d.damage_percentage !== null && d.damage_percentage !== undefined)
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((d) => ({
      date: d.created_at,
      damagePercentage: Number(d.damage_percentage || 0),
      label: d.leaf_id,
    }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#294B3C]">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
            Longitudinal Canopy Tracking
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1 flex items-center gap-2.5">
            <Sprout className="w-7 h-7 text-[#6AE6A0]" />
            <span>Plant-Level Monitoring Timeline</span>
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1">
            Track individual tagged tomato plants over time, observing mine lesion expansion or containment across repeated canopy scans.
          </p>
        </div>

        {/* Plant Selector */}
        {availablePlants.length > 0 && (
          <div className="flex items-center gap-2 bg-[#152C22] p-2 rounded-xl border border-[#294B3C]">
            <span className="text-xs text-[#A6B9AE] font-semibold pl-2">Select Plant:</span>
            <select
              value={plantId}
              onChange={(e) => setPlantId(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#10231B] border border-[#294B3C] text-xs font-mono font-bold text-[#F2FAF5] focus:outline-hidden focus:border-[#6AE6A0]"
            >
              {availablePlants.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Agronomic Comparability Notice */}
      <div className="p-4 rounded-2xl bg-[#10231B] border border-[#294B3C] text-xs text-[#A6B9AE] flex items-start gap-3">
        <Info className="w-4 h-4 text-[#6AE6A0] shrink-0 mt-0.5" />
        <span className="leading-relaxed">
          <strong>Longitudinal Validity Principle:</strong> Inspections are linked exclusively by confirmed plant identifier.
          Delta percentages reflect surface damage changes on photographed leaflets. Do not assume biological comparability if canopy angles or lighting differ markedly.
        </span>
      </div>

      {/* Progression Chart Card */}
      <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE]">
              Plant Progression: {plantId || "Unselected"}
            </span>
            <h3 className="text-base font-bold text-[#F2FAF5]">
              Canopy Damage Trajectory (%)
            </h3>
          </div>
          <span className="text-xs font-mono text-[#6AE6A0] bg-[#10231B] px-3 py-1 rounded-full border border-[#294B3C]">
            {chartPoints.length} Sequential Scans
          </span>
        </div>

        <TrendChart data={chartPoints} height={200} />
      </div>

      {/* Chronological Timeline Chain */}
      <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg">
        <h3 className="text-base font-bold text-[#F2FAF5] mb-6">
          Chronological Inspection Events
        </h3>

        {loading ? (
          <div className="p-8 text-center text-xs text-[#A6B9AE] animate-pulse">
            Loading plant timeline history...
          </div>
        ) : timelineData.length === 0 ? (
          <EmptyState
            title="No Timeline Records for Plant"
            description="No sequential scans have been recorded yet for this plant ID. Run multiple scans over time with the same plant identifier to build a progression history."
            actionText="Scan This Plant Now"
            actionHref="/analyze"
          />
        ) : (
          <div className="relative border-l-2 border-[#294B3C] ml-4 sm:ml-6 space-y-8 pl-6 sm:pl-8">
            {timelineData.map((event, idx) => (
              <div key={event.id} className="relative group">
                {/* Timeline node icon */}
                <div className="absolute -left-[35px] sm:-left-[43px] top-1.5 w-6 h-6 rounded-full bg-[#10231B] border-2 border-[#6AE6A0] flex items-center justify-center text-[10px] font-bold text-[#6AE6A0]">
                  {timelineData.length - idx}
                </div>

                {/* Card Container */}
                <div className="bg-[#10231B] rounded-2xl border border-[#294B3C] p-5 hover:border-[#6AE6A0]/40 transition-all shadow-md">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3 pb-3 border-b border-[#294B3C]">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-sm text-[#F2FAF5]">
                          {event.id}
                        </span>
                        <StatusIndicator status={event.severity_tier} size="sm" />
                      </div>
                      <span className="text-[11px] text-[#768D80] flex items-center gap-1.5 mt-1">
                        <Clock className="w-3 h-3" />
                        {event.created_at.replace("T", " ").slice(0, 19)}
                        <span className="text-[#294B3C]">•</span>
                        <span>Leaf Tier: {event.leaf_id}</span>
                      </span>
                    </div>

                    {/* Damage & Progression Delta */}
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <span className="text-[10px] text-[#768D80] block">Damage Area</span>
                        <span
                          className={`font-mono font-black text-lg ${
                            (event.damage_percentage || 0) > 20
                              ? "text-[#F17878]"
                              : (event.damage_percentage || 0) > 5
                              ? "text-[#F1BA65]"
                              : "text-[#6AE6A0]"
                          }`}
                        >
                          {event.damage_percentage !== null && event.damage_percentage !== undefined
                            ? `${event.damage_percentage.toFixed(2)}%`
                            : "N/A"}
                        </span>
                      </div>

                      {event.progression_delta !== undefined && (
                        <div className="p-2 rounded-xl bg-[#152C22] border border-[#294B3C] text-right">
                          <span className="text-[9px] text-[#768D80] block">Delta</span>
                          <span
                            className={`font-mono font-bold text-xs flex items-center gap-0.5 ${
                              event.progression_delta > 0
                                ? "text-[#F17878]"
                                : "text-[#6AE6A0]"
                            }`}
                          >
                            {event.progression_delta > 0 ? (
                              <TrendingUp className="w-3 h-3" />
                            ) : (
                              <TrendingDown className="w-3 h-3" />
                            )}
                            {event.progression_delta > 0 ? "+" : ""}
                            {event.progression_delta.toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
                    <p className="text-[#A6B9AE]">
                      Decision: <strong className="text-[#F2FAF5] capitalize">{event.decision.replace(/_/g, " ")}</strong>
                    </p>

                    <Link
                      href={`/analyze?id=${event.id}`}
                      className="inline-flex items-center gap-1.5 font-semibold text-[#6AE6A0] hover:text-[#8EF0AF]"
                    >
                      <span>Open in Diagnostic Lab</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
