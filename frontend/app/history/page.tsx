"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { 
  ArrowRight, 
  Calendar, 
  Clock, 
  Download, 
  Filter, 
  Leaf, 
  Search, 
  Sprout,
  Image as ImageIcon,
  Cpu,
  RefreshCw
} from "lucide-react";
import StatusIndicator from "@/components/StatusIndicator";
import { useAuth } from "@/context/AuthContext";
import EmptyState from "@/components/EmptyState";

interface InspectionRecord {
  id: string;
  farm_id: string;
  plant_id: string;
  leaf_id: string;
  created_at: string;
  damage_detected: boolean;
  damage_percentage: number | null;
  severity_tier: string;
  decision: string;
  quality_status: string;
  model_version?: string;
  image_urls?: {
    original: string;
    annotated: string;
  };
}

export default function HistoryPage() {
  const { idToken } = useAuth();
  const [inspections, setInspections] = useState<InspectionRecord[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);

  // Filters
  const [farmFilter, setFarmFilter] = useState(() => typeof window === "undefined" ? "" : new URLSearchParams(window.location.search).get("farm") ?? "");
  const [plantFilter, setPlantFilter] = useState("");
  const [severityFilter, setSeverityFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const loadData = () => {
    setLoading(true);
    let url = "/api/v1/inspections?limit=100";
    if (farmFilter) url += `&farm_id=${encodeURIComponent(farmFilter)}`;
    if (plantFilter) url += `&plant_id=${encodeURIComponent(plantFilter)}`;
    if (severityFilter) url += `&severity_tier=${encodeURIComponent(severityFilter)}`;

    fetch(url, { headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined })
      .then((r) => (r.ok ? r.json() : { items: [], total: 0 }))
      .then((data) => {
        setInspections(data.items || []);
        setTotalCount(data.total || 0);
        setLoading(false);
      })
      .catch((e) => {
        console.error("Failed to load history:", e);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData();
  }, [farmFilter, plantFilter, severityFilter, idToken]);

  const filteredItems = inspections.filter((item) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.id.toLowerCase().includes(q) ||
      item.plant_id.toLowerCase().includes(q) ||
      item.farm_id.toLowerCase().includes(q) ||
      item.decision.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#294B3C]">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
            Audit Trail & Evidence Archive
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1">
            Inspection History & Provenance
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1">
            Searchable repository of recorded diagnostic evaluations with original photographs and segmentation masks.
          </p>
        </div>

        <button
          onClick={loadData}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#152C22] hover:bg-[#1D4032] border border-[#294B3C] text-xs font-semibold text-[#F2FAF5] transition-all shadow-md"
        >
          <RefreshCw className="w-3.5 h-3.5 text-[#6AE6A0]" />
          <span>Refresh Records</span>
        </button>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-4 shadow-lg space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Search Box */}
          <div className="relative">
            <Search className="w-4 h-4 text-[#768D80] absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search by ID, plant, or decision..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-xs text-[#F2FAF5] placeholder-[#768D80] focus:outline-hidden focus:border-[#6AE6A0]"
            />
          </div>

          {/* Severity Dropdown */}
          <div>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-xs text-[#F2FAF5] focus:outline-hidden focus:border-[#6AE6A0]"
            >
              <option value="">All Severity Tiers</option>
              <option value="none">Zero Damage / Healthy</option>
              <option value="low_infestation">Low Infestation (&lt;5%)</option>
              <option value="moderate_infestation">Moderate Infestation (5–20%)</option>
              <option value="severe_infestation">Severe Infestation (&gt;20%)</option>
            </select>
          </div>

          {/* Farm ID input */}
          <div>
            <input
              type="text"
              placeholder="Filter by Farm ID..."
              value={farmFilter}
              onChange={(e) => setFarmFilter(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-xs text-[#F2FAF5] placeholder-[#768D80] focus:outline-hidden focus:border-[#6AE6A0]"
            />
          </div>

          {/* Plant ID input */}
          <div>
            <input
              type="text"
              placeholder="Filter by Plant ID..."
              value={plantFilter}
              onChange={(e) => setPlantFilter(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-[#10231B] border border-[#294B3C] text-xs text-[#F2FAF5] placeholder-[#768D80] focus:outline-hidden focus:border-[#6AE6A0]"
            />
          </div>
        </div>

        <div className="flex items-center justify-between text-[11px] text-[#A6B9AE] pt-2 border-t border-[#294B3C]">
          <span>
            Showing <strong className="text-[#F2FAF5]">{filteredItems.length}</strong> of {totalCount} records
          </span>
          {(farmFilter || plantFilter || severityFilter || searchQuery) && (
            <button
              onClick={() => {
                setFarmFilter("");
                setPlantFilter("");
                setSeverityFilter("");
                setSearchQuery("");
              }}
              className="text-[#6AE6A0] hover:underline font-semibold"
            >
              Reset Filters
            </button>
          )}
        </div>
      </div>

      {/* Inspection Cards / List */}
      {loading ? (
        <div className="p-12 text-center text-xs text-[#A6B9AE] animate-pulse">
          Loading inspection archive records...
        </div>
      ) : filteredItems.length === 0 ? (
        <EmptyState
          title="No Inspection Records Found"
          description="No records match your search criteria. Try clearing filters or run a new inspection scan."
          actionText="Run New Scan"
          actionHref="/analyze"
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredItems.map((insp) => (
            <div
              key={insp.id}
              className="bg-[#152C22] rounded-2xl border border-[#294B3C] overflow-hidden hover:border-[#6AE6A0]/40 transition-all shadow-lg flex flex-col justify-between"
            >
              <div>
                {/* Thumbnail Preview Banner */}
                <div className="relative h-44 bg-[#0A1814] flex items-center justify-center overflow-hidden border-b border-[#294B3C]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/api/inspections/${insp.id}/images/annotated`}
                    alt={`Inspection ${insp.id}`}
                    onError={(e) => {
                      // Fallback to original if annotated not yet generated
                      (e.target as HTMLImageElement).src = `/api/inspections/${insp.id}/images/original`;
                    }}
                    className="w-full h-full object-cover opacity-90 hover:opacity-100 transition-opacity"
                  />
                  <div className="absolute top-3 right-3">
                    <StatusIndicator status={insp.severity_tier} size="sm" />
                  </div>
                </div>

                {/* Details Section */}
                <div className="p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono font-bold text-[#F2FAF5] truncate max-w-[180px]">
                      {insp.id}
                    </span>
                    <span className="text-[11px] text-[#768D80] flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {insp.created_at.split("T")[0]}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-[#10231B] p-2 rounded-xl border border-[#294B3C]">
                      <span className="text-[#768D80] block text-[10px]">Plant ID</span>
                      <span className="font-mono font-semibold text-[#F2FAF5] truncate block">
                        {insp.plant_id}
                      </span>
                    </div>

                    <div className="bg-[#10231B] p-2 rounded-xl border border-[#294B3C]">
                      <span className="text-[#768D80] block text-[10px]">Visible Damage %</span>
                      <span
                        className={`font-mono font-bold text-sm block ${
                          (insp.damage_percentage || 0) > 20
                            ? "text-[#F17878]"
                            : (insp.damage_percentage || 0) > 5
                            ? "text-[#F1BA65]"
                            : "text-[#6AE6A0]"
                        }`}
                      >
                        {insp.damage_percentage !== null && insp.damage_percentage !== undefined
                          ? `${insp.damage_percentage.toFixed(2)}%`
                          : "N/A"}
                      </span>
                    </div>
                  </div>

                  <div className="text-xs text-[#A6B9AE]">
                    <span className="text-[#768D80]">Decision: </span>
                    <span className="capitalize font-medium text-[#F2FAF5]">
                      {insp.decision.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Footer */}
              <div className="p-4 pt-2 border-t border-[#294B3C] flex items-center justify-between text-xs">
                <Link
                  href={`/analyze?id=${insp.id}`}
                  className="font-semibold text-[#6AE6A0] hover:text-[#8EF0AF] flex items-center gap-1 transition-colors"
                >
                  <span>Open Diagnostic Lab</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
