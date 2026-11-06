"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { 
  Activity, 
  AlertTriangle, 
  ArrowRight, 
  CheckCircle2, 
  Clock, 
  Filter, 
  Leaf, 
  ScanLine, 
  ShieldAlert, 
  Sprout,
  Cpu,
  TrendingUp,
  BarChart2,
  Calendar
} from "lucide-react";
import StatCard from "@/components/StatCard";
import StatusIndicator from "@/components/StatusIndicator";
import TrendChart from "@/components/TrendChart";
import { useAuth } from "@/context/AuthContext";
import EmptyState from "@/components/EmptyState";

interface FarmOverview {
  total_inspections: number;
  total_farms: number;
  total_plants: number;
  average_damage_pct: number | null;
  active_infestations: number;
  severity_distribution: Record<string, number>;
}

interface InspectionItem {
  id: string;
  farm_id: string;
  plant_id: string;
  leaf_id: string;
  created_at: string;
  quality_status: string;
  damage_detected: boolean;
  damage_percentage: number | null;
  severity_tier: string;
  decision: string;
  model_version?: string;
  image_urls?: {
    original: string;
    annotated: string;
  };
}

export default function DashboardPage() {
  const { idToken } = useAuth();
  const [overview, setOverview] = useState<FarmOverview | null>(null);
  const [inspections, setInspections] = useState<InspectionItem[]>([]);
  const [modelStatus, setModelStatus] = useState<any>(null);
  const [filterSeverity, setFilterSeverity] = useState<string>("all");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch farm overview
    fetch("/api/v1/farms", { headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setOverview(data))
      .catch((e) => console.error("Error loading overview:", e));

    // Fetch recent inspections
    fetch("/api/v1/inspections?limit=25", { headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined })
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => {
        setInspections(data.items || []);
        setLoading(false);
      })
      .catch((e) => {
        console.error("Error loading inspections:", e);
        setLoading(false);
      });

    // Fetch model status
    fetch("/api/v1/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setModelStatus(data))
      .catch(() => null);
  }, [idToken]);

  const filteredInspections = inspections.filter((insp) => {
    if (filterSeverity === "all") return true;
    if (filterSeverity === "damaged") return insp.damage_detected;
    if (filterSeverity === "healthy") return insp.damage_detected === false;
    return insp.severity_tier === filterSeverity;
  });

  // Prepare data for TrendChart
  const chartData = inspections
    .filter((insp) => insp.damage_percentage !== null && insp.damage_percentage !== undefined)
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((insp) => ({
      date: insp.created_at,
      damagePercentage: Number(insp.damage_percentage || 0),
      label: insp.plant_id.split("_").pop() || "P",
    }));

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-[#294B3C]">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
            Agricultural Intelligence Dashboard
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1">
            Farm Health & Severity Overview
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1">
            Real-time canopy inspection telemetry, infestation monitoring, and longitudinal progression.
          </p>
        </div>

        <Link
          href="/analyze"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-bold text-sm shadow-md shadow-[#6AE6A0]/20 transition-all hover:scale-102 shrink-0"
        >
          <ScanLine className="w-4 h-4 text-[#0A1814]" />
          <span>New Leaf Scan</span>
        </Link>
      </div>

      {/* Top KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Inspections"
          value={overview ? overview.total_inspections : inspections.length}
          subtext="Recorded canopy assessments"
          icon={Activity}
          trend={{ value: "Verified DB", isNeutral: true }}
        />

        <StatCard
          label="Monitored Plants"
          value={overview?.total_plants ?? "—"}
          subtext="Unique tomato specimens"
          icon={Sprout}
          trend={{ value: "Active", isPositive: true }}
        />

        <StatCard
          label="Scans With Model Evidence"
          value={overview?.active_infestations ?? "—"}
          subtext="Experimental model evidence"
          icon={ShieldAlert}
          trend={{
            value: (overview?.active_infestations || 0) > 0 ? "Evidence to review" : "No qualifying records",
            isPositive: (overview?.active_infestations || 0) === 0,
          }}
          highlight={(overview?.active_infestations || 0) > 0}
        />

        <StatCard
          label="Mean Visible Damage"
          value={
            overview && overview.average_damage_pct != null
              ? `${overview.average_damage_pct.toFixed(1)}%`
              : "—"
          }
          subtext="Average canopy surface mined"
          icon={BarChart2}
          trend={{
            value: "Canopy Ratio",
            isNeutral: true,
          }}
        />
      </div>

      {/* Charts & Model Availability Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trend Chart (2 columns) */}
        <div className="lg:col-span-2 bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE]">
                Longitudinal Infestation Trend
              </span>
              <h3 className="text-base font-bold text-[#F2FAF5]">
                Canopy Damage Severity Progression (%)
              </h3>
            </div>
            <span className="text-[11px] font-mono text-[#6AE6A0] bg-[#10231B] px-2.5 py-1 rounded-full border border-[#294B3C]">
              {chartData.length} Data Points
            </span>
          </div>

          <TrendChart data={chartData} height={230} />
        </div>

        {/* Diagnostic Engine Status (1 column) */}
        <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-[#A6B9AE]">
                Diagnostic Engine
              </span>
              <span className="w-2.5 h-2.5 rounded-full bg-[#6AE6A0] animate-pulse" />
            </div>

            <h3 className="text-base font-bold text-[#F2FAF5] mb-2">
              Vision Models Status
            </h3>

            <p className="text-xs text-[#A6B9AE] leading-relaxed mb-4">
              PhytoTrace executes two deep learning vision models on genuine tomato canopy samples.
            </p>

            <div className="space-y-3">
              {[["Canopy boundary segmenter", "unet_leaf_segmenter"], ["Damage instance detector", "yolo_mine_detector"]].map(([name, key]) => (
                <div key={key} className="flex items-center justify-between rounded-xl border border-[#294B3C] bg-[#10231B] p-3 text-xs">
                  <span>{name}</span>
                  <StatusIndicator status={modelStatus?.models_active?.[key] ? "Checkpoint loaded" : "Unavailable"} size="sm" />
                </div>
              ))}
            </div>
          </div>

          <Link
            href="/research"
            className="mt-4 pt-3 border-t border-[#294B3C] text-xs font-semibold text-[#6AE6A0] hover:text-[#8EF0AF] flex items-center justify-between"
          >
            <span>View Full Model Audit & Metrics</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Recent Inspections Table */}
      <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] overflow-hidden shadow-lg">
        {/* Table Header with Filters */}
        <div className="p-4 sm:p-5 border-b border-[#294B3C] flex flex-wrap items-center justify-between gap-3 bg-[#10231B]">
          <div>
            <h3 className="text-base font-bold text-[#F2FAF5]">
              Recent Leaf Inspections
            </h3>
            <span className="text-xs text-[#A6B9AE]">
              Showing {filteredInspections.length} recorded diagnostic evaluations
            </span>
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-1.5 bg-[#152C22] p-1 rounded-xl border border-[#294B3C]">
            <button
              onClick={() => setFilterSeverity("all")}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                filterSeverity === "all"
                  ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40"
                  : "text-[#A6B9AE] hover:text-[#F2FAF5]"
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilterSeverity("damaged")}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                filterSeverity === "damaged"
                  ? "bg-[#271515] text-[#F17878] border border-[#F17878]/40"
                  : "text-[#A6B9AE] hover:text-[#F2FAF5]"
              }`}
            >
              Damaged Only
            </button>
            <button
              onClick={() => setFilterSeverity("healthy")}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                filterSeverity === "healthy"
                  ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40"
                  : "text-[#A6B9AE] hover:text-[#F2FAF5]"
              }`}
            >
              Healthy Only
            </button>
          </div>
        </div>

        {/* Inspections Table Content */}
        {loading ? (
          <div className="p-8 text-center text-xs text-[#A6B9AE] animate-pulse">
            Loading inspection records...
          </div>
        ) : filteredInspections.length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No Inspection Records Found"
              description="No canopy scans match the selected filter. Upload a leaf photo to generate an explainable diagnostic assessment."
              actionText="Run First Leaf Scan"
              actionHref="/analyze"
            />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#10231B] text-[#A6B9AE] font-semibold border-b border-[#294B3C]">
                <tr>
                  <th className="py-3 px-4">Inspection ID</th>
                  <th className="py-3 px-4">Plant Identifier</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4">Damage %</th>
                  <th className="py-3 px-4">Severity Tier</th>
                  <th className="py-3 px-4">Decision</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#294B3C]">
                {filteredInspections.map((insp) => (
                  <tr
                    key={insp.id}
                    className="hover:bg-[#1D4032]/60 transition-colors"
                  >
                    <td className="py-3 px-4 font-mono font-bold text-[#F2FAF5] max-w-[140px] truncate">
                      {insp.id}
                    </td>
                    <td className="py-3 px-4 font-mono text-[#A6B9AE]">
                      {insp.plant_id}
                    </td>
                    <td className="py-3 px-4 text-[#768D80]">
                      {insp.created_at.split("T")[0]}{" "}
                      <span className="text-[10px]">
                        {insp.created_at.split("T")[1]?.slice(0, 5)}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-bold font-mono">
                      {insp.damage_percentage !== null && insp.damage_percentage !== undefined ? (
                        <span
                          className={
                            insp.damage_percentage > 20
                              ? "text-[#F17878]"
                              : insp.damage_percentage > 5
                              ? "text-[#F1BA65]"
                              : "text-[#6AE6A0]"
                          }
                        >
                          {insp.damage_percentage.toFixed(2)}%
                        </span>
                      ) : (
                        <span className="text-[#768D80]">N/A</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <StatusIndicator status={insp.severity_tier} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-[#A6B9AE] capitalize">
                      {insp.decision.replace(/_/g, " ")}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <Link
                        href={`/analyze?id=${insp.id}`}
                        className="inline-flex items-center gap-1 font-semibold text-[#6AE6A0] hover:text-[#8EF0AF]"
                      >
                        <span>View</span>
                        <ArrowRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
