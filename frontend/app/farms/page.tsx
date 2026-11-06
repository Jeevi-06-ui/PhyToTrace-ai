"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { 
  AlertTriangle, 
  ArrowRight, 
  BarChart3, 
  Building2, 
  CheckCircle2, 
  Leaf, 
  ShieldCheck, 
  Sprout,
  ScanLine,
  Tractor
} from "lucide-react";
import StatCard from "@/components/StatCard";
import StatusIndicator from "@/components/StatusIndicator";
import { useAuth } from "@/context/AuthContext";
import EmptyState from "@/components/EmptyState";

interface FarmSummary {
  farm_id: string;
  inspections: number;
  plants: number;
  average_damage_pct: number | null;
}

interface FarmOverviewData {
  total_inspections: number;
  total_farms: number;
  total_plants: number;
  average_damage_pct: number | null;
  active_infestations: number;
  severity_distribution: Record<string, number>;
  farms: FarmSummary[];
}

export default function FarmsPage() {
  const { idToken } = useAuth();
  const [data, setData] = useState<FarmOverviewData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/v1/farms", { headers: idToken ? { Authorization: `Bearer ${idToken}` } : undefined })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        console.error("Could not load farms:", e);
        setLoading(false);
      });
  }, [idToken]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#294B3C]">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
            Agricultural Enterprise Structure
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1 flex items-center gap-2.5">
            <Tractor className="w-7 h-7 text-[#6AE6A0]" />
            <span>Farm & Greenhouse Canopy Map</span>
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1">
            Aggregated infestation severity and plant health metrics across all registered commercial greenhouse facilities.
          </p>
        </div>

        <Link
          href="/analyze"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] text-xs font-bold shadow-md shadow-[#6AE6A0]/20 transition-all hover:scale-102"
        >
          <ScanLine className="w-4 h-4 text-[#0A1814]" />
          <span>New Inspection</span>
        </Link>
      </div>

      {loading ? (
        <div className="p-12 text-center text-xs text-[#A6B9AE] animate-pulse">
          Loading farm and greenhouse data...
        </div>
      ) : !data ? (
        <EmptyState
          title="Unable to Load Farm Records"
          description="Could not connect to the database repository. Verify that FastAPI is active."
          actionText="Retry"
          actionHref="/farms"
        />
      ) : (
        <div className="space-y-8">
          {/* Top Aggregate KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              label="Registered Greenhouses"
              value={data.total_farms || data.farms.length}
              subtext="Monitored commercial zones"
              icon={Building2}
              trend={{ value: "Operational", isPositive: true }}
            />

            <StatCard
              label="Canopy Plants"
              value={data.total_plants}
              subtext="Catalogued tomato vines"
              icon={Sprout}
              trend={{ value: "Active", isPositive: true }}
            />

            <StatCard
              label="Active Infestation Hotspots"
              value={data.active_infestations}
              subtext="Greenhouse zones with damage"
              icon={AlertTriangle}
              trend={{
                value: data.active_infestations > 0 ? "Evidence to review" : "No qualifying records",
                isPositive: data.active_infestations === 0,
              }}
              highlight={data.active_infestations > 0}
            />

            <StatCard
              label="Mean Farm Damage %"
              value={
                data.average_damage_pct != null
                  ? `${data.average_damage_pct.toFixed(1)}%`
                  : "—"
              }
              subtext="Average canopy loss across blocks"
              icon={BarChart3}
              trend={{ value: "Canopy Ratio", isNeutral: true }}
            />
          </div>

          {/* Severity Distribution Card */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg">
            <h3 className="text-base font-bold text-[#F2FAF5] mb-4">
              Canopy Severity Tier Distribution
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl bg-[#10231B] border border-[#294B3C]">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-[#A6B9AE]">Zero Damage</span>
                  <span className="font-mono font-bold text-[#6AE6A0]">
                    {data.severity_distribution?.none || 0}
                  </span>
                </div>
                <div className="w-full bg-[#152C22] h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#6AE6A0] rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        (((data.severity_distribution?.none || 0) /
                          (data.total_inspections || 1)) *
                          100)
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#10231B] border border-[#294B3C]">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-[#A6B9AE]">Low (&lt;5%)</span>
                  <span className="font-mono font-bold text-[#F1BA65]">
                    {data.severity_distribution?.low_infestation || 0}
                  </span>
                </div>
                <div className="w-full bg-[#152C22] h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#F1BA65] rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        (((data.severity_distribution?.low_infestation || 0) /
                          (data.total_inspections || 1)) *
                          100)
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#10231B] border border-[#294B3C]">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-[#A6B9AE]">Moderate (5–20%)</span>
                  <span className="font-mono font-bold text-[#F1BA65]">
                    {data.severity_distribution?.moderate_infestation || 0}
                  </span>
                </div>
                <div className="w-full bg-[#152C22] h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#F1BA65] rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        (((data.severity_distribution?.moderate_infestation || 0) /
                          (data.total_inspections || 1)) *
                          100)
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#10231B] border border-[#294B3C]">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-[#A6B9AE]">Severe (&gt;20%)</span>
                  <span className="font-mono font-bold text-[#F17878]">
                    {data.severity_distribution?.severe_infestation || 0}
                  </span>
                </div>
                <div className="w-full bg-[#152C22] h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#F17878] rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        (((data.severity_distribution?.severe_infestation || 0) /
                          (data.total_inspections || 1)) *
                          100)
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Managed Greenhouse Facilities List */}
          <div className="space-y-4">
            <h3 className="text-base font-bold text-[#F2FAF5]">
              Commercial Facilities & Greenhouse Blocks
            </h3>

            {data.farms.length === 0 ? (
              <EmptyState
                title="No Facilities Registered"
                description="No farm facility entries exist in the current database. Record a scan with a farm ID to register a greenhouse zone."
                actionText="Scan New Facility Leaf"
                actionHref="/analyze"
              />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {data.farms.map((f) => (
                  <div
                    key={f.farm_id}
                    className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-5 shadow-lg flex flex-col justify-between hover:border-[#6AE6A0]/40 transition-all"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <div className="w-10 h-10 rounded-xl bg-[#10231B] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0]">
                          <Building2 className="w-5 h-5 text-[#6AE6A0]" />
                        </div>
                        <StatusIndicator
                          status={f.average_damage_pct == null ? "Measurement unavailable" : "Measurements recorded"}
                          size="sm"
                        />
                      </div>

                      <h4 className="font-bold text-base text-[#F2FAF5] mb-1 font-mono">
                        {f.farm_id}
                      </h4>
                      <p className="text-xs text-[#A6B9AE] leading-relaxed mb-4">
                        Commercial tomato greenhouse block under active visual surveillance.
                      </p>

                      <div className="grid grid-cols-3 gap-2 text-xs pt-3 border-t border-[#294B3C]">
                        <div className="bg-[#10231B] p-2 rounded-xl border border-[#294B3C]">
                          <span className="text-[#768D80] block text-[10px]">Scans</span>
                          <span className="font-mono font-bold text-[#F2FAF5] block mt-0.5">
                            {f.inspections}
                          </span>
                        </div>
                        <div className="bg-[#10231B] p-2 rounded-xl border border-[#294B3C]">
                          <span className="text-[#768D80] block text-[10px]">Plants</span>
                          <span className="font-mono font-bold text-[#F2FAF5] block mt-0.5">
                            {f.plants}
                          </span>
                        </div>
                        <div className="bg-[#10231B] p-2 rounded-xl border border-[#294B3C]">
                          <span className="text-[#768D80] block text-[10px]">Avg Dmg</span>
                          <span
                            className={`font-mono font-bold block mt-0.5 ${
                              (f.average_damage_pct ?? 0) > 20
                                ? "text-[#F17878]"
                                : (f.average_damage_pct ?? 0) > 5
                                ? "text-[#F1BA65]"
                                : "text-[#6AE6A0]"
                            }`}
                          >
                            {f.average_damage_pct == null ? "—" : `${f.average_damage_pct.toFixed(1)}%`}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-[#294B3C] flex items-center justify-between text-xs">
                      <Link
                        href={`/history?farm=${encodeURIComponent(f.farm_id)}`}
                        className="text-[#6AE6A0] hover:text-[#8EF0AF] font-semibold flex items-center gap-1"
                      >
                        <span>View Inspections</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
