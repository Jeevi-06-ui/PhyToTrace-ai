"use client";

import Link from "next/link";
import { useLanguage } from "@/context/LanguageContext";
import InteractiveJuryShowcase from "@/components/InteractiveJuryShowcase";
import { 
  Activity, 
  ArrowRight, 
  CheckCircle2, 
  Cpu, 
  FileCheck, 
  Layers, 
  Leaf, 
  Scan, 
  ShieldCheck, 
  Sparkles, 
  Sprout,
  ScanLine,
  ChevronRight,
  Database,
  BarChart3
} from "lucide-react";

export default function Home() {
  const { t } = useLanguage();
  return (
    <div className="flex flex-col space-y-16 py-4">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-b from-[#152C22] via-[#10231B] to-[#0A1814] border border-[#294B3C] p-8 sm:p-14 lg:p-16 shadow-2xl">
        {/* Subtle background glow effect */}
        <div className="absolute top-0 right-1/4 -z-10 w-96 h-96 bg-[#6AE6A0]/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/4 -z-10 w-96 h-96 bg-[#1D4032]/40 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-4xl mx-auto text-center">
          {/* Botanical Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#10231B] border border-[#6AE6A0]/30 text-xs font-semibold text-[#6AE6A0] mb-6 shadow-sm">
            <Sparkles className="w-3.5 h-3.5 text-[#6AE6A0]" />
            <span>Experimental tomato leaf scanner</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-[#F2FAF5] leading-tight">
            {t("hero.headline")}
          </h1>

          <p className="mt-6 text-base sm:text-lg text-[#A6B9AE] leading-relaxed max-w-2xl mx-auto">
            {t("hero.subheading")}
          </p>

          {/* Action CTAs */}
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/analyze"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] px-7 py-3.5 rounded-xl font-bold text-base shadow-lg shadow-[#6AE6A0]/20 transition-all hover:scale-102"
            >
              <ScanLine className="w-5 h-5 text-[#0A1814]" />
              <span>{t("hero.scan_cta")}</span>
              <ArrowRight className="w-4 h-4 text-[#0A1814]" />
            </Link>
            <a
              href="#interactive-showcase"
              className="w-full sm:w-auto flex items-center justify-center gap-2 bg-[#10231B] hover:bg-[#152C22] text-[#6AE6A0] border border-[#6AE6A0]/40 px-6 py-3.5 rounded-xl font-semibold text-base transition-all hover:border-[#6AE6A0]"
            >
              <Sparkles className="w-4 h-4 text-[#6AE6A0]" />
              <span>Live AI Demo Showcase</span>
            </a>
            <Link
              href="/dashboard"
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 bg-[#152C22] hover:bg-[#1D4032] text-[#F2FAF5] border border-[#294B3C] px-7 py-3.5 rounded-xl font-semibold text-base transition-all"
            >
              <Activity className="w-5 h-5 text-[#6AE6A0]" />
              <span>{t("hero.dashboard_cta")}</span>
            </Link>
          </div>

          {/* Hardware & Integrity Badges */}
          <div className="mt-12 pt-8 border-t border-[#294B3C]/80 flex flex-wrap items-center justify-center gap-6 text-xs text-[#A6B9AE]">
            <div className="flex items-center gap-2 font-medium">
              <CheckCircle2 className="w-4 h-4 text-[#6AE6A0]" />
              <span>PyTorch inference</span>
            </div>
            <div className="flex items-center gap-2 font-medium">
              <CheckCircle2 className="w-4 h-4 text-[#6AE6A0]" />
              <span>Zero Synthetic Mask Fabrication</span>
            </div>
            <div className="flex items-center gap-2 font-medium">
              <CheckCircle2 className="w-4 h-4 text-[#6AE6A0]" />
              <span>Reviewed CVAT training masks</span>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Faculty & Jury Demo Hub */}
      <InteractiveJuryShowcase />

      {/* Multi-Stage Vision Pipeline Architecture */}
      <section className="space-y-6">
        <div className="text-center max-w-2xl mx-auto">
          <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
            Multi-Stage Diagnostic Architecture
          </span>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] mt-1">
            How PhytoTrace analyzes a photo
          </h2>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-2">
            The scanner checks photo quality, attempts leaf segmentation, and evaluates model candidates against that boundary.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-5 rounded-2xl border border-[#294B3C] bg-[#152C22] flex flex-col justify-between hover:border-[#6AE6A0]/40 transition-all">
            <div>
              <div className="w-10 h-10 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#6AE6A0] flex items-center justify-center font-bold text-sm mb-4">
                01
              </div>
              <h3 className="font-bold text-[#F2FAF5] text-base mb-1">
                Input Verification
              </h3>
              <p className="text-xs text-[#A6B9AE] leading-relaxed">
                Checks supported formats, resolution, brightness, and sharpness.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#294B3C] text-[11px] font-mono text-[#6AE6A0] font-semibold">
              Preflight Quality Gate
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-[#294B3C] bg-[#152C22] flex flex-col justify-between hover:border-[#6AE6A0]/40 transition-all">
            <div>
              <div className="w-10 h-10 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#6AE6A0] flex items-center justify-center font-bold text-sm mb-4">
                02
              </div>
              <h3 className="font-bold text-[#F2FAF5] text-base mb-1">
                Canopy Segmentation
              </h3>
              <p className="text-xs text-[#A6B9AE] leading-relaxed">
                The canopy segmenter isolates the leaf surface from the background.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#294B3C] text-[11px] font-mono text-[#6AE6A0] font-semibold">
              Canopy boundary estimate
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-[#294B3C] bg-[#152C22] flex flex-col justify-between hover:border-[#6AE6A0]/40 transition-all">
            <div>
              <div className="w-10 h-10 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#F17878] flex items-center justify-center font-bold text-sm mb-4">
                03
              </div>
              <h3 className="font-bold text-[#F2FAF5] text-base mb-1">
                Lesion Localization
              </h3>
              <p className="text-xs text-[#A6B9AE] leading-relaxed">
                The damage segmenter localizes suspicious mine lesions for inspection.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#294B3C] text-[11px] font-mono text-[#F17878] font-semibold">
              Damage Instance Segmenter &bull; High-res input
            </div>
          </div>

          <div className="p-5 rounded-2xl border border-[#294B3C] bg-[#152C22] flex flex-col justify-between hover:border-[#6AE6A0]/40 transition-all">
            <div>
              <div className="w-10 h-10 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#8EF0AF] flex items-center justify-center font-bold text-sm mb-4">
                04
              </div>
              <h3 className="font-bold text-[#F2FAF5] text-base mb-1">
                Severity Ratio & Delta
              </h3>
              <p className="text-xs text-[#A6B9AE] leading-relaxed">
                Calculates exact surface ratio: 100 &times; (Mines &cap; Leaf) / Leaf. Compares against previous plant scans for longitudinal expansion.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#294B3C] text-[11px] font-mono text-[#8EF0AF] font-semibold">
              Severity % &bull; Longitudinal Delta
            </div>
          </div>
        </div>
      </section>

      {/* Platform Navigation Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <Link
          href="/dashboard"
          className="p-6 rounded-2xl bg-[#152C22] border border-[#294B3C] hover:border-[#6AE6A0] hover:bg-[#1D4032] transition-all group shadow-lg"
        >
          <div className="w-12 h-12 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#6AE6A0] flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
            <Activity className="w-6 h-6 text-[#6AE6A0]" />
          </div>
          <h3 className="font-bold text-[#F2FAF5] text-lg group-hover:text-[#6AE6A0] transition-colors">
            Farmer Dashboard
          </h3>
          <p className="text-xs text-[#A6B9AE] mt-2 leading-relaxed">
            Real-time farm health score, active infestation alerts, average damage %, and recent diagnostic records from SQLite repository.
          </p>
          <div className="mt-4 flex items-center gap-1.5 text-xs font-semibold text-[#6AE6A0]">
            <span>Open Dashboard</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </Link>

        <Link
          href="/timeline"
          className="p-6 rounded-2xl bg-[#152C22] border border-[#294B3C] hover:border-[#6AE6A0] hover:bg-[#1D4032] transition-all group shadow-lg"
        >
          <div className="w-12 h-12 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#6AE6A0] flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
            <Sprout className="w-6 h-6 text-[#6AE6A0]" />
          </div>
          <h3 className="font-bold text-[#F2FAF5] text-lg group-hover:text-[#6AE6A0] transition-colors">
            Plant Monitoring Timeline
          </h3>
          <p className="text-xs text-[#A6B9AE] mt-2 leading-relaxed">
            Track leaf-miner expansion or recession over time on specific tagged plants and greenhouse rows with damage delta metrics.
          </p>
          <div className="mt-4 flex items-center gap-1.5 text-xs font-semibold text-[#6AE6A0]">
            <span>View Timelines</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </Link>

        <Link
          href="/research"
          className="p-6 rounded-2xl bg-[#152C22] border border-[#294B3C] hover:border-[#6AE6A0] hover:bg-[#1D4032] transition-all group shadow-lg"
        >
          <div className="w-12 h-12 rounded-xl bg-[#10231B] border border-[#294B3C] text-[#6AE6A0] flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
            <Cpu className="w-6 h-6 text-[#6AE6A0]" />
          </div>
          <h3 className="font-bold text-[#F2FAF5] text-lg group-hover:text-[#6AE6A0] transition-colors">
            Model Status & Audit
          </h3>
          <p className="text-xs text-[#A6B9AE] mt-2 leading-relaxed">
            Inspect checkpoint locations, Stage B classification accuracy, canopy Dice score, lesion detection metrics, and dataset gates.
          </p>
          <div className="mt-4 flex items-center gap-1.5 text-xs font-semibold text-[#6AE6A0]">
            <span>Audit Models</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </div>
        </Link>
      </section>

      {/* Scientific Transparency & Limitations Notice */}
      <section className="p-6 rounded-2xl bg-[#10231B] border border-[#294B3C] text-xs text-[#A6B9AE] space-y-3">
        <div className="flex items-center gap-2 text-sm font-bold text-[#F2FAF5]">
          <ShieldCheck className="w-4 h-4 text-[#6AE6A0]" />
          <span>Scientific Rigor & Machine Learning Transparency Statement</span>
        </div>
        <p className="leading-relaxed">
          PhytoTrace AI models are trained on genuine annotated tomato photographs from pilot greenhouse blocks.
          All performance metrics are measured strictly against isolated validation and test partitions.
          Visible surface leaf-miner injury estimates provide localized leaf-level diagnostics and do not replace whole-field scouting protocols.
        </p>
      </section>
    </div>
  );
}
