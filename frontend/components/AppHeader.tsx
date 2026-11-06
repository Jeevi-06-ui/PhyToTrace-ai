"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { 
  Menu, 
  ScanLine, 
  Activity, 
  Cpu, 
  ShieldCheck, 
  Sparkles,
  HelpCircle,
  ExternalLink,
  ChevronRight,
  Globe,
  LogIn,
  LogOut,
  User
} from "lucide-react";
import { useLanguage, Locale } from "@/context/LanguageContext";
import { useAuth } from "@/context/AuthContext";

interface AppHeaderProps {
  onOpenMobile: () => void;
  collapsed: boolean;
}

export default function AppHeader({ onOpenMobile, collapsed }: AppHeaderProps) {
  const pathname = usePathname();
  const { locale, setLocale, t } = useLanguage();
  const { user, isGuest, authError, signInWithGoogle, signOut } = useAuth();
  const [modelStatus, setModelStatus] = useState<any>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setModelStatus(data))
      .catch(() => null);
  }, []);

  const getPageInfo = () => {
    switch (pathname) {
      case "/":
        return { title: t("nav.dashboard", "Overview"), breadcrumb: "Home" };
      case "/dashboard":
        return { title: t("nav.dashboard", "Farm Intelligence Dashboard"), breadcrumb: t("nav.dashboard", "Dashboard") };
      case "/analyze":
        return { title: t("scanner.title", "AI Leaf Scanner Workspace"), breadcrumb: t("nav.scanner", "Diagnostics") };
      case "/farms":
        return { title: t("nav.farms", "Farm & Greenhouse Canopy Map"), breadcrumb: t("nav.farms", "Farm Overview") };
      case "/history":
        return { title: t("nav.history", "Inspection Archive & Provenance"), breadcrumb: t("nav.history", "History") };
      case "/timeline":
        return { title: t("nav.timeline", "Plant Longitudinal Monitoring"), breadcrumb: t("nav.timeline", "Plant Timeline") };
      case "/weather":
        return { title: t("weather.title", "Weather & Crop Intelligence"), breadcrumb: t("nav.weather", "Weather") };
      case "/assistant":
        return { title: t("assistant.title", "PhytoTrace Agri Assistant"), breadcrumb: t("nav.assistant", "Assistant") };
      case "/research":
        return { title: t("nav.research", "Model Architecture & Scientific Audit"), breadcrumb: t("nav.research", "Research") };
      default:
        return { title: "PhytoTrace AI", breadcrumb: "Platform" };
    }
  };

  const pageInfo = getPageInfo();

  return (
    <header className="sticky top-0 z-30 h-16 bg-[#101F19]/90 backdrop-blur-md border-b border-[#294B3C] px-4 sm:px-6 flex items-center justify-between transition-all">
      {/* Left section: Mobile menu + Breadcrumb */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobile}
          className="p-2 rounded-xl border border-[#294B3C] bg-[#152C22] text-[#A6B9AE] hover:text-[#F2FAF5] lg:hidden focus:outline-hidden"
          aria-label="Open mobile menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2 text-xs text-[#768D80]">
          <Link href="/" className="hover:text-[#6AE6A0] transition-colors font-medium hidden sm:inline">
            PhytoTrace
          </Link>
          <ChevronRight className="w-3.5 h-3.5 hidden sm:inline text-[#294B3C]" />
          <span className="text-[#A6B9AE] font-medium hidden sm:inline">
            {pageInfo.breadcrumb}
          </span>
          <ChevronRight className="w-3.5 h-3.5 hidden sm:inline text-[#294B3C]" />
          <span className="text-[#F2FAF5] font-semibold text-xs sm:text-sm truncate max-w-[200px] sm:max-w-none">
            {pageInfo.title}
          </span>
        </div>
      </div>

      {/* Right section: Language Selector + Auth + Model pill + Scan CTA */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Multilingual Selector */}
        <div className="flex items-center bg-[#152C22] border border-[#294B3C] rounded-xl p-0.5">
          <button
            onClick={() => setLocale("en")}
            className={`px-2 py-1 rounded-lg text-xs font-semibold transition-all ${
              locale === "en"
                ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40"
                : "text-[#A6B9AE] hover:text-[#F2FAF5]"
            }`}
            title="English"
          >
            EN
          </button>
          <button
            onClick={() => setLocale("kn")}
            className={`px-2 py-1 rounded-lg text-xs font-semibold transition-all ${
              locale === "kn"
                ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40"
                : "text-[#A6B9AE] hover:text-[#F2FAF5]"
            }`}
            title="ಕನ್ನಡ (Kannada)"
          >
            ಕನ್ನಡ
          </button>
          <button
            onClick={() => setLocale("hi")}
            className={`px-2 py-1 rounded-lg text-xs font-semibold transition-all ${
              locale === "hi"
                ? "bg-[#1D4032] text-[#6AE6A0] border border-[#6AE6A0]/40"
                : "text-[#A6B9AE] hover:text-[#F2FAF5]"
            }`}
            title="हिंदी (Hindi)"
          >
            हिंदी
          </button>
        </div>

        {/* Model Readiness Badge */}
        <div className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#152C22] border border-[#294B3C] text-xs">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#6AE6A0] opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#6AE6A0]"></span>
          </span>
          <span className="text-[#A6B9AE] font-medium">Models:</span>
          <span className="text-[#6AE6A0] font-semibold font-mono text-[11px]">
            Dual-Stage Vision Engine
          </span>
        </div>

        {authError && <span role="alert" className="max-w-48 text-xs text-[#F1BA65]">{authError}</span>}
        {/* Auth Button */}
        {isGuest ? (
          <button
            onClick={signInWithGoogle}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#152C22] hover:bg-[#1D4032] border border-[#294B3C] text-xs font-semibold text-[#A6B9AE] hover:text-[#F2FAF5] transition-colors"
          >
            <LogIn className="w-3.5 h-3.5 text-[#6AE6A0]" />
            <span className="hidden sm:inline">{t("nav.sign_in", "Sign In")}</span>
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#6AE6A0] font-semibold hidden md:inline truncate max-w-[120px]">
              {user?.displayName}
            </span>
            <button
              onClick={signOut}
              className="p-1.5 rounded-xl bg-[#152C22] hover:bg-[#271515] border border-[#294B3C] hover:border-[#F17878]/40 text-[#A6B9AE] hover:text-[#F17878] transition-colors"
              title="Sign Out"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Quick Scan Action */}
        <Link
          href="/analyze"
          className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-semibold text-xs shadow-md shadow-[#6AE6A0]/20 transition-all hover:scale-102"
        >
          <ScanLine className="w-4 h-4 text-[#0A1814]" />
          <span className="hidden sm:inline">{t("nav.start_scan", "Start Leaf Scan")}</span>
          <span className="sm:hidden">{t("nav.new_scan", "Scan")}</span>
        </Link>
      </div>
    </header>
  );
}
