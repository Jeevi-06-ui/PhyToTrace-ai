"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { 
  LayoutDashboard, 
  ScanLine, 
  Tractor, 
  History, 
  Sprout, 
  Cpu, 
  Settings, 
  ChevronLeft, 
  ChevronRight,
  ShieldCheck,
  Leaf,
  Activity,
  Sparkles,
  CloudSun,
  Bot
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";
import { useAuth } from "@/context/AuthContext";

interface AppSidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export default function AppSidebar({
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}: AppSidebarProps) {
  const pathname = usePathname();
  const { t } = useLanguage();
  const { user, isGuest } = useAuth();
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [checkingHealth, setCheckingHealth] = useState<boolean>(false);

  const checkHealth = async () => {
    setCheckingHealth(true);
    try {
      let res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) {
        res = await fetch("/api/v1/health", { cache: "no-store" });
      }
      if (res.ok) {
        setBackendOnline(true);
        setCheckingHealth(false);
        return;
      }
    } catch {}

    // Quick retry before declaring offline to prevent flashing offline on minor network ticks
    try {
      await new Promise((r) => setTimeout(r, 600));
      let res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) {
        res = await fetch("/api/v1/health", { cache: "no-store" });
      }
      setBackendOnline(res.ok);
    } catch {
      setBackendOnline(false);
    } finally {
      setCheckingHealth(false);
    }
  };

  useEffect(() => {
    void checkHealth();
    // Poll every 8 seconds to automatically reflect daemon recovery
    const interval = setInterval(() => {
      void checkHealth();
    }, 8000);

    const onFocus = () => {
      void checkHealth();
    };
    window.addEventListener("focus", onFocus);

    return () => {
      clearInterval(interval);
      window.removeEventListener("focus", onFocus);
    };
  }, [pathname]);

  const navItems = [
    {
      name: t("nav.dashboard", "Dashboard"),
      href: "/dashboard",
      icon: LayoutDashboard,
      badge: null,
    },
    {
      name: t("nav.scanner", "AI Leaf Scanner"),
      href: "/analyze",
      icon: ScanLine,
      badge: "MVP",
    },
    {
      name: t("nav.farms", "Farm Overview"),
      href: "/farms",
      icon: Tractor,
      badge: null,
    },
    {
      name: t("nav.history", "Inspection History"),
      href: "/history",
      icon: History,
      badge: null,
    },
    {
      name: t("nav.timeline", "Plant Monitoring"),
      href: "/timeline",
      icon: Sprout,
      badge: null,
    },
    {
      name: t("nav.weather", "Weather Intelligence"),
      href: "/weather",
      icon: CloudSun,
      badge: "New",
    },
    {
      name: t("nav.assistant", "Agri Assistant"),
      href: "/assistant",
      icon: Bot,
      badge: "AI",
    },
    {
      name: t("nav.research", "Research & Models"),
      href: "/research",
      icon: Cpu,
      badge: "Audit",
    },
    {
      name: t("settings.title", "Profile and settings"),
      href: "/settings",
      icon: Settings,
      badge: null,
    },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs lg:hidden"
          onClick={onCloseMobile}
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex flex-col bg-[#10231B] border-r border-[#294B3C] transition-all duration-300 ease-in-out ${
          collapsed ? "w-20" : "w-64"
        } ${
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-[#294B3C]">
          <Link
            href="/"
            onClick={onCloseMobile}
            className="flex items-center gap-3 group overflow-hidden"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#1D4032] to-[#152C22] border border-[#294B3C] flex items-center justify-center shrink-0 shadow-inner group-hover:border-[#6AE6A0] transition-colors">
              <Leaf className="w-5 h-5 text-[#6AE6A0]" />
            </div>
            {!collapsed && (
              <div className="flex flex-col min-w-0">
                <span className="font-bold text-base tracking-tight text-[#F2FAF5] flex items-center gap-1.5 truncate">
                  PhytoTrace
                  <span className="text-[10px] font-semibold text-[#6AE6A0] px-1.5 py-0.5 rounded-full bg-[#152C22] border border-[#294B3C]">
                    AI
                  </span>
                </span>
                <span className="text-[11px] text-[#768D80] truncate font-medium">
                  {t("brand.tagline", "Tomato Canopy Defense")}
                </span>
              </div>
            )}
          </Link>

          {/* Desktop collapse toggle */}
          <button
            onClick={onToggleCollapse}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="hidden lg:flex w-7 h-7 rounded-lg border border-[#294B3C] bg-[#152C22] hover:bg-[#1D4032] text-[#A6B9AE] hover:text-[#F2FAF5] items-center justify-center transition-colors"
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Primary Navigation */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          <div className={`px-2 mb-2 ${collapsed ? "text-center" : ""}`}>
            <span className="text-[10px] font-bold uppercase tracking-wider text-[#768D80]">
              {collapsed ? "•••" : "Platform Modules"}
            </span>
          </div>

          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onCloseMobile}
                title={collapsed ? item.name : undefined}
                className={`flex items-center gap-3 px-3 py-2 rounded-xl text-sm font-medium transition-all group ${
                  isActive
                    ? "bg-[#1D4032] text-[#6AE6A0] border border-[#294B3C] shadow-sm shadow-[#6AE6A0]/10"
                    : "text-[#A6B9AE] hover:text-[#F2FAF5] hover:bg-[#152C22] border border-transparent"
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 transition-transform group-hover:scale-105 ${
                    isActive ? "text-[#6AE6A0]" : "text-[#768D80] group-hover:text-[#A6B9AE]"
                  }`}
                />
                {!collapsed && (
                  <span className="truncate flex-1 text-xs">{item.name}</span>
                )}
                {!collapsed && item.badge && (
                  <span
                    className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${
                      isActive
                        ? "bg-[#152C22] text-[#6AE6A0] border border-[#294B3C]"
                        : "bg-[#152C22] text-[#768D80] border border-[#294B3C]"
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* System & Hardware Status Card */}
        <div className="p-3 border-t border-[#294B3C] space-y-2">
          {!collapsed ? (
            <div
              onClick={checkHealth}
              title="Click to refresh Engine Telemetry status"
              className="p-2.5 rounded-xl bg-[#152C22] border border-[#294B3C] text-xs cursor-pointer hover:border-[#6AE6A0]/40 transition group"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-semibold text-[#F2FAF5] flex items-center gap-1.5 text-[11px] group-hover:text-[#6AE6A0] transition-colors">
                  <Activity className={`w-3.5 h-3.5 text-[#6AE6A0] ${checkingHealth ? "animate-spin" : ""}`} />
                  Engine Telemetry
                </span>
                <span className="flex items-center gap-1.5">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      backendOnline === true
                        ? "bg-[#6AE6A0] animate-pulse"
                        : backendOnline === false
                        ? "bg-[#F17878]"
                        : "bg-[#F1BA65]"
                    }`}
                  />
                  <span className={`text-[10px] font-mono ${backendOnline === true ? "text-[#6AE6A0]" : "text-[#A6B9AE]"}`}>
                    {backendOnline === true ? "Online" : backendOnline === false ? "Offline" : "Checking..."}
                  </span>
                </span>
              </div>
              <div className="space-y-1 text-[10px] text-[#768D80]">
                <div className="flex justify-between">
                  <span>Leaf Segmenter:</span>
                  <span className="text-[#A6B9AE] font-mono">Neural Canopy</span>
                </div>
                <div className="flex justify-between">
                  <span>Mine Detector:</span>
                  <span className="text-[#A6B9AE] font-mono">Neural Damage</span>
                </div>
                <div className="flex justify-between">
                  <span>Hardware:</span>
                  <span className="text-[#6AE6A0] font-mono">PyTorch</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex justify-center py-2" title="Engine Status">
              <span
                className={`w-3 h-3 rounded-full ${
                  backendOnline === true
                    ? "bg-[#6AE6A0] animate-pulse"
                    : "bg-[#F17878]"
                }`}
              />
            </div>
          )}

          {/* User / Session Badge */}
          <div className="flex items-center gap-2 px-2 py-1 text-xs text-[#768D80]">
            <div className="w-6 h-6 rounded-full bg-[#1D4032] border border-[#294B3C] flex items-center justify-center shrink-0 text-[#6AE6A0] font-bold text-[10px]">
              {isGuest ? "GT" : "AG"}
            </div>
            {!collapsed && (
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-semibold text-[#F2FAF5] truncate">
                  {user?.displayName || "Guest Agronomist"}
                </span>
                <span className="text-[10px] text-[#768D80] truncate">
                  {isGuest ? "Unauthenticated Guest" : "Verified Scout • Active"}
                </span>
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
