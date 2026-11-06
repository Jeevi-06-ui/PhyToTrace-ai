"use client";

import { useState, useEffect } from "react";
import AppSidebar from "@/components/AppSidebar";
import AppHeader from "@/components/AppHeader";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Auto-collapse sidebar on smaller laptop screens (under 1280px)
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1280 && window.innerWidth >= 1024) {
        setCollapsed(true);
      }
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="min-h-screen bg-[#0A1814] text-[#F2FAF5] flex flex-col">
      <AppSidebar
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div
        className={`flex-1 flex flex-col transition-all duration-300 ${
          collapsed ? "lg:pl-20" : "lg:pl-64"
        }`}
      >
        <AppHeader
          collapsed={collapsed}
          onOpenMobile={() => setMobileOpen(true)}
        />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1920px] w-full mx-auto">
          {children}
        </main>

        <footer className="border-t border-[#294B3C] py-6 px-6 text-center text-xs text-[#768D80] bg-[#101F19]">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 max-w-7xl mx-auto">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#6AE6A0]"></span>
              <span className="font-semibold text-[#A6B9AE]">PhytoTrace AI Engine v1.0.0</span>
              <span className="text-[#294B3C]">•</span>
              <span>Tomato Canopy Defense System</span>
            </div>
            <div className="flex items-center gap-4 text-[11px]">
              <span className="text-[#A6B9AE]">Canopy Boundary Segmenter</span>
              <span className="text-[#294B3C]">•</span>
              <span className="text-[#A6B9AE]">Damage Instance Detector</span>
              <span className="text-[#294B3C]">•</span>
              <span className="text-[#6AE6A0]">Experimental models</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

