"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { 
  Activity, 
  BarChart3, 
  Clock, 
  Cpu, 
  Home, 
  Leaf, 
  Layers, 
  Scan, 
  ShieldCheck, 
  Sprout 
} from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();
  const [healthStatus, setHealthStatus] = useState<{ status: string; device: string; mps_available: boolean } | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((data) => setHealthStatus(data))
      .catch(() => setHealthStatus(null));
  }, []);

  const navLinks = [
    { href: "/dashboard", label: "Dashboard", icon: Activity },
    { href: "/analyze", label: "Analyze Leaf", icon: Scan },
    { href: "/history", label: "History", icon: Clock },
    { href: "/timeline", label: "Timeline", icon: Sprout },
    { href: "/farms", label: "Farms", icon: BarChart3 },
    { href: "/research", label: "Models & Audit", icon: Cpu },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-emerald-100 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-700 to-green-500 flex items-center justify-center text-white shadow-sm shadow-emerald-700/20 group-hover:scale-105 transition-transform">
              <Leaf className="w-5 h-5 fill-white/20" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight text-slate-900 flex items-center gap-1.5">
                Phyto<span className="text-emerald-600">Trace</span>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 ml-1">
                  AI MVP
                </span>
              </span>
              <p className="text-[11px] text-slate-500 font-medium -mt-1 hidden sm:block">
                Tomato Leaf-Miner Diagnostic Engine
              </p>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href));
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-emerald-50 text-emerald-700 font-semibold"
                      : "text-slate-600 hover:text-emerald-700 hover:bg-slate-50"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? "text-emerald-600" : "text-slate-400"}`} />
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Status Badge & CTA */}
          <div className="flex items-center gap-3">
            {healthStatus && (
              <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200/80 text-xs text-slate-600">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-medium text-slate-700">
                  {healthStatus.mps_available ? "Apple M5 • MPS" : healthStatus.device.toUpperCase()}
                </span>
              </div>
            )}
            <Link
              href="/analyze"
              className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-xl text-sm font-semibold shadow-sm shadow-emerald-700/25 transition-all hover:shadow-md"
            >
              <Scan className="w-4 h-4" />
              <span>Inspect Leaf</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Mobile Sub-Navigation Bar */}
      <div className="md:hidden flex overflow-x-auto py-2 px-4 gap-1 border-t border-slate-100 bg-slate-50/70 scrollbar-none">
        {navLinks.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs whitespace-nowrap ${
                isActive
                  ? "bg-emerald-600 text-white font-medium"
                  : "text-slate-600 hover:bg-white"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {link.label}
            </Link>
          );
        })}
      </div>
    </header>
  );
}

