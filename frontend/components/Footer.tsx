import { AlertCircle, CheckCircle2, Shield } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-slate-900 font-bold text-sm">
              <span>PhytoTrace AI</span>
              <span className="text-slate-400">•</span>
              <span className="text-xs font-normal text-slate-500">v0.1.0-alpha MVP</span>
            </div>
            <p className="text-xs text-slate-500 mt-1 max-w-xl">
              Strict scientific data integrity: Real deep neural foliage and lesion instance segmentation with Apple Silicon acceleration. Zero synthetic mask fabrication.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 text-xs text-slate-500">
            <div className="flex items-center gap-1.5 text-emerald-700 font-medium bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-100">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Training Gate: Unlocked (10 Train / 3 Val)</span>
            </div>
            <div className="flex items-center gap-1 text-slate-600">
              <Shield className="w-3.5 h-3.5 text-slate-400" />
              <span>Held-Out Test Set Quarantined</span>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-100 text-[11px] text-slate-400 flex flex-col sm:flex-row justify-between items-center gap-2">
          <p>
            Advisory Notice: This decision is an explainable machine learning aid. Do not apply chemical pesticides or alter IPM schedules without secondary field scout verification.
          </p>
          <p className="shrink-0">Tomato Integrated Pest Management (IPM) System</p>
        </div>
      </div>
    </footer>
  );
}

