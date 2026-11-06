"use client";

import { useEffect, useState } from "react";
import { useLanguage } from "@/context/LanguageContext";

interface ModelInventory {
  device: string;
  models: Array<{ id: string; name: string; checkpoint: string; loaded: boolean }>;
}

export default function ResearchPage() {
  const { t } = useLanguage();
  const [inventory, setInventory] = useState<ModelInventory | null>(null);
  const [error, setError] = useState(false);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/v1/models/status", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error();
        return response.json();
      })
      .then((data) => { setInventory(data); setError(false); })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, [refresh]);

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-3xl font-bold">{t("nav.research")}</h1>
        <button onClick={() => setRefresh((value) => value + 1)} className="rounded-xl border border-[#294B3C] px-4 py-2">
          Refresh model inventory
        </button>
      </header>
      <p className="text-[#A6B9AE]">
        This inventory shows which checkpoints the server loaded. The scanner reports probability and candidate checks for each photograph.
      </p>
      {error ? <p role="alert">Model inventory unavailable.</p> : !inventory ? <p>Loading model inventory…</p> : (
        <>
          <p className="font-mono text-sm">Device {inventory.device}</p>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {inventory.models.map((model) => (
              <article key={model.id} className="space-y-3 rounded-2xl border border-[#294B3C] bg-[#152C22] p-5">
                <h2 className="font-bold">{model.name}</h2>
                <p className="text-sm">{model.loaded ? "Checkpoint loaded" : "Checkpoint unavailable"}</p>
                <p className="break-all font-mono text-xs text-[#A6B9AE]">{model.checkpoint}</p>
              </article>
            ))}
          </div>
        </>
      )}
      <aside className="rounded-2xl border border-[#294B3C] p-5 text-sm text-[#A6B9AE]">
        The models are experimental. Mine-like patterns require field confirmation. Species, disease and infestation stage are undetermined.
      </aside>
    </section>
  );
}
