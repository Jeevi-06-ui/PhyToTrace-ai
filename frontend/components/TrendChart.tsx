"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface DataPoint {
  date: string;
  damagePercentage: number;
  label?: string;
}

export default function TrendChart({ data, title = "Estimated damage area (%)", height = 200 }: {
  data: DataPoint[];
  title?: string;
  height?: number;
}) {
  if (!data.length) {
    return <p className="grid place-content-center rounded-xl bg-[#10231B] p-6 text-xs text-[#A6B9AE]" style={{ height }}>
      No measurable inspections in this view
    </p>;
  }
  return (
    <figure className="w-full min-w-0" aria-label={title} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} accessibilityLayer>
          <CartesianGrid stroke="#294B3C" strokeDasharray="3 3" />
          <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} tick={{ fill: "#A6B9AE", fontSize: 10 }} />
          <YAxis domain={[0, 100]} width={45} unit="%" tick={{ fill: "#A6B9AE", fontSize: 10 }} />
          <Tooltip labelFormatter={(value) => new Date(String(value)).toLocaleString()} contentStyle={{ background: "#10231B", borderColor: "#294B3C" }} />
          <Area dataKey="damagePercentage" name="Estimated damage area (%)" stroke="#6AE6A0" fill="#1D4032" dot isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </figure>
  );
}
