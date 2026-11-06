import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  distDir: process.env.PHYTOTRACE_NEXT_DIST_DIR || ".next",
  async rewrites() {
    const backend = process.env.PHYTOTRACE_API_ORIGIN || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
