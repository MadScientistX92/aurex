import type { NextConfig } from "next";

/**
 * Every page is statically generated at build time from the committed JSON in
 * `public-data/`. There is no server-side model execution, no runtime data fetch and
 * no API key anywhere in this app — a deploy is a snapshot of what the engine had
 * published when it ran, and the nightly commit is what triggers the next one.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The data lives outside `web/`, at the repository root. Tracing it explicitly keeps
  // the standalone output honest about what the build actually read.
  outputFileTracingRoot: process.cwd(),
  eslint: { ignoreDuringBuilds: false },
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
