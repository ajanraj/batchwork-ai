import { defineConfig } from "blume";

import { docsSidebar } from "./lib/navigation";

export default defineConfig({
  analytics: {
    scripts: [
      {
        attributes: {
          "data-website-id": "f1772f22-87d7-429a-99b7-4b7e119c7918",
        },
        src: "https://umami.ajanraj.com/script.js",
        strategy: "defer",
      },
      // Cloudflare Rocket Loader defers Blume's pre-paint theme script, which
      // makes dark-mode pages paint light first. This copy is marked
      // data-cfasync="false" so Rocket Loader leaves it alone and the theme is
      // set before first paint even when Rocket Loader is enabled.
      {
        attributes: { "data-cfasync": "false" },
        content:
          '(()=>{const s=localStorage.getItem("blume-theme");const sys=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.dataset.theme=s??sys;})();',
      },
    ],
  },
  content: {
    sources: [{ prefix: "docs", root: "docs", type: "filesystem" }],
  },
  description:
    "Unified async Python batch API for OpenAI, Anthropic, Google, Groq, Mistral, Together, and xAI.",
  deployment: { site: "https://batchwork.ajanraj.com" },
  github: {
    branch: "main",
    dir: "docs/docs",
    owner: "ajanraj",
    repo: "batchwork-ai",
  },
  logo: { href: "/", image: "/logo.svg", text: "Batchwork" },
  navigation: {
    sidebar: docsSidebar,
    tabs: [{ label: "Docs", path: "/docs" }],
  },
  redirects: [
    { from: "/docs/guides/client", status: 301, to: "/docs/guides/jobs" },
  ],
  theme: { accent: "orange" },
  title: "Batchwork",
});
