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
    ],
  },
  content: {
    sources: [{ prefix: "docs", root: "docs", type: "filesystem" }],
  },
  description:
    "Unified async Python batch API for OpenAI, Anthropic, Google, Groq, Mistral, Together, and xAI.",
  deployment: { site: "https://batchwork.ajanraj.com" },
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
