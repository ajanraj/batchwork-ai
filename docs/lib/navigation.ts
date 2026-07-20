export interface FooterLink {
  href: string;
  label: string;
}

export interface FooterSection {
  links: FooterLink[];
  title: string;
}

export const footerSections: FooterSection[] = [
  {
    links: [
      { href: "/docs", label: "Overview" },
      { href: "/docs/installation", label: "Installation" },
      { href: "/docs/configuration", label: "Configuration" },
      { href: "/docs/examples", label: "Examples" },
    ],
    title: "Start",
  },
  {
    links: [
      { href: "/docs/guides/jobs", label: "Jobs" },
      { href: "/docs/guides/results", label: "Results" },
      { href: "/docs/guides/server", label: "Polling and webhooks" },
      { href: "/docs/guides/stores", label: "Stores" },
      { href: "/docs/guides/security", label: "Security" },
    ],
    title: "Guides",
  },
  {
    links: [
      { href: "/docs/modalities/text", label: "Text" },
      { href: "/docs/modalities/embeddings", label: "Embeddings" },
      { href: "/docs/modalities/images", label: "Images" },
      { href: "/docs/providers", label: "Provider overview" },
    ],
    title: "Workloads",
  },
  {
    links: [
      { href: "/docs/api", label: "Public API" },
      { href: "/docs/faq", label: "FAQ" },
      { href: "https://github.com/ajanraj/batchwork-ai", label: "GitHub" },
      { href: "https://pypi.org/project/batchwork-ai/", label: "PyPI" },
      { href: "https://github.com/ajanraj/batchwork-ai/issues", label: "Issues" },
    ],
    title: "Reference",
  },
];

export const docsSidebar = [
  {
    label: "Docs",
    root: "/docs",
    items: [
      {
        label: "Getting started",
        items: [
          "/docs",
          "/docs/installation",
          "/docs/cli",
          "/docs/configuration",
          "/docs/examples",
        ],
      },
      {
        label: "Guides",
        items: [
          "/docs/guides/jobs",
          "/docs/guides/results",
          "/docs/guides/cli-exits",
          "/docs/guides/classification",
          "/docs/guides/server",
          "/docs/guides/stores",
          "/docs/guides/security",
        ],
      },
      {
        label: "API reference",
        items: [
          "/docs/api",
          "/docs/reference/cli-machine-schema",
          "/docs/reference/cli-configuration-registry",
          "/docs/modalities/text",
          "/docs/modalities/embeddings",
          "/docs/modalities/images",
        ],
      },
      {
        label: "Providers",
        items: [
          "/docs/providers",
          "/docs/providers/openai",
          "/docs/providers/anthropic",
          "/docs/providers/google",
          "/docs/providers/groq",
          "/docs/providers/mistral",
          "/docs/providers/together",
          "/docs/providers/xai",
        ],
      },
      {
        label: "Help",
        items: ["/docs/faq"],
      },
    ],
  },
];
