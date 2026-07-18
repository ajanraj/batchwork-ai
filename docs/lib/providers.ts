export interface ProviderSummary {
  apiKey: string;
  icon: string;
  inputs: string[];
  metadata: "Forwarded" | "Ignored";
  name: string;
  outputs: {
    embeddings: boolean;
    images: boolean;
    text: boolean;
  };
  results: string;
  route: string;
  slug: string;
  submission: string;
}

export const providers: ProviderSummary[] = [
  {
    apiKey: "OPENAI_API_KEY",
    icon: "/logos/openai.svg",
    inputs: ["Images", "PDFs with Responses"],
    metadata: "Forwarded",
    name: "OpenAI",
    outputs: { embeddings: true, images: true, text: true },
    results: "Output and error JSONL files",
    route: "/docs/providers/openai",
    slug: "openai",
    submission: "JSONL file upload",
  },
  {
    apiKey: "ANTHROPIC_API_KEY",
    icon: "/logos/anthropic.svg",
    inputs: ["Images", "PDFs"],
    metadata: "Ignored",
    name: "Anthropic",
    outputs: { embeddings: false, images: false, text: true },
    results: "Same-origin JSONL result URL",
    route: "/docs/providers/anthropic",
    slug: "anthropic",
    submission: "Inline Message Batch",
  },
  {
    apiKey: "GOOGLE_GENERATIVE_AI_API_KEY",
    icon: "/logos/gemini.svg",
    inputs: ["Images", "Files URLs", "YouTube URLs"],
    metadata: "Ignored",
    name: "Google Gemini",
    outputs: { embeddings: true, images: true, text: true },
    results: "Inline operation response",
    route: "/docs/providers/google",
    slug: "google",
    submission: "Inline long-running operation",
  },
  {
    apiKey: "GROQ_API_KEY",
    icon: "/logos/groq.svg",
    inputs: ["Images"],
    metadata: "Forwarded",
    name: "Groq",
    outputs: { embeddings: false, images: false, text: true },
    results: "Output and error JSONL files",
    route: "/docs/providers/groq",
    slug: "groq",
    submission: "JSONL file upload",
  },
  {
    apiKey: "MISTRAL_API_KEY",
    icon: "/logos/mistral.svg",
    inputs: ["Images", "PDFs"],
    metadata: "Forwarded",
    name: "Mistral",
    outputs: { embeddings: true, images: false, text: true },
    results: "Output and error JSONL files",
    route: "/docs/providers/mistral",
    slug: "mistral",
    submission: "JSONL file upload",
  },
  {
    apiKey: "TOGETHER_API_KEY",
    icon: "/logos/together.svg",
    inputs: ["Images", "PDFs", "Text files", "Audio"],
    metadata: "Forwarded",
    name: "Together AI",
    outputs: { embeddings: false, images: false, text: true },
    results: "Output and error JSONL files",
    route: "/docs/providers/together",
    slug: "together",
    submission: "Presigned JSONL upload",
  },
  {
    apiKey: "XAI_API_KEY",
    icon: "/logos/xai.svg",
    inputs: ["Images", "Text files", "PDFs"],
    metadata: "Ignored",
    name: "xAI",
    outputs: { embeddings: false, images: true, text: true },
    results: "Paginated results endpoint",
    route: "/docs/providers/xai",
    slug: "xai",
    submission: "JSONL file upload",
  },
];
