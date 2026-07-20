# Inputs, defaults, and provider options

## Input transports

One creation command accepts one regular file or `-` for stdin.

- JSON: one canonical request object or a non-empty array.
- JSONL: one canonical object per non-empty line.
- CSV: constrained scalar fields for the selected modality.
- Text: one prompt, embedding value, or image prompt per non-whitespace line.

Known file extensions select the format. stdin and unknown extensions require explicit
`--format`. Batchwork never sniffs content. Every record is parsed and validated before remote
work; any invalid record rejects the complete source with source coordinates.

JSON and JSONL map directly to `BatchRequest`, `BatchEmbeddingRequest`, or
`BatchImageRequest`. Preserve explicit `custom_id`. Missing IDs become `request-0`,
`request-1`, and so on; duplicates and explicit/generated collisions fail locally.

Structured media paths are resolved relative to the workload file or current directory for
stdin, then frozen during preflight. Ordinary text and provider-option strings are never
treated as paths.

## Canonical defaults

Creation flags apply only when the matching record value is null or absent. Record values win.
Lists and objects replace atomically.

Text defaults include system, output-token limit, sampling controls, stop sequences, literal
tool choice, and endpoint. Embeddings expose dimensions. Images expose count, aspect ratio,
seed, and size. Providers reject known unsupported canonical settings before network access.

`--batch-metadata KEY=VALUE` is provider-retained submission metadata. Never put credentials,
secret headers, prompts, or sensitive content in it.

## Provider options

For a small selected-provider inner object:

```console
batchwork submit text requests.jsonl --model openai/gpt-5 \
  --provider-options '{"reasoningEffort":"high"}'
```

For nested or nontrivial JSON use `--provider-options-file`. Do not include the outer provider
key in command-level files. Canonical records retain the outer form:

```json
{"provider_options":{"openai":{"reasoningEffort":"high"}}}
```

Keys are exact and case-sensitive. Command options are a shallow base; record keys replace
matching keys, and nested values replace atomically. Unknown keys fail except Together's
documented passthrough. Together still rejects reserved canonical fields. Any option that
collides semantically with a canonical setting fails; neither silently wins.

Consult the shipped provider documentation for implemented keys, endpoints, collisions, value
shapes, and limits. Provider APIs remain authoritative for model-specific constraints.

## Volume authorization

Hard limits are 50,000 requests, 20 MiB per serialized provider request, and 200 MiB aggregate
upload, with lower provider limits authoritative. `--allow-large-batch` never bypasses hard
limits.

The soft gate is above 10,000 requests, 50 MiB aggregate serialized upload, or 100 requested
generated images. Report the measured gate and obtain separate explicit authorization before
adding `--allow-large-batch`; never answer an interactive prompt on the user's behalf.
