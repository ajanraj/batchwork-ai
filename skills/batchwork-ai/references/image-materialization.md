# Image materialization

Retrieval never writes files unless `--output-dir` is explicit:

```console
batchwork --jsonl results JOB --output-dir ./generated-images
```

For a direct reference also provide `--modality images`. Obtain explicit user authorization
for the destination and operation.

Before provider mutation or retrieval, the target must be absent or an empty non-symlink
directory. Batchwork may create an absent directory with user-only permissions. Never clear a
directory, overwrite files, follow a symlink, silently append a suffix, or select a different
target.

Inline image data wins when both data and URL exist; an invalid inline value does not fall back
to URL. Downloads are bounded HTTPS requests with redirect and address validation. Provider
credentials, configured headers, cookies, authorization, source URLs, and referrers are not
forwarded to image hosts or written to the manifest.

PNG, JPEG, GIF, and WebP receive detected extensions; other valid image media uses `.bin`.
Each complete image is written atomically with a deterministic custom-ID/hash/index filename.
`manifest.json` is atomically updated after each image and records canonical identity,
relative path, `custom_id`, image index, source kind, media type, byte count, and SHA-256.

The limits are 20 MiB per decoded image, 200 MiB aggregate materialization, a 30-second
connect/read timeout, and at most five redirects.

On partial failure, preserve completed files, manifest entries, complete machine records, and
partial counters. Never restart result retrieval automatically. A later user-directed replay
starts from the beginning and deduplicates by canonical job identity plus `custom_id`.
