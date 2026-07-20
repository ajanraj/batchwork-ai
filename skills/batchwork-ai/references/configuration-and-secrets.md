# Configuration, routes, and secrets

Inspect without reading credential values:

```console
batchwork --json config path
batchwork --json config validate
batchwork --profile work --json config show
```

## Paths and precedence

Configuration path: root `--config`, `BATCHWORK_CONFIG`, then the OS user-config default.
Registry path: root `--registry`, `BATCHWORK_REGISTRY`, then the OS user-data default.

The default config may be absent. An explicit or environment-selected path must exist. On
POSIX, config must be a current-user-owned regular non-symlink file that is not group/other
writable.

Profile selection for new or direct work is root `--profile`, `BATCHWORK_PROFILE`, configured
`default_profile`, then none. Local records retain their route and optional profile; ambient
configuration cannot redirect them.

## Schema

TOML uses `schema_version = 1`, optional `default_profile`, and named profiles. Profiles may
contain:

- `models.text`, `models.embeddings`, and `models.images`;
- per-provider `api_key_env`;
- per-provider `base_url`;
- non-secret literal `headers`;
- secret `header_env` mappings to environment-variable names.

Unknown keys, wrong types, malformed TOML, and unsupported schema versions fail closed. There
is no project config discovery, parent traversal, dotenv loading, shell expansion, or implicit
file merge.

## Secrets

Credentials and secret headers are environment-only. Store variable names, never resolved
values, in config and machine output. Do not inspect or print the full environment. Sensitive
literal headers such as Authorization, Cookie, Proxy-Authorization, and API-key headers are
rejected; unknown secret-bearing headers must also use `header_env`.

Credential-variable selection is explicit flag, selected profile, standard provider variable,
then provider default behavior. A selected variable name that is missing or empty is an error,
not permission to fall through. Google checks `GOOGLE_GENERATIVE_AI_API_KEY` before
`GEMINI_API_KEY`.

## Endpoint trust

Custom base URLs receive provider credentials, configured headers, prompts, media, and
workload metadata. Require explicit authorization before introducing one. Endpoints must be
absolute HTTPS, except HTTP loopback development, and may not contain userinfo, query, or
fragment.

## Safe edits

Only edit config when requested. Resolve the path, preserve unrelated profiles/settings and
restrictive permissions, and make a minimal non-secret change. Validate and show the effective
result. If validation fails, restore the original content and stop.
