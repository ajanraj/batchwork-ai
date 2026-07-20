# Commands and selectors

Global controls must precede the command:

```console
batchwork --profile work --json --quiet status JOB
```

Global controls are `--config`, `--registry`, `--profile`, `--human`, `--json`, `--jsonl`,
`--quiet`, `--progress`, `--color`, and `--no-color`.

## Creation

```console
batchwork submit text SOURCE --model PROVIDER/MODEL
batchwork submit embeddings SOURCE --model PROVIDER/MODEL
batchwork submit images SOURCE --model PROVIDER/MODEL

batchwork run text SOURCE --model PROVIDER/MODEL
batchwork run embeddings SOURCE --model PROVIDER/MODEL
batchwork run images SOURCE --model PROVIDER/MODEL
```

`submit` returns after provider acceptance and local registration. `run` is transparent
`submit` → `wait` → `results`. It emits identity immediately in JSONL mode. A timeout or
signal leaves remote work running.

Use `--timeout 30m` and a positive `--poll-interval` only on `run` or `wait`. The default wait
is unlimited. `--output-dir` is valid only for image `run` and image `results`.

## Lifecycle

```console
batchwork --json status JOB
batchwork --json wait JOB --timeout 2h
batchwork --jsonl results JOB
batchwork --json cancel JOB
```

- `status` refreshes once and exits 0 for any successfully observed state.
- `wait` polls until terminal. It exits 0 only for completed and 6 for another terminal state.
- `results` refreshes once, never waits, and retrieves available output for a terminal job.
- `cancel` refreshes first, sends at most one cancellation request, and is a no-op for a
  terminal job.

Never automatically repeat any invocation.

## Selectors and route identity

`JOB` accepts:

- local alias for interactive convenience;
- immutable `bw_` record ID;
- direct `provider:provider-job-id`;
- bare provider job ID with explicit `--provider`.

For machine continuity retain the record ID. If unregistered, retain the direct provider
reference plus `routing_fingerprint` from machine output. Never infer provider, route, account,
credential variable, or endpoint.

Direct operations bypass registry lookup and mutation. `--save` adopts a direct job only after
a successful provider operation; it and optional `--name` require explicit authorization.
For direct image adoption use `--modality images`.

Local selectors use their persisted immutable routing descriptor. A root profile is a
compatibility check, not permission to reroute. On mismatch, use a direct reference and, if
authorized, save a distinct record.

## Local administration

```console
batchwork --json list --provider openai --status completed --limit 20
batchwork --json forget JOB
batchwork --json prune --older-than 30d
batchwork --json prune --older-than 30d --yes
batchwork --json config path
batchwork --json config validate
batchwork --json config show
batchwork --json registry check
batchwork --json registry reset --backup
```

`list` reads only cached local metadata. Repeated status filters are ORed; other filters are
ANDed. `forget`, committed prune, and reset never cancel or delete provider jobs. Prune without
`--yes` is a preview.
