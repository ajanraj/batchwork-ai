# Failures and recovery

Interpret stdout, stderr, and process status together. Preserve complete stdout records before
handling the single structured `error` envelope on stderr.

Diagnostics and examples must never expose raw provider response bodies, arbitrary headers,
prompts, media, signed or source URLs, credentials, or environment values. Relay only fields
permitted by a validated Batchwork machine envelope.

## Exit categories

- 1: unexpected internal failure.
- 2: usage, input, capability, or preflight failure.
- 3: configuration or credentials.
- 4: definite provider rejection.
- 5: transport, availability, or provider protocol.
- 6: unsuccessful job state or partial item outcome.
- 7: local wait timeout; remote job unchanged.
- 8: local registry, persistence, or output preservation.
- 130: interrupted; remote job unchanged.
- 143: terminated; remote job unchanged.

`status` exits 0 whenever observation succeeds, regardless of terminal outcome. `wait` exits 0
only for completed. `results` may emit useful items and then exit 6. Do not discard output
because status is nonzero.

## Structured recovery

Use stable `error.code`, not prose, to choose recovery. Check `error.recovery.command` before
presenting or executing it. It must preserve the exact record ID or direct provider reference,
route fingerprint, profile/credential-variable route, and output destination.

Never automatically repeat any failed invocation. `retryable: true` means a later invocation
may be safe after a condition changes or the user directs it; it is not authorization for
automatic retry.

Never retry submission, upload, batch creation, cancellation, or registry mutation. If
`submission_outcome` is `unknown`, warn about duplicate cost and do not resubmit. If outcome is
`accepted` but registry write failed, retain the direct `provider:provider-job-id` and
`routing_fingerprint`, stop `run`, and use direct lifecycle inspection. Adopt with `--save`
only after explicit authorization.

Timeout, SIGINT, SIGTERM, broken pipe, output failure, and registry failure do not cancel
remote work. Resume with the same canonical selector. Cancellation is a distinct guarded
operation.

After partial JSONL or image output, preserve complete records, files, and the manifest. Replay
from the beginning and deduplicate by canonical identity plus `custom_id`.

For local registry damage, run `registry check`. Reset only with explicit authorization and
the required `registry reset --backup`; preserve the reported recovery-set path. Direct
lifecycle operations remain the fallback when local continuity is unavailable.
