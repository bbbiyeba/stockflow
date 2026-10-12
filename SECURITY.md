# Security

Stockflow handles inventory positions, unit costs and branch performance. At a real
distributor that is confidential commercial data. This file states what we assume, what
the code does about it, and what a deployment still has to provide.

## Threat assumptions

| Asset | Threat | Assumed attacker |
|---|---|---|
| Cost, margin and stock data in `data/`, `outputs/`, `reports/` | Disclosure (screenshots, logs, a public repo, an open dashboard) | Curious insiders, accidental sharing |
| Transfer recommendations | Tampering, so stock is moved where it should not be | Someone with write access to output files or the decision log |
| Decision log (`decisions/`) | Repudiation ("I never approved that"), forged approvals | Insider without the approver role |
| Raw extracts from ERP/WMS | Malformed or hostile files (injection via text fields, oversized files) | Upstream system bug, compromised feed |
| Model artefact (`lightgbm_model.txt`) | Swapped for a malicious or stale model | Anyone with write access to `outputs/` |
| Dependencies | Known-vulnerable or typo-squatted packages | Supply chain |

Out of scope: hardening the host OS and network, and the identity provider itself.

## Controls in the code

- **No secrets in the repo.** Nothing in code or git history holds credentials or connection strings. Configuration that varies by environment comes from environment variables (see `.env.example`). `.env` and `.streamlit/secrets.toml` are git-ignored, and a pre-commit hook blocks private keys.
- **Recommendations only.** The optimiser writes a list. Nothing in this repo calls an ERP/WMS or moves stock. Approving in the dashboard appends a record to an audit log; raising the transfer is a separate, manual step.
- **Roles.** `viewer` sees everything, and only `approver` can record decisions (`src/access.py`). With `STOCKFLOW_REQUIRE_LOGIN=1` the app requires Streamlit OIDC sign-in, and approvers are an explicit e-mail allow-list (`STOCKFLOW_APPROVERS`). Demo mode is labelled on every page and must not be used with real data.
- **Tamper-evident decisions.** The log is append-only and file-locked. Each row carries reviewer, role, UTC timestamp, pipeline `run_id` and `transfer_id`. Inputs are validated (id format, decision values, length limits). Notes are flattened to one line and spreadsheet-formula prefixes (`= + - @`) are neutralised.
- **Input validation.** Every raw table is validated at the pipeline boundary before use:
  - pandera schemas cover the small tables;
  - SQL checks cover the large extracts (types, nulls, referential integrity, allowed transaction types, absurd quantities).
  
  Malformed CSVs are rejected with a readable message and exit code 2, not a stack trace. Outputs are validated before they are written.
- **No string-built SQL from inputs.** DuckDB files are read through the relational API, so file paths are arguments. Settings are passed as connection config, and the memory limit is regex-checked. The remaining SQL interpolates only regex-validated identifiers and module constants.
- **No unsafe deserialisation.**
  - No `eval`, `exec` or `pickle`.
  - NumPy archives are loaded with `allow_pickle=False` (the default) and hold only numeric and fixed-width string arrays.
  - The model is saved in LightGBM's text format. A warm start loads it only if its SHA-256 matches the hash recorded in the run manifest.
- **No sensitive detail in the UI.**
  - `showErrorDetails = "none"`, and every loader catches errors and shows a plain message without paths or data.
  - Data-derived text is HTML-escaped before it reaches markup.
  - `STOCKFLOW_MASK_FINANCIALS=1` hides every dollar figure for demos and screenshots.
- **No raw records in logs.** Logs carry counts, timings, run ids and file names only.
- **No third-party calls from the browser.** The US map outlines are bundled in `static/`, so the dashboard does not fetch from a CDN at view time.
- **Pinned, scanned dependencies.** All versions are pinned. CI runs `pip-audit` (known vulnerabilities) and `bandit` (code smells) on every push.

## What a deployment must add

1. Put the dashboard behind company SSO (`STOCKFLOW_REQUIRE_LOGIN=1` + OIDC in `.streamlit/secrets.toml`, or an authenticating reverse proxy) and TLS.
2. Restrict filesystem access. Only the batch job writes `outputs/` and `reports/`, and only the app writes `decisions/`. Back up `decisions/`, or move it to a database table with the same columns (`src/decisions.py` is the only code that touches it).
3. Run the batch job with a read-only database account and pull credentials from a secrets manager, never from files in the repo.
4. Keep `pip-audit` green; rebuild the Docker image when it is not.
5. Treat exported CSVs as confidential, like any other report.

## Reporting a vulnerability

Please open a private security advisory on the repository rather than a public issue.
