# Dependency Provenance & Network Hygiene

All Python dependencies are installed from the official PyPI registry over HTTPS.

| Package | Version | Source | Notes |
|---------|---------|--------|-------|
| lean-dojo | 4.20.0 | PyPI / github.com/lean-dojo | Maintainer: Kaiyu Yang (Meta). MIT license |
| ray | 2.51.2 | PyPI / Anyscale | Usage telemetry disabled via `RAY_USAGE_STATS_ENABLED=0` (set in lean/train.py) |
| opentelemetry-*, opencensus | — | PyPI / CNCF | Library only; no exporter configured — sends nothing |
| pygithub | 2.9.1 | PyPI | Used by LeanDojo for repo metadata; only active when tracing repos (we use the offline benchmark — no calls) |
| torch, numpy, einops | — | PyPI | Standard |

Data sources:

| Dataset | Source | Integrity |
|---------|--------|-----------|
| LeanDojo Benchmark 4 | zenodo.org/record/12740403 (DOI 10.5281/zenodo.8242200) | Official Zenodo deposit; Mathlib4 @ 29dcec07 |

Training is fully offline: dataset is processed locally; no network is required
or used during training/evaluation. Telemetry env vars are set in `lean/train.py`.
