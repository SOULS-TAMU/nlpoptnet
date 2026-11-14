# Changelog

## 2.0.0

- Add optional original-problem KKT diagnostics and adaptive projection depth during inference.
- Report actual Chambolle–Pock iterations with PURIN 1.0.2 or later.
- Expand CPU tests for formulation, scaling, artifact reload and reporting.

- Add optional structural sparse inequality products to the shared CP forward and implicit adjoint; retain dense equality operations and the existing vector diagonal proximal map. Dense behavior remains the default for library callers.

- Use the renamed `ohagi` and `purin` dependencies without changing frontend APIs.

- Organize shared projection code in PURIN and shared learning services in OHAGI.
- Preserve frontend APIs and JAX metadata inference.
- Add distribution metadata and repository documentation.

## Shared engine and frontend ownership

Framework problem APIs and task policies live in their frontend packages. OHAGI executes all training through shared epoch plans and gradient/update builders; PURIN owns projection mathematics. Historical imports, benchmark call signatures, metadata formats, and checkpoint transport are preserved.

## Shared model bundles and frontend layouts

- DAEHN and NLPOPTNet now use canonical `api/` and `serialization/` directories,
  with historical problem/training imports retained through exact aliases.
- All new model saves use the shared OHAGI JAX bundle and numeric parameter
  codec. Redundant legacy writers, native/portable save switches, and standalone
  StableHLO/derivative exporters were removed.
- JAX serialization is private to OHAGI. Metadata-driven single-instance
  inference and separate Orbax checkpoints remain supported.

- Remove legacy saved-model readers; accept only shared OHAGI JAX bundles.
- Remove top-level `jaxmodel` and `opt` packages; formulation adapters remain under `nlpoptnet.api`.

- Replace legacy optimization/problem hierarchies and private aliases with flat API adapters and one formulation package; preserve model APIs and OHAGI JAX bundles.
