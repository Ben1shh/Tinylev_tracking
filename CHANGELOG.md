# Changelog

## 2026-10-05 — full Experiment Manager v0.5 source release

- Package experiment/catalog management, still-image center calibration,
  full/fast tracking, manager annotations and Phase A-D dependencies.
- Add portable root launchers, dependency list, Windows CI, synthetic tests,
  source provenance hashes and a reproducible UI screenshot.
- Replace historical tracking center/scale and analysis center/error defaults
  with unset values. Require explicit finite calibration for tracking and
  explicit center/error values for analysis. Clear calibration across experiments.
- Preserve old root-level v0.1 package under a separate legacy launcher;
  the existing default launcher now opens the manager.
- Reuse the existing identical model. Exclude experiment data, private catalogs,
  notebooks, manuscript files, outputs, caches and local environments.

Tracking formulas and identity algorithms are not merged between versions.
Existing source workspaces and historical scientific outputs are untouched.
