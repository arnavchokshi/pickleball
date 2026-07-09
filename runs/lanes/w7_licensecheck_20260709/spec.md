# LANE w7_licensecheck_20260709 — P7-4d training-data + vendored-code LICENSING gate (READ-ONLY)

## HARD RULES
STRICTLY READ-ONLY on the repo: you edit NOTHING outside runs/lanes/w7_licensecheck_20260709/. No commits. No network — work from vendored LICENSE/README files, dataset README.roboflow.txt files, docs, and manifests; mark anything not locally determinable as UNRESOLVED-needs-network (do NOT guess license terms).

## OBJECTIVE (NORTH_STAR R6.4 / Part IV rule 6 2026-07-09 correction: private-use freedom is NOT commercial clearance)
Build the P7-4d licensing inventory vs FUTURE MONETIZATION (Stripe scaffold exists in the product plan). Owner's standing ruling: private use = licenses unconstrained; this gate is about what CHANGES if the product monetizes. Inventory every component with: component | where used (file/stage) | license (from local evidence, cite the file) | commercial-use verdict (OK / restricted / viral-GPL / NC / unknown) | is it load-bearing or swappable | remediation option.
Cover at minimum: PnLCalib (GPL — named in the ruling), the 65 Roboflow Universe datasets (per-dataset licenses in data/roboflow_universe_20260706/*/README.roboflow.txt + manifest.json; summarize the license distribution, name every non-permissive one), WASB/TrackNet/TOTNet/RacketVision vendored code + checkpoints (third_party/), SAM-3D-Body / MHR / SMPL-family body models (their weights' license class matters), harvested YouTube footage (P0-1b — owner waived for private use; state the monetization delta honestly), any font/asset/library in web/replay + ios with distribution terms, Roboflow ToS constraints on trained-model commercialization (from local ToS notes if present; else needs-network).
Deliverable: runs/lanes/w7_licensecheck_20260709/LICENSE_INVENTORY.md (the table + a P7-4d gate verdict: what MUST be resolved/swapped before Stripe flips on) + report.json.

## REPORT
Self-write runs/lanes/w7_licensecheck_20260709/report.json (lane_report.schema.json structure): acceptance = inventory completeness (components covered / unresolved-needs-network count / non-permissive count), full_suite N/A-read-only, BEST-STACK DELTA none, honest_issues, next.
