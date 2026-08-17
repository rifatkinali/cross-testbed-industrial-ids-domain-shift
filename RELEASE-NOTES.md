# v1.0.0-rc2 — Preregistered cross-testbed correction

RC2 replaces the unpublished private RC1 analysis with a result-blind,
run-separated protocol. It is intended to invite independent reproduction,
methodological criticism, and support for a second authorized testbed.

## Why this release matters

Protocol and process-aware models achieved median OpenPLC validation AUCs of
0.90–0.97 but only 0.32–0.40 event AUC on the held-out MaCySTe campaign. The
strong source-lab result therefore did not transfer in this case study.

This is evidence about one transfer direction and one model family—not all
industrial IDSs, real vessels, or a commercial product.

## Corrections since private RC1

- main training is restricted to `flow/modbus` and `normal/attack` events;
- all eight fault-bearing run contexts are excluded from model fitting and
  threshold selection;
- source splits are run-level and use all 20 preregistered seeds (`42–61`);
- 3,337 fault events from eight unseen benign runs are evaluated separately;
- the target point estimate and scripted-run sensitivity interval use the same
  20-model ensemble score;
- MaCySTe threshold-dependent false-positive and recall values are descriptive,
  not headline claims;
- the release carries no `date-released` until it is actually published;
- the archive root is exactly versioned and contains no staging suffix.

## Main frozen values

| Schema | Source validation AUC, median | MaCySTe ensemble event AUC | Scenario-balanced AUC |
|---|---:|---:|---:|
| `flow` | 0.723309 | 0.321893 | 0.393842 |
| `protocol` | 0.904289 | 0.318365 | 0.389290 |
| `physical_strict` | 0.936170 | 0.328890 | 0.384612 |
| `physical_proxy` | 0.968221 | 0.400285 | 0.454040 |

## Support wanted

Useful next steps are an independent clean-machine R1 rerun, a second testbed
or reverse-transfer experiment, baseline comparisons, and conversations with
authorized maritime laboratories, simulator operators, integrators, owners,
research sponsors, or design partners.

Use a GitHub issue for public reproduction evidence or method discussion. For
a private introduction, contact **info@nauticmall.com**. Do not share vessel
data or sensitive captures publicly.
