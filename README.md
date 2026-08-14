# Cross-Testbed Domain Shift in Industrial IDS

## An OpenPLC → MaCySTe reproducible case study

> **Release state:** local release candidate; not yet a public release  
> **Primary result:** an IDS trained and thresholded only on OpenPLC does not
> transfer reliably to the held-out MaCySTe campaign used here.

The result motivates—but does not validate—a product architecture based on
deployment-specific discovery, semantic mapping, local calibration, explicit
degraded-state reporting, and change-triggered revalidation. The private
GÖZCÜ EDGE implementation of those capabilities is not part of this repository.

This is the deliberately narrow, publishable research slice of Maritime-Lab.
It is designed to support thesis evaluation, independent technical scrutiny,
and early product discovery without publishing the GÖZCÜ EDGE product source,
customer delivery methods, or production security configuration.

## Sixty-second result

| Feature schema | OpenPLC validation AUC | MaCySTe event AUC | MaCySTe scenario-balanced AUC |
|---|---:|---:|---:|
| `flow` | 0.9543 | 0.3220 | 0.3940 |
| `protocol` | 0.9759 | 0.3186 | 0.3895 |
| `physical_strict` | 0.9813 | 0.3289 | 0.3845 |
| `physical_proxy` | 0.9842 | 0.4001 | 0.4595 |

The threshold was selected only on OpenPLC for a target false-positive rate of
1%. On the held-out MaCySTe campaign, the observed false-positive rate rises to
approximately 26–50%, depending on the feature schema. AUC values below 0.5
mean that the target-domain ranking is not merely weaker; it can be inverted.

The result is bounded to one transfer direction, one OpenPLC–MaCySTe testbed
pair, one 200-tree Random Forest model family, and the included attack/fault
implementations. It is not evidence about all industrial IDSs, cross-vendor
performance, or real-vessel operation. `physical_proxy` is the least degraded
schema in this case study, but that is a secondary hypothesis-generating
observation—not evidence that process-aware features generally transfer.

## What can be tested

The package has four reproducibility levels:

| Level | Meaning | Current state |
|---|---|---|
| R0 | Verify every distributed file and frozen result by SHA-256 | Available |
| R1 | Re-run tests and the complete analysis from the 12 derived MaCySTe event files | Available |
| R2 | Re-create derived events from the exact raw PCAP/EVE campaign | Pending external immutable data archive/DOI |
| R3 | Independent party repeats the method on another testbed or capture | Not yet completed |

R1 is the minimum useful GitHub research artifact: a fresh clone can recompute
the headline result without access to the private product repository. R2 should
be delivered through Zenodo or another immutable research-data archive rather
than placing the approximately 200 MiB raw campaign in Git history.

## Quick verification

```bash
python scripts/verify_artifact.py
```

This verifies `artifact-manifest.json`, rejects unexpected product/sensitive
paths, and checks all distributed bytes.

## Full derived-data reproduction

Create an isolated environment, install the pinned research dependencies, then
run:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_openplc_macyste.py
```

The command:

1. runs the 153 academic contract tests;
2. verifies the OpenPLC dataset snapshot;
3. validates the 12 MaCySTe derived files and their frozen manifest;
4. rebuilds the exact 34,949-row combined target table;
5. recomputes LODO, unseen-category, stratification, and score-discreteness
   results;
6. compares all five new JSON outputs with the frozen results after normalizing
   only the host-specific absolute path of the temporary combined CSV.

The executable analysis modules and their contract tests are exported from the
exact `v0.5.0-academic-freeze` Git tag, not from the evolving product working
tree. The tag and commit are recorded in `artifact-manifest.json`; the original
full runtime dependency file and freeze entrypoint are retained under
`provenance/`. The release-specific test dependency is security-maintained in
the root `requirements.txt` rather than copying the historical development pin.

Use `--skip-tests` only for a faster local rerun after a full verified run.

## Scientific contribution

The artifact tests a narrower and more defensible question than “does ML work
for maritime cybersecurity?”:

> How does an IDS trained and calibrated in one maritime OT testbed behave
> under a held-out testbed with different protocol encodings and operational
> distributions, and which observable mechanisms explain transfer failure?

The contribution is the combination of:

- a leakage-resistant cross-testbed protocol;
- source-only threshold selection;
- event-weighted and scenario-balanced reporting;
- explicit unseen-category mass diagnostics;
- score-support/threshold plateau diagnostics;
- machine-verifiable provenance and result boundaries.

## Product-learning boundary

The artifact can test an important product hypothesis without being the
product itself: if static models do not transfer safely between testbeds, a
commercial observer may need vessel-specific discovery, semantic mapping,
calibration, degraded-state reporting, and evidence-backed change management.

Evidence that would strengthen this hypothesis:

- independent R1 reproductions;
- the same experiment using IPAL/SIMPLE/GeCo baselines;
- a second independent target testbed or reverse-direction transfer;
- a qualified integrator or owner confirming that calibration and evidence
  gaps are a purchasing problem;
- later, authorized simulator/VDR/field validation.

Repository stars alone are not product validation. Useful signals are external
reproductions, citations, dataset reuse, integration pull requests, qualified
industry conversations, and a paid design-partner path.

## Deliberately excluded

This artifact must not contain:

- `edge_agent`, `edge_console`, `edge_forwarder`, or production deployment
  source;
- production PKI, IAM, firewall, trust-root, update, or support configuration;
- executable attack campaign clients or evasion material;
- customer, vessel, topology, pricing, or field-delivery records;
- raw security scan/remediation reports;
- real-vessel or restricted validation data;
- private keys, credentials, runtime logs, PCAP/PCAPNG, or debrief material.

## Publication blockers

The candidate is technically testable, but public release still requires:

1. replacing placeholder author/repository fields in `CITATION.cff`;
2. confirming contributor ownership for project-owned code and data;
3. completing an IP/patent disclosure decision before first public disclosure;
4. archiving the raw campaign and provenance record under a DOI if
   R2 reproduction will be claimed;
5. performing privacy, secret, malware, and executable-scenario review on the
   generated archive;
6. running the release from a clean commit and recording the commit/tag/DOI.

## Licenses and provenance

Maritime-Lab software included in the candidate currently follows the root MIT
license. The OpenPLC-derived research dataset and the project-owned MaCySTe
experimental observations are declared CC BY 4.0 in the dataset card.

The MaCySTe-derived data distribution review was completed on 14 August 2026.
AGPL-3.0 section 2 states that output from running a covered work is covered
only when the output's content constitutes a covered work. The 12 distributed
`events-v0.4.csv` files contain generated protocol/process observations,
project labels, and project-computed fields; they contain no MaCySTe source
code. The publication decision is therefore to distribute those observations
under CC BY 4.0 with explicit MaCySTe attribution. This decision does not
relicense MaCySTe source code. Upstream attribution, license, and README are
preserved in `THIRD_PARTY_LICENSES/`.
