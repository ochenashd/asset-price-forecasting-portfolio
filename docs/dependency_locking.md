# Candidate v6 current status

The CI coverage registration now includes all 32 approved test modules, including
the 16 portable documentation checks added in v5. Unknown modules still reject;
all previous modules and all eight Linux-native cases remain mandatory under the
same platform rules. This is a coverage repair, not lock promotion: macOS remains
LOCK_PARTIAL and Linux LOCK_UNAVAILABLE. The real CLI must reject incomplete locks.

P4-D2B subsequently observed a narrow fresh macOS Core installation from 17 verified
wheel inputs, successful pip check and exact replay of 55 predictions. That does
not establish complete build/bootstrap reproducibility. Decision-time evidence
remains incomplete. The following P4-D1 text is preserved historical evidence,
including its then-current no-clean-installation statements and 31-module count.
Private audit filenames identify prior evidence; those reports are not public
source inputs. Current scope and limits are summarized here and in reproducibility.md.

---

# Candidate v4 dependency evidence and integrity contracts

PRIVATE. BLOCKED_BY_IP_OR_LICENSE. These are offline evidence contracts, not
publication permission or a reproduced environment. No clean installation, new
dependency resolution or byte-for-byte double build was performed in P4-D1.
No package was installed or downloaded. Scientific source bytes are unchanged.

## Targets and profiles

The locked interpreter assumption is CPython 3.11.5, within the package's broader
>=3.11,<3.12 metadata. Other patches are not replay-validated. macOS arm64 retains
the existing validated environment evidence; Linux x86-64/glibc is intended only.
macOS 14+ is a proposed native-wheel floor, not validation of all such releases.

| Primary profile | macOS known packages | Current status |
| --- | ---: | --- |
| core | 16 | LOCK_PARTIAL |
| tracking (includes core) | 91 | LOCK_PARTIAL |
| test (full regression, includes tracking) | 94 | LOCK_PARTIAL |
| build | 2 | LOCK_PARTIAL |

All four Linux profiles are LOCK_UNAVAILABLE. The core lacks eight target binary
wheels; tracking and test lack 25, not 25 core wheels. Their unknown greenlet and
nvidia-nccl-cu12 metadata/version/transitive closures are separate additional gaps.
Build has two known universal wheel inputs but Linux build/runtime compatibility
is not established. No Linux installation requirements file is fabricated.

Six original P4-C macOS .partial.txt files are retained byte-for-byte. Primary test
maps to full-tests; core+pytest-only (20 entries) and bootstrap-input-only (pip,
one entry) are auxiliary evidence, not additional competing supported demo profiles.
Core+pytest is insufficient for the full suite. The bootstrap input is not a
verified interpreter/venv/build closure; local ANTLR/SQLAlchemy wheels are retained
source builds, not independently reproduced native artifacts. No LightGBM or
PostgreSQL dependency is added. pyproject.toml and scientific pins are unchanged.

## Manifest contract

requirements/locks/manifest.json uses apf-dependency-evidence-v1. It records the
candidate/v3 source inventory reference, 50 scientific source hashes, exact Python
and target marker contexts, build requirements, 97 normalized artifact records,
filenames/tags/hashes/origin codes, recorded metadata requirements and each frozen
profile's roots/edges/membership/unresolved entries. It includes no private path,
credential, source payload or provider certification. manifest.sha256 binds its
stored bytes. Canonical JSON is deterministic for mapping key order; fixed lists
retain their declared order. These are new dependency evidence identities, never
replacements for research receipts, bundles or scientific fingerprints.

The stdlib-only scripts/verify_portfolio_environment.py validates exact schema,
duplicate keys/packages/profiles, normalized names, bounded known version syntax,
wheel identity/tags, profile separation, reachable frozen dependency graph, build
requirements, missing transitives and explicit status. It is not a new dependency
resolver, full PEP 440/508 engine or independent proof that callers supplied the
correct upstream metadata. Recorded P4-C metadata and markers preserve that lineage.
Coherent alteration of both an unsigned manifest and its companion is not protected
by self-hashing: compare the private evidence inventory/trusted review record.

This schema deliberately cannot express a promoted LOCK_VERIFIED result: editing
status/booleans cannot certify installation/build completeness. Future promotion
needs a separately reviewed schema/evidence procedure. Hash checks verify exact
local bytes, not publisher authenticity, license compliance or absence of malware.
The wheel/platform classifier is bounded to the recorded P4-C filename/tag forms;
it is not native ABI proof. No network lookup, install, model import or inference
is performed by the verifier. Unknown/missing files reject; there is no fallback.

## Supported offline inspection commands

Run from the v4 project checkout using the already validated environment:

```bash
python -B scripts/verify_portfolio_environment.py --manifest requirements/locks/manifest.json --checkout . --target macos-arm64-cp3115 --profile core --mode inspect
python -B scripts/verify_portfolio_environment.py --manifest requirements/locks/manifest.json --checkout . --target linux-x86_64-cp3115 --profile test --mode inspect
```

Inspection can succeed while reporting LOCK_PARTIAL/LOCK_UNAVAILABLE; it never
means an install succeeded. For explicit local byte checks use --mode check-hashes
with repeatable --artifact-root arguments naming reviewed, existing flat artifact
directories. No artifact directories, wheels or source archives are copied here.
This command requires every selected artifact and rejects a missing, corrupted,
symlinked or conflicting same-name file. Linux unresolved entries reject rather
than being counted as verified. A successful macOS hash check remains LOCK_PARTIAL.

--mode require-complete checks actual CPython/host identity and then completeness.
It returns nonzero for all current profiles by design. It is NOT an installation
command and must not be bypassed to present a partial lock as complete. No runnable
partial-lock installation recipe is provided in this stage.

## CI and regression contract

Both workflows remain manual-only, contents:read, pinned official actions and
Python 3.11.5. A stdlib preflight precedes virtualenv/dependency installation. The
research job requires the Linux test lock; the filesystem job has its own explicit
unavailable gate because its minimal OmegaConf 2.3.0 profile is not resolved.
Neither job can currently report lock-enforced success. No CI was dispatched.
Post-gate historical installation commands remain unreachable, not new approved
locked-install commands; they need replacement in a future authorized resolution
stage before promotion. Do not unlock a job merely by editing a status.

The full research driver retains all 30 v3 test modules and adds one dependency
contract module (31 total). It checks manifest, scientific hashes and lock files
before its worker or installed import probe. If eventually authorized with complete
evidence, it still runs the entire tests directory, mandatory scientific/MLflow
cases and all eight native Linux cases on real Linux; no smaller success path.
The v3 CI-contract expected module count is updated 30 to 31 and explicitly asserts
the new module; existing pass/fail/native/financial assertions remain active.

For safe existing-environment developer testing the historical macOS command is
still `python -m pytest tests -k 'not native_linux'`. That is not a new environment
reproduction claim; dependency-locked CI remains blocked. Eight native cases are
unexecuted locally and must never be relabelled passed. This stage runs offline
contract/identity/isolation and relevant regressions; exact counts are in the
private P4D1_VALIDATION.md, not inferred from historical v3 results.

## Open findings and next authorized stage

VULNERABILITY_STATUS_UNVERIFIED; three missing full-license-text evidence items;
no verified clean installation or repeated build; incomplete Linux artifacts and
metadata; no native Linux execution or power-loss/PIT certification. Personal APF
authorship is declared, not an independently verified grant; new/adapted portfolio
files and license scope remain under review. No public LICENSE is applied.

Next authorization must cover exact official-source metadata/artifacts, Linux and
minimal filesystem resolution, bootstrap/build closure, then isolated hash-enforced
installs, real Linux validation and a controlled two-build experiment. Any resolver
conflict is reported, not resolved by speculative pin changes or --no-deps. Preserve
all historical reference/source hashes and scientific behavior throughout.
