# P0-A Baseline and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository's current K3 baseline, target architecture and source-review obligations explicit before any DeepSeek or q36 code is integrated.

**Architecture:** P0-A is documentation and process only. It adds an auditable source register and contribution gate while preserving every tracked runtime/build/test byte except the README and contributor documentation. No binary, namespace, test target or model behavior changes.

**Tech Stack:** Markdown, Git, existing GitHub pull-request template, existing K3 build/test commands.

## Global Constraints

- Target baseline is `main@85ab2cd901aa81b70caac7711f06864d594b8ff3`.
- Existing K3 code remains Apache-2.0 and its `NOTICE` remains intact.
- q36-derived code is MIT and must preserve its copyright and permission notice.
- No model weights or generated packed artifacts are committed.
- No claim of DeepSeek support is made in P0-A.
- Every future external integration must first review the owning repository at an exact commit.
- `make test` and `make portable` remain the baseline verification commands.

---

### Task 1: Record the approved design and current project state

**Files:**
- Create: `docs/superpowers/specs/2026-08-06-deepseek-v4-bc250-integrated-design.md`
- Create: `docs/PROJECT_STATUS.md`
- Modify: `README.md:1`

**Interfaces:**
- Consumes: approved integrated implementation document and source audit.
- Produces: authoritative design path and visible repository status links used by later PRs.

- [ ] **Step 1: Verify the source baseline before editing**

Run:

```bash
git status --short
git rev-parse HEAD
git remote -v
git branch -a
```

Expected: clean tree; HEAD `85ab2cd901aa81b70caac7711f06864d594b8ff3` or an explicitly reviewed descendant.

- [ ] **Step 2: Add the approved integrated design**

Create the exact design document supplied with this plan. It must state KEEP,
PORT, REWRITE, EXPERIMENT and DEFER boundaries, workstream order and gate rules.

- [ ] **Step 3: Add the project status document**

Create `docs/PROJECT_STATUS.md` and state explicitly that the current runtime is
still Kimi K3, that DeepSeek code is not implemented, and that P0-A is
non-semantic.

- [ ] **Step 4: Add a README status banner**

Insert before the existing `<div align="center">`:

```markdown
> [!IMPORTANT]
> **Development status (2026-08-06):** this repository currently contains the
> unchanged Kimi K3 CPU runtime inherited from `FareedKhan-dev/kimi-k3-in-c`.
> DeepSeek V4 Flash support is under design and is not implemented yet. See
> [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) and
> [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
>
```

Do not rewrite or remove the historical K3 documentation in this task.

- [ ] **Step 5: Check links and diff scope**

Run:

```bash
git diff --check
git diff -- README.md docs/PROJECT_STATUS.md \
  docs/superpowers/specs/2026-08-06-deepseek-v4-bc250-integrated-design.md
```

Expected: only the banner and new documents; no source/build changes.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/PROJECT_STATUS.md \
  docs/superpowers/specs/2026-08-06-deepseek-v4-bc250-integrated-design.md
git commit -m "docs: record DeepSeek migration status and design"
```

### Task 2: Add the provenance register and mandatory source-review checklist

**Files:**
- Create: `docs/PROVENANCE.md`
- Create: `docs/SOURCE_REVIEW_CHECKLIST.md`

**Interfaces:**
- Consumes: exact repository refs and license findings from the P0 audit.
- Produces: required provenance schema consumed by every later porting PR.

- [ ] **Step 1: Re-resolve source refs**

Check the target fork, K3 upstream, q36 main and relevant q36 branches. Check the
official DeepSeek model/inference repository and record the observed revision.
Do not reuse a branch name as if it were immutable.

- [ ] **Step 2: Write the provenance register**

Record source repository, full SHA/revision, license/NOTICE, decision and import
status. State that no implementation code is imported by P0-A.

- [ ] **Step 3: Write the review checklist**

Require repository/ref, branch/PR/recent-commit review, license/NOTICE, source
and test inspection, target comparison, integration design, provenance update
and verification evidence.

- [ ] **Step 4: Validate content**

Run:

```bash
grep -F '85ab2cd901aa81b70caac7711f06864d594b8ff3' docs/PROVENANCE.md
grep -F '5b6da88f5039b9124a239b16e6c7988e4183273f' docs/PROVENANCE.md
grep -F 'feat/dsv4-mini-oracle' docs/PROVENANCE.md
grep -F 'Owning repository identified' docs/SOURCE_REVIEW_CHECKLIST.md
git diff --check
```

Expected: every grep succeeds; no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add docs/PROVENANCE.md docs/SOURCE_REVIEW_CHECKLIST.md
git commit -m "docs: add source provenance and review gate"
```

### Task 3: Enforce source review in contribution and PR workflows

**Files:**
- Modify: `CONTRIBUTING.md`
- Modify: `.github/PULL_REQUEST_TEMPLATE.md`

**Interfaces:**
- Consumes: `docs/SOURCE_REVIEW_CHECKLIST.md` and `docs/PROVENANCE.md`.
- Produces: contributor-facing gate that prevents unreviewed ports from being considered complete.

- [ ] **Step 1: Add a source-porting section to CONTRIBUTING**

Insert after `## Before anything` and its build block:

```markdown
## Before porting or deriving code

Review the owning repository before implementation. Follow
[`docs/SOURCE_REVIEW_CHECKLIST.md`](docs/SOURCE_REVIEW_CHECKLIST.md) and record
all adopted designs or code in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
A branch name alone is not a source identity: record a full commit SHA, inspect
its tests and license/NOTICE, and explain local semantic differences.
```

- [ ] **Step 2: Add a Source review section to the PR template**

Insert before `## Verification`:

```markdown
## Source review, when code or design is ported/derived

- [ ] Owning repository and full source commit are listed below.
- [ ] Relevant source files, tests, branches/PRs and recent commits were reviewed.
- [ ] License and NOTICE obligations were checked and preserved.
- [ ] `docs/PROVENANCE.md` was updated with the decision and local differences.

<!-- Write "Not applicable" for original work. Otherwise list repository, SHA,
     files reviewed, decision (KEEP/PORT/REWRITE/EXPERIMENT/REJECT/DEFER), and
     semantic differences. -->
```

- [ ] **Step 3: Verify exact gate text and existing checks**

Run:

```bash
grep -F 'Before porting or deriving code' CONTRIBUTING.md
grep -F 'Owning repository and full source commit' .github/PULL_REQUEST_TEMPLATE.md
grep -F '`make test` passes' .github/PULL_REQUEST_TEMPLATE.md
git diff --check
```

Expected: all existing verification checks remain and the new source gate is present.

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md .github/PULL_REQUEST_TEMPLATE.md
git commit -m "docs: require source review for ports"
```

### Task 4: Run the non-semantic baseline gate

**Files:**
- Test only; no expected modifications.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: evidence that P0-A did not change the existing runtime.

- [ ] **Step 1: Verify repository hygiene**

```bash
git diff --check main...HEAD
git status --short
```

Expected: no whitespace errors; no untracked generated files.

- [ ] **Step 2: Build the portable baseline**

```bash
make portable -j"$(nproc)"
```

Expected: successful build with no new warnings.

- [ ] **Step 3: Run the weightless suite**

```bash
make test
```

Expected: all existing K3 weightless tests pass. This validates only that P0-A
is non-semantic; it is not evidence of DeepSeek correctness.

- [ ] **Step 4: Confirm source files were not changed**

```bash
git diff --name-only main...HEAD | \
  grep -Ev '^(README\.md|CONTRIBUTING\.md|\.github/PULL_REQUEST_TEMPLATE\.md|docs/)'
```

Expected: no output.

- [ ] **Step 5: Record final commit**

```bash
git log --oneline --decorate main..HEAD
git status --short
```

Expected: three documentation commits and a clean tree.
