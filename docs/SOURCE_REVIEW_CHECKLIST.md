# Source Review Checklist

Use this checklist before writing or integrating code based on another
repository. Complete it in the PR description and update `docs/PROVENANCE.md`.

## 1. Resolve the source

- [ ] Owning repository identified.
- [ ] Full source commit SHA recorded.
- [ ] Current branches and relevant pull requests inspected.
- [ ] Recent commits touching the candidate files inspected.
- [ ] Source repository contribution instructions read.

## 2. Legal and attribution

- [ ] License read at the recorded ref.
- [ ] NOTICE and third-party notices read.
- [ ] Copied/substantially derived files identified.
- [ ] Required copyright, permission and SPDX text preserved.
- [ ] Model/data license considered separately from source-code license.

## 3. Technical review

- [ ] Candidate implementation files read in full.
- [ ] Corresponding tests and fixtures read.
- [ ] Design documents and known limitations read.
- [ ] Target implementation and invariants compared.
- [ ] Conflicting assumptions, dimensions, formats and lifecycle rules listed.
- [ ] Decision recorded as KEEP, PORT, REWRITE, EXPERIMENT, REJECT or DEFER.

## 4. Integration design

- [ ] Public interfaces and ownership boundaries named.
- [ ] Source behavior that must remain unchanged stated explicitly.
- [ ] Source behavior intentionally changed stated explicitly.
- [ ] Error handling and partial-failure behavior specified.
- [ ] Test oracle and parity gates specified before implementation.
- [ ] Hardware-only claims marked `HARDWARE_PENDING` until measured.

## 5. Completion evidence

- [ ] Provenance register updated.
- [ ] Weightless tests pass.
- [ ] Source-derived tests or equivalent adversarial fixtures pass.
- [ ] Fallback/reference parity passes.
- [ ] Performance claims include raw data and repeated runs.
- [ ] Documentation names the exact supported model/profile and limitations.
