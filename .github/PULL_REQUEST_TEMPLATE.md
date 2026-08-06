## What this changes

<!-- One or two sentences. What behaviour is different after this PR? -->

## Why

<!-- The problem being solved. Link an issue if there is one. -->

## Source review, when code or design is ported/derived

- [ ] Owning repository and full source commit are listed below.
- [ ] Relevant source files, tests, branches/PRs and recent commits were reviewed.
- [ ] License and NOTICE obligations were checked and preserved.
- [ ] `docs/PROVENANCE.md` was updated with the decision and local differences.

<!-- Write "Not applicable" for original work. Otherwise list repository, SHA,
     files reviewed, decision (KEEP/PORT/REWRITE/EXPERIMENT/REJECT/DEFER), and
     semantic differences. -->

## Verification

- [ ] `make test` passes (all weightless gates)
- [ ] `make portable` builds with no new warnings
- [ ] If kernels changed: `./bin/test_ops tests/fixtures/ops` still passes at declared tolerance
- [ ] If the config or tokenizer path changed: `make cfg` and `make tok` pass
- [ ] If output could change: the oracle gates (`./bin/k3_model tests/fixtures`) still match exactly

## Numbers, if this is a performance change

<!-- Timing claims need a noise floor. Report at least 3 runs of each arm, or say
     explicitly that the effect was not measured against variance. See docs/BENCHMARKING.md. -->

| arm | run 1 | run 2 | run 3 | mean |
|-----|-------|-------|-------|------|
|     |       |       |       |      |

## Risk

<!-- What could this break that the tests would not catch? -->
