# TEST_STRATEGY.md — Screenalytics

Version: 1.0

## Scope
Unit, integration, and pipeline smoke tests on a short sample clip.

## Minimum Bars
- Unit coverage ≥ 85% for promoted modules.
- Integration tests per stage: detection, tracking, identity, fusion, aggregation.
- CI “smoke” run processes sample episode end-to-end in ≤ 15 min.

## Suites
- tests/api/ — route contracts (FastAPI)
- tests/workers/ — stage units and adapters
- tests/pipelines/ — end-to-end manifest checks
- tests/fixtures/ — sample frames/audio & expected outputs
