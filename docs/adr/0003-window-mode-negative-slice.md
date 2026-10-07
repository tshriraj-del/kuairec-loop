# ADR-0003: Default to the intended window; keep the EasyRL4Rec slice as a mode

Status: accepted (Phase 0)

## Context
EasyRL4Rec builds the window as `self.sequence_action[t - N:t]` on a list of length t. For 0 < t < N the start index is negative, and Python counts it from the end. For N = 9 the window at t = 5..8 is `seq[2t-9:t]`, so t=5 sees items 1-4 (item 0 dropped) and t=8 sees only item 7. Items that should be in the window are skipped, so the reference undercounts and quits later.

## Decision
- `window_mode = "intended"` is the default: the window is the last `min(t, N)` recommendations.
- `window_mode = "easyrl4rec_slice"` reproduces the reference exactly and exists for the differential test and for any comparison with published EasyRL4Rec numbers.

## Evidence
`.venv/bin/python scripts/differential.py` on 847 search users and 5,929 sequences: in intended mode, 1,348 sequences quit at a different step than the reference, all at t ∈ {5,6,7,8}, and ours quits earlier in all 1,348. In slice mode there are 0 quit-step differences.

## Consequences
Our session lengths are shorter than an EasyRL4Rec KuaiEnv run with the same parameters would report. Choosing compatibility over correctness is your call (OPEN_QUESTIONS Q2).
