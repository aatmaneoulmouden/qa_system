# RAG Chatbot Test Report

**Run date:** 2026-10-01  
**Command:** `.venv/bin/pytest -v -rs`  
**Environment:** Python 3.14.7, pytest

## Result

| Result | Count |
| --- | ---: |
| Passed | 13 |
| Failed | 0 |
| Skipped | 1 |
| Total | 14 |

## Coverage

- Non-empty chatbot response: all 12 test cases passed, including supported questions, paraphrases, ambiguous input, unsupported input, retrieval failure, blank input, and the adversarial prompt.
- Conservative unsupported response: passed for `unsupported_01`.
- Supported-answer keyword assertions: skipped; no cases matched the test selector.

## Limitation

The supported-answer test in `tests/test_rag_system.py` selects cases where `type` is `supported`, but the current `test_cases.json` labels those cases `supported_factual`. Pytest therefore reported `got empty parameter set for (case)`. The run did not verify expected factual keywords against chatbot answers. Align the test selector and case labels before treating supported-answer accuracy as covered.

## Summary

No tests failed. The chatbot returned non-empty responses for every test case and handled the unsupported salary question conservatively. Supported factual correctness remains unverified by this run because its keyword-assertion test was skipped.