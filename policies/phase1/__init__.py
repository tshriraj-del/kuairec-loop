"""Phase 1 base recommender: public-data scorers behind one rule-aware slate wrapper.

Policy-side code: it reads only the big matrix, item and user features and captions (through
kuairec_eval.data and kuairec_eval.labels), never an evaluator-side module. The isolation
tests in tests/test_policy_isolation.py scan this package like every other policy file.
"""
