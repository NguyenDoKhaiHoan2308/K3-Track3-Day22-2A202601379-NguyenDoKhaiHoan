# Data Card: Sample Preference Pairs

- Dataset name: `sample_preferences`
- Source: Synthetic educational examples included with this lab.
- License/permission: Course material; use is limited to the repository's lab purpose.
- Schema: JSONL rows with `prompt`, `chosen`, `rejected`, and optional `metadata`.
- Labeling rubric: Factual accuracy and explanatory usefulness.
- Known biases: English-only machine-learning questions; chosen answers are usually longer.
- Safety/PII checks: Optional email/phone-pattern rejection via `load_jsonl(check_pii=True)`.
- Train/validation/test split method: Seeded prompt-group split to prevent prompt leakage.
