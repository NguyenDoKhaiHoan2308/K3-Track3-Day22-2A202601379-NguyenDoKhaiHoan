# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24`
- **Validation issues found**: The original first JSONL row contained unescaped quotes.
- **Cleaning steps taken**: Repaired that prompt and added line-numbered schema/JSON errors,
  normalized duplicate-prompt detection, near-duplicate response checks, and optional PII checks.

### Split Strategy
- **Train/Val Ratio**: `80/20`
- **Leakage Prevention**: Prompts are normalized, grouped, and seeded before whole groups are split.

## 2. Implementation: DPO

### Objective Selection
- **Why this method?**: DPO directly optimizes pairwise preferences against a reference margin
  and fits the small, offline dataset used in this lab.
- **Key Hyperparameters**:
    - `beta`: `0.1`
    - `lambda_orpo` (also implemented for comparison): `0.1`

### Numerical Stability
- **Challenges**: Extreme preference margins can overflow a direct `-log(sigmoid(x))`.
- **Solutions**: DPO uses `logaddexp(0, -x)`; ORPO uses stable `log1p`/`expm1` branches.

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | `100.00%` |
| Final Loss (CPU DPO smoke run) | `0.6734` |

### Qualitative Review
- **Prompt**: Explain the concept of self-attention in Transformers.
- **Chosen Response**: Describes weighting words and capturing long-range dependencies.
- **Rejected Response**: Incorrectly characterizes self-attention as a simpler RNN.
- **Baseline Preference**: Correct for this pair.

## 4. Discussion & Failure Modes

- **What went well?**: Schema validation, stable objectives, deterministic splitting, and JSON
  metric output are independently testable on CPU.
- **Observed Bias**: The deterministic evaluator rewards informative length and lexical diversity,
  so verbosity can be mistaken for quality. It is a pipeline baseline, not a reward model.
- **Safety**: Regression prompts remain documented for before/after model checks. The CPU scorer
  does not generate responses, so generative safety requires a trained-model evaluation.
