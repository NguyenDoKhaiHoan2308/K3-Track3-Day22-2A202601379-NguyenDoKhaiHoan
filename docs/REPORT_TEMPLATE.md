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

## 5. Deep-Dive Checks

### What happens without a reference preference margin?

Setting `ref_chosen_logps == ref_rejected_logps` makes the reference margin zero. The DPO loss
still produces a finite value and optimizes the policy's chosen-versus-rejected margin. However,
this removes the anchor to the reference policy, so the policy has less protection against drifting
away from its original behavior. The reference model is therefore not required for the arithmetic
to run, but it is important for controlling the optimization target.

### Is evaluation reproducible?

Yes. The CPU evaluator contains no random sampling: it tokenizes each response deterministically
and scores informative length plus lexical diversity. Two consecutive evaluations of the same
dataset produce the same `pairwise_accuracy`. Dataset splitting also uses the configured seed
(`42`) through a local random-number generator.

### Is the scorer biased toward longer answers?

Yes. The `log1p(word_count)` term generally rewards longer responses, even though lexical diversity
partly moderates repetition. The current `pairwise_accuracy = 1.0` therefore demonstrates that the
evaluation pipeline works on this dataset; it does not prove that response quality is measured
reliably. A production evaluation should replace this baseline with model log-probabilities, a
trained reward model, or human judgments and should report results by response-length bucket.
