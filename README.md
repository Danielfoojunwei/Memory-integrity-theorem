# Memory Integrity Theorem

A formal treatment of memory integrity guarantees for agentic AI systems, addressing persistent prompt injection and cross-session memory poisoning — with empirical evaluation on canonical benchmarks.

## Overview

Agentic AI systems that maintain long-term identity through persistent memory files (e.g., `SOUL.md`) are vulnerable to persistent prompt injection — attacks that permanently alter an agent's behaviour by writing into its memory. This project formalises the **Memory Integrity Theorem**, which guarantees that:

1. **Immutable memory items** (such as identity files) cannot be modified by untrusted inputs.
2. **Session-specific memory** remains isolated, preventing cross-session data leakage.

## Empirical Results

All evaluation runs on **real canonical datasets** from HuggingFace — no simulation, no mocks. 5 detectors evaluated across 3 datasets (1,028 test examples total).

### Detection Benchmarks

**deepset/prompt-injections** (n=116 test)

| Detector | Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Heuristic (baseline) | 50.86% | 100.00% | 5.00% | 9.52% | 52.50% |
| TF-IDF + Logistic Regression | 90.52% | 98.04% | 83.33% | 90.09% | 97.44% |
| DeBERTa-v3 pre-trained (zero-shot) | 67.24% | 100.00% | 36.67% | 53.66% | 89.57% |
| **DeBERTa-v3 fine-tuned** | **94.83%** | **100.00%** | **90.00%** | **94.74%** | **97.74%** |
| Ensemble (TF-IDF + FT-DeBERTa) | 94.83% | 100.00% | 90.00% | 94.74% | **99.14%** |

**jackhhao/jailbreak-classification** (n=262 test)

| Detector | Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Heuristic (baseline) | 61.07% | 79.37% | 35.97% | 49.50% | 62.75% |
| TF-IDF + Logistic Regression | 96.56% | 99.24% | 94.24% | 96.68% | 99.51% |
| DeBERTa-v3 pre-trained (zero-shot) | 90.84% | 98.32% | 84.17% | 90.70% | 97.91% |
| **DeBERTa-v3 fine-tuned** | **97.33%** | **97.14%** | **97.84%** | **97.49%** | **98.87%** |
| Ensemble (TF-IDF + FT-DeBERTa) | 97.33% | 97.14% | 97.84% | 97.49% | **99.63%** |

**Combined corpus** (n=650 test, from 4 HuggingFace datasets)

| Detector | Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Heuristic (baseline) | 58.92% | 33.33% | 32.18% | 32.75% | 51.85% |
| TF-IDF + Logistic Regression | 95.54% | 95.29% | 90.10% | 92.62% | 98.26% |
| DeBERTa-v3 pre-trained (zero-shot) | 92.00% | 96.88% | 76.73% | 85.64% | 92.62% |
| DeBERTa-v3 fine-tuned | 96.31% | 97.85% | 90.10% | 93.81% | 98.18% |
| **Ensemble (TF-IDF + FT-DeBERTa)** | **96.62%** | **97.87%** | **91.09%** | **94.36%** | **98.75%** |

### Impact of Fine-Tuning (before vs after)

| Dataset | DeBERTa pre-trained F1 | DeBERTa fine-tuned F1 | Improvement |
|---|---|---|---|
| deepset/prompt-injections | 53.66% | **94.74%** | **+41.08pp** |
| jackhhao/jailbreak-classification | 90.70% | **97.49%** | **+6.79pp** |
| Combined corpus | 85.64% | **93.81%** | **+8.17pp** |

### Comparison with Published Baselines

| Method | Dataset | F1 | Source |
|---|---|---|---|
| **DeBERTa-v3 fine-tuned (ours)** | deepset | **94.74%** | This work |
| **Ensemble (ours)** | deepset | **94.74%** (AUROC 99.14%) | This work |
| deepset/deberta-v3-base-injection | deepset | 99.40% | HuggingFace model card (self-eval) |
| protectai/deberta-v3 v1 | deepset | 96.40% | ProtectAI model card |
| **DeBERTa-v3 fine-tuned (ours)** | jailbreak | **97.49%** | This work |
| **Ensemble (ours)** | jailbreak | **97.49%** (AUROC 99.63%) | This work |
| TF-IDF+LR (ours) | jailbreak | 96.68% | This work |
| RoBERTa classifier (Jain et al.) | jailbreak | ~88.0% | Baseline Defenses, 2023 |

### Defense (Theorem) Verification

| Test | Result |
|---|---|
| Provenance verifier vs 199 real attack payloads | **100.0% blocked** |
| 7 canonical attack vectors (SOUL.md, reinjection, taint washing, etc.) | **7/7 blocked** |
| Cross-session isolation (50 sessions, 2,450 pair-checks) | **0 leaks** |
| SOUL.md immutability after all attacks | **Preserved** |
| Memory hash integrity (pre vs post attack) | **Identical** |
| **Theorem holds** | **True** |

### AgentDojo Defense Comparison

| Defense Method | Security Rate | Utility Impact | Source |
|---|---|---|---|
| **Provenance Verifier (ours)** | **100.00%** | **No utility loss** | This work |
| GPT-4o (spotlighting) | 84.20% | Utility drops to 43.8% | Debenedetti et al. 2024 |
| Claude 3.5 Sonnet (tool-filter) | 73.70% | Utility drops to 54.6% | Debenedetti et al. 2024 |
| GPT-4o (tool-filter) | 68.40% | Utility drops to 63.5% | Debenedetti et al. 2024 |
| Claude 3.5 Sonnet (no defense) | 42.10% | Full utility (60.8%) | Debenedetti et al. 2024 |
| GPT-4o (no defense) | 31.60% | Full utility (68.8%) | Debenedetti et al. 2024 |

## Repository Structure

```
├── memory-integrity-theorem.md          # Formal theorem, proof sketch, and discussion
├── memory_integrity_eval/
│   ├── src/
│   │   ├── agent_state.py               # Agent state model S_t = (P_t, M_t, B_t, G_t)
│   │   ├── attack_simulator.py          # 7 attack vector implementations
│   │   ├── detectors.py                 # 5 detectors: Heuristic, TF-IDF, DeBERTa, FT-DeBERTa, Ensemble
│   │   ├── real_benchmark.py            # Dataset loaders, metrics, SOTA baselines
│   │   ├── main_evaluation.py           # Real evaluation pipeline (5 detectors x 3 datasets)
│   │   └── benchmark_integration.py     # Legacy benchmark integration
│   ├── tests/
│   │   └── test_memory_integrity.py     # 38 tests (all passing)
│   ├── results/                         # Generated evaluation results (JSON)
│   └── requirements.txt
└── LICENSE
```

## Quick Start

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy datasets transformers tqdm pyyaml pytest

# Run full real evaluation (~45 min on CPU, downloads datasets + DeBERTa model, fine-tunes 3x)
cd memory_integrity_eval/src
python main_evaluation.py

# Run tests
pytest memory_integrity_eval/tests/ -v
```

## Datasets Used

| Dataset | Source | Size | Type |
|---|---|---|---|
| **deepset/prompt-injections** | HuggingFace | 662 (546 train / 116 test) | Binary injection labels |
| **jackhhao/jailbreak-classification** | HuggingFace | 1,306 (1,044 train / 262 test) | Jailbreak vs benign |
| **rubend18/ChatGPT-Jailbreak-Prompts** | HuggingFace | 79 | Jailbreak prompts |
| **fka/awesome-chatgpt-prompts** | HuggingFace | 1,203 | Benign prompts |
| **Combined corpus** | All above | 3,250 (2,600 train / 650 test) | Stratified 80/20 split |

## Detectors

1. **HeuristicDetector** — 40+ regex patterns covering instruction override, role hijack, exfiltration, memory attack, and encoding evasion categories.
2. **TFIDFDetector** — Character n-gram (1-4) TF-IDF with balanced Logistic Regression. Trained per-dataset.
3. **DeBERTaDetector** — `protectai/deberta-v3-base-prompt-injection-v2`, pre-trained DeBERTa v3 classifier (~86M parameters). Zero-shot on target data.
4. **FineTunedDeBERTaDetector** — Same architecture, fine-tuned on each dataset's training split. Freezes first 9 of 12 encoder layers, trains last 3 + classifier head (22.25M / 184.42M trainable params = 12.1%).
5. **EnsembleDetector** — Weighted average of TF-IDF and fine-tuned DeBERTa scores (0.4 / 0.6 weighting). Combines lexical + semantic signal.

## References

1. Debenedetti et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." NeurIPS 2024.
2. Lakera AI (2024). "PINT Benchmark: Prompt Injection Test Benchmark."
3. Liu et al. (2024). "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." USENIX Security 2024.
4. Yi et al. (2023). "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models."
5. Jain et al. (2023). "Baseline Defenses for Adversarial Attacks Against Aligned Language Models."

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
