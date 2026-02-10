# Memory Integrity Theorem

A formal treatment of memory integrity guarantees for agentic AI systems, addressing persistent prompt injection and cross-session memory poisoning — with empirical evaluation on canonical benchmarks.

## Overview

Agentic AI systems that maintain long-term identity through persistent memory files (e.g., `SOUL.md`) are vulnerable to persistent prompt injection — attacks that permanently alter an agent's behaviour by writing into its memory. This project formalises the **Memory Integrity Theorem**, which guarantees that:

1. **Immutable memory items** (such as identity files) cannot be modified by untrusted inputs.
2. **Session-specific memory** remains isolated, preventing cross-session data leakage.

## Empirical Results

All evaluation runs on **real canonical datasets** from HuggingFace — no simulation, no mocks.

### Detection Benchmarks

| Detector | deepset/prompt-injections (n=116) | jackhhao/jailbreak-classification (n=262) | Combined Corpus (n=650) |
|---|---|---|---|
| | Acc / F1 / AUROC | Acc / F1 / AUROC | Acc / F1 / AUROC |
| **Heuristic (baseline)** | 50.86% / 9.52% / 52.50% | 61.07% / 49.50% / 62.75% | 58.92% / 32.75% / 51.85% |
| **TF-IDF + Logistic Regression** | 90.52% / 90.09% / 97.44% | 96.56% / 96.68% / 99.51% | 95.54% / 92.62% / 98.26% |
| **DeBERTa-v3-prompt-injection-v2** | 67.24% / 53.66% / 89.57% | 90.84% / 90.70% / 97.91% | 92.00% / 85.64% / 92.62% |

### Comparison with Published Baselines

| Method | Dataset | F1 | Source |
|---|---|---|---|
| TF-IDF+LR (ours) | deepset/prompt-injections | 90.09% | This work |
| deepset/deberta-v3-base-injection | deepset/prompt-injections | 99.40% | HuggingFace model card |
| TF-IDF+LR (ours) | jackhhao/jailbreak-classification | 96.68% | This work |
| DeBERTa-v3-injection-v2 (ours) | jackhhao/jailbreak-classification | 90.70% | This work |
| RoBERTa classifier (Jain et al.) | jailbreak-classification | ~88.0% | Baseline Defenses, 2023 |

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

| Defense Method | Security Rate | Source |
|---|---|---|
| **Provenance Verifier (ours)** | **100.00%** | This work |
| GPT-4o (spotlighting) | 84.20% | Debenedetti et al. 2024 |
| GPT-4o (tool-filter) | 68.40% | Debenedetti et al. 2024 |
| Claude 3.5 Sonnet (tool-filter) | 73.70% | Debenedetti et al. 2024 |
| Claude 3.5 Sonnet (no defense) | 42.10% | Debenedetti et al. 2024 |
| GPT-4o (no defense) | 31.60% | Debenedetti et al. 2024 |

## Repository Structure

```
├── memory-integrity-theorem.md          # Formal theorem, proof sketch, and discussion
├── memory_integrity_eval/
│   ├── src/
│   │   ├── agent_state.py               # Agent state model S_t = (P_t, M_t, B_t, G_t)
│   │   ├── attack_simulator.py          # 7 attack vector implementations
│   │   ├── detectors.py                 # Heuristic, TF-IDF+LR, DeBERTa detectors
│   │   ├── real_benchmark.py            # Dataset loaders, metrics, SOTA baselines
│   │   ├── main_evaluation.py           # Real evaluation pipeline
│   │   └── benchmark_integration.py     # Legacy benchmark integration
│   ├── tests/
│   │   └── test_memory_integrity.py     # 32 tests (all passing)
│   ├── results/                         # Generated evaluation results (JSON)
│   └── requirements.txt
└── LICENSE
```

## Quick Start

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy datasets transformers tqdm pyyaml pytest

# Run full real evaluation (~10 min, downloads datasets + DeBERTa model)
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
3. **DeBERTaDetector** — `protectai/deberta-v3-base-prompt-injection-v2`, a fine-tuned DeBERTa v3 classifier (~86M parameters).

## References

1. Debenedetti et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." NeurIPS 2024.
2. Lakera AI (2024). "PINT Benchmark: Prompt Injection Test Benchmark."
3. Liu et al. (2024). "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." USENIX Security 2024.
4. Yi et al. (2023). "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models."
5. Jain et al. (2023). "Baseline Defenses for Adversarial Attacks Against Aligned Language Models."

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
