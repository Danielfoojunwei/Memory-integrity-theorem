# Memory Integrity Theorem

A formal treatment of memory integrity guarantees for agentic AI systems, addressing persistent prompt injection and cross-session memory poisoning — with an empirical evaluation framework.

## Overview

Agentic AI systems that maintain long-term identity through persistent memory files (e.g., `SOUL.md`) are vulnerable to persistent prompt injection — attacks that permanently alter an agent's behaviour by writing into its memory. This project formalises the **Memory Integrity Theorem**, which provides guarantees that:

1. **Immutable memory items** (such as identity files) cannot be modified by untrusted inputs.
2. **Session-specific memory** remains isolated, preventing cross-session data leakage.

## Repository Structure

```
├── memory-integrity-theorem.md          # Formal theorem, proof sketch, and discussion
├── memory_integrity_eval/               # Empirical evaluation framework
│   ├── src/
│   │   ├── agent_state.py               # Agent state model S_t = (P_t, M_t, B_t, G_t)
│   │   ├── attack_simulator.py          # 7 attack scenario implementations
│   │   ├── benchmark_integration.py     # PINT & AgentDojo integration
│   │   └── main_evaluation.py           # Main evaluation runner
│   ├── tests/
│   │   └── test_memory_integrity.py     # Comprehensive test suite
│   ├── benchmarks/                      # Benchmark datasets
│   ├── data/                            # Input data
│   ├── results/                         # Evaluation results
│   ├── docs/                            # Documentation
│   └── requirements.txt                 # Python dependencies
└── LICENSE                              # Apache License 2.0
```

## Key Concepts

- **Memory update rules**: Write proposals, immutability by default, cross-session isolation, and verification.
- **Provenance tracking**: Each memory item carries a provenance principal and taint bit to distinguish trusted from untrusted sources.
- **Taint propagation**: Untrusted inputs are automatically tainted and cannot modify memory without trusted provenance.
- **Session namespace isolation**: Each session has its own logical memory namespace; promotion to shared memory requires explicit trusted action.

## Evaluation Framework

The empirical framework validates the theorem using real benchmarks:

| Benchmark | Source | Scale |
|-----------|--------|-------|
| **Lakera PINT** | Prompt Injection Test Benchmark | 4,314 test cases, 24+ languages |
| **AgentDojo** | NeurIPS 2024 | 97 tasks, 629 security test cases |
| **Attack Simulation** | Custom | 7 attack vectors |

### Attack Vectors Tested

1. SOUL.md direct modification (persistent injection)
2. Hidden instruction injection (indirect injection)
3. Scheduled reinjection (cron-style persistent backdoor)
4. Tool output poisoning (malicious tool responses)
5. Cross-session data leak (session isolation breach)
6. Dependency chain poisoning (taint propagation)
7. Taint washing (privilege escalation)

### Quick Start

```bash
# Install dependencies
pip install -r memory_integrity_eval/requirements.txt

# Run the full evaluation
cd memory_integrity_eval/src
python main_evaluation.py

# Run the test suite
pytest memory_integrity_eval/tests/ -v
```

### Overall Security Score

Weighted combination of benchmark results:
- PINT F1 Score (30%)
- AgentDojo Block Rate (30%)
- Attack Simulation Block Rate (40%)

## References

1. Debenedetti et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." NeurIPS 2024.
2. Lakera AI (2024). "PINT Benchmark: Prompt Injection Test Benchmark."
3. Liu et al. (2024). "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." USENIX Security 2024.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
