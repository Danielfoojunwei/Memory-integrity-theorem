# Memory Integrity Theorem

A formal treatment of memory integrity guarantees for agentic AI systems, addressing persistent prompt injection and cross-session memory poisoning.

## Overview

Agentic AI systems that maintain long-term identity through persistent memory files (e.g., `SOUL.md`) are vulnerable to persistent prompt injection — attacks that permanently alter an agent's behaviour by writing into its memory. This project formalises the **Memory Integrity Theorem**, which provides guarantees that:

1. **Immutable memory items** (such as identity files) cannot be modified by untrusted inputs.
2. **Session-specific memory** remains isolated, preventing cross-session data leakage.

## Contents

- [`memory-integrity-theorem.md`](memory-integrity-theorem.md) — Full theorem statement, proof sketch, discussion, and evaluation plan.
- [`LICENSE`](LICENSE) — Apache License 2.0.

## Key Concepts

- **Memory update rules**: Write proposals, immutability by default, cross-session isolation, and verification.
- **Provenance tracking**: Each memory item carries a provenance principal and taint bit to distinguish trusted from untrusted sources.
- **Evaluation plan**: Concrete experiments including self-modification attempts, scheduled reinjection attacks, cross-session leak tests, false positive checks, and audit logging.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
