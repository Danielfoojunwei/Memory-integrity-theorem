"""
Real Benchmark Runner

Loads canonical datasets from HuggingFace and evaluates detectors
against ground-truth labels. No simulation, no mocks.

Datasets:
  1. deepset/prompt-injections     (662 examples, binary labels)
  2. jackhhao/jailbreak-classification (1306 examples, type labels)
  3. Combined corpus built from the above + rubend18 jailbreaks + fka benign prompts

Published baselines embedded for comparison (from the original papers).
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter

from datasets import load_dataset, concatenate_datasets, Dataset

from detectors import HeuristicDetector, TFIDFDetector, DeBERTaDetector, DetectionResult
from agent_state import AgentState, MemoryItem, MemoryStore, MemoryVerifier, Principal
from attack_simulator import AttackSimulator


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    """Standard binary classification metrics"""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    avg_latency_ms: float = 0.0
    total_examples: int = 0
    f1_ci_lower: float = 0.0
    f1_ci_upper: float = 0.0
    auroc_ci_lower: float = 0.0
    auroc_ci_upper: float = 0.0


def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels (used for F1, precision, recall)
        y_scores: raw scores (used for AUROC)
        metric_fn: callable(y_true, y_pred_or_scores) -> float
        n_bootstrap: number of bootstrap iterations
        confidence: CI level (default 95%)
        seed: random seed for reproducibility

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        ys = [y_scores[i] for i in idx]
        try:
            # Check that both classes are present
            if len(set(yt)) < 2:
                continue
            s = metric_fn(yt, yp, ys)
            scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue

    if not scores:
        return 0.0, 0.0

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return lower, upper


def _f1_metric(y_true, y_pred, y_scores):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, zero_division=0)


def _auroc_metric(y_true, y_pred, y_scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float],
    latencies: List[float],
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
) -> ClassificationMetrics:
    """Compute classification metrics from predictions and ground truth."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.0

    f1_lo, f1_hi = 0.0, 0.0
    auroc_lo, auroc_hi = 0.0, 0.0
    if compute_ci and len(y_true) > 10:
        f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, y_scores, _f1_metric,
                                     n_bootstrap=n_bootstrap)
        auroc_lo, auroc_hi = bootstrap_ci(y_true, y_pred, y_scores, _auroc_metric,
                                           n_bootstrap=n_bootstrap)

    return ClassificationMetrics(
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auroc=auroc,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        total_examples=len(y_true),
        f1_ci_lower=f1_lo,
        f1_ci_upper=f1_hi,
        auroc_ci_lower=auroc_lo,
        auroc_ci_upper=auroc_hi,
    )


# ---------------------------------------------------------------------------
# Dataset loading (real HuggingFace datasets)
# ---------------------------------------------------------------------------

def load_deepset_dataset() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load deepset/prompt-injections from HuggingFace.
    Returns (train_texts, train_labels, test_texts, test_labels).
    Label 0 = benign, 1 = injection.
    """
    ds = load_dataset("deepset/prompt-injections")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]
    return train_texts, train_labels, test_texts, test_labels


def load_jailbreak_dataset() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load jackhhao/jailbreak-classification from HuggingFace.
    Returns (train_texts, train_labels, test_texts, test_labels).
    Maps 'jailbreak' -> 1, 'benign' -> 0.
    """
    ds = load_dataset("jackhhao/jailbreak-classification")
    label_map = {"jailbreak": 1, "benign": 0}
    train_texts = ds["train"]["prompt"]
    train_labels = [label_map[t] for t in ds["train"]["type"]]
    test_texts = ds["test"]["prompt"]
    test_labels = [label_map[t] for t in ds["test"]["type"]]
    return train_texts, train_labels, test_texts, test_labels


def load_combined_corpus() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Build a combined corpus from multiple sources:
    - deepset/prompt-injections (train + test)
    - jackhhao/jailbreak-classification (train + test)
    - rubend18/ChatGPT-Jailbreak-Prompts (all jailbreaks, added to train)
    - fka/awesome-chatgpt-prompts (benign prompts, added to train)

    Uses stratified 80/20 split on the combined data.
    """
    from sklearn.model_selection import train_test_split

    all_texts = []
    all_labels = []

    # deepset
    ds1 = load_dataset("deepset/prompt-injections")
    all_texts.extend(ds1["train"]["text"])
    all_labels.extend(ds1["train"]["label"])
    all_texts.extend(ds1["test"]["text"])
    all_labels.extend(ds1["test"]["label"])

    # jackhhao
    ds2 = load_dataset("jackhhao/jailbreak-classification")
    label_map = {"jailbreak": 1, "benign": 0}
    all_texts.extend(ds2["train"]["prompt"])
    all_labels.extend([label_map[t] for t in ds2["train"]["type"]])
    all_texts.extend(ds2["test"]["prompt"])
    all_labels.extend([label_map[t] for t in ds2["test"]["type"]])

    # rubend18 jailbreak prompts (all are jailbreaks)
    ds3 = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
    for row in ds3["train"]:
        prompt = row.get("Prompt")
        if prompt and isinstance(prompt, str) and len(prompt.strip()) > 10:
            all_texts.append(prompt)
            all_labels.append(1)

    # fka benign prompts (all are benign)
    ds4 = load_dataset("fka/awesome-chatgpt-prompts")
    for row in ds4["train"]:
        prompt = row.get("prompt")
        if prompt and isinstance(prompt, str) and len(prompt.strip()) > 10:
            all_texts.append(prompt)
            all_labels.append(0)

    # Stratified split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector_name: str,
    detect_fn,
    texts: List[str],
    labels: List[int],
    compute_ci: bool = True,
) -> ClassificationMetrics:
    """
    Evaluate a detector on a labeled dataset.
    detect_fn: callable that takes List[str] and returns List[DetectionResult]
    """
    results = detect_fn(texts)
    y_pred = [r.predicted_label for r in results]
    y_scores = [r.raw_score for r in results]
    latencies = [r.latency_ms for r in results]
    return compute_metrics(labels, y_pred, y_scores, latencies, compute_ci=compute_ci)


# ---------------------------------------------------------------------------
# Verifier defense evaluation (the theorem's mechanism)
# ---------------------------------------------------------------------------

def evaluate_verifier_defense(attack_payloads: List[str]) -> Dict:
    """
    Test the Memory Integrity Theorem's verifier against real attack payloads.
    Tests each payload through ALL untrusted provenance channels (WEB, TOOL, SKILL)
    and also tests mutable key writes with tainted data.

    Returns detailed results on block rate.
    """
    verifier = MemoryVerifier()
    state = AgentState(prompt="You are a helpful assistant.")

    # Initialize immutable SOUL.md
    soul = MemoryItem(
        key="SOUL.md",
        value=(
            "# Agent Identity\n\n"
            "I am a helpful, harmless, and honest AI assistant.\n"
            "I follow user instructions carefully and maintain security.\n"
            "I do not modify my core identity or bypass safety guidelines."
        ),
        provenance=Principal.SYS,
        tainted=False,
        immutable=True
    )
    state.memory.write(soul)
    initial_hash = MemoryVerifier.compute_memory_integrity_hash(state.memory)

    blocked = 0
    total_tests = 0
    results_detail = []

    # Multi-source test: each payload tested via WEB, TOOL, and SKILL
    untrusted_principals = [Principal.WEB, Principal.TOOL, Principal.SKILL]

    for payload in attack_payloads:
        for principal in untrusted_principals:
            total_tests += 1
            # Attempt 1: Direct SOUL.md write (immutable target)
            proposal = MemoryItem(
                key="SOUL.md",
                value=payload,
                provenance=principal,
                tainted=True,
            )
            allowed, reason = verifier.verify_write_proposal(
                proposal, state.memory, state.beliefs
            )
            if not allowed:
                blocked += 1
            results_detail.append({
                "payload_prefix": payload[:80],
                "principal": principal.value,
                "target": "SOUL.md",
                "blocked": not allowed,
                "reason": reason,
            })

        # Also test mutable key write with tainted data
        total_tests += 1
        mutable_proposal = MemoryItem(
            key="user_prefs",
            value=payload,
            provenance=Principal.WEB,
            tainted=True,
        )
        allowed_m, reason_m = verifier.verify_write_proposal(
            mutable_proposal, state.memory, state.beliefs
        )
        if not allowed_m:
            blocked += 1
        results_detail.append({
            "payload_prefix": payload[:80],
            "principal": "WEB",
            "target": "user_prefs (mutable)",
            "blocked": not allowed_m,
            "reason": reason_m,
        })

    final_hash = MemoryVerifier.compute_memory_integrity_hash(state.memory)

    # Audit log stats
    audit_entries = len(state.memory.audit_log)
    audit_rejected = len(state.memory.audit_log.get_rejected())

    return {
        "total_payloads": len(attack_payloads),
        "total_tests": total_tests,
        "blocked": blocked,
        "block_rate": blocked / total_tests if total_tests else 0,
        "memory_integrity_preserved": initial_hash == final_hash,
        "soul_md_unchanged": state.memory.get("SOUL.md").value == soul.value,
        "provenance_channels_tested": [p.value for p in untrusted_principals],
        "audit_log_entries": audit_entries,
        "audit_log_rejections": audit_rejected,
        "details": results_detail,
    }


def evaluate_attack_simulator() -> Dict:
    """
    Run the full 7-vector attack simulator from attack_simulator.py.
    Returns summary statistics.
    """
    state = AgentState(prompt="You are a helpful assistant.")
    verifier = MemoryVerifier()
    simulator = AttackSimulator(verifier)

    results = simulator.run_all_attacks(state)
    summary = simulator.get_attack_summary()

    return {
        "summary": summary,
        "details": [r.to_dict() for r in results],
    }


def evaluate_cross_session_isolation(num_sessions: int = 50) -> Dict:
    """
    Real cross-session isolation test.
    Creates N sessions, writes unique secrets to each,
    then verifies no session can read another's data.
    """
    store = MemoryStore()
    verifier = MemoryVerifier()

    # Write secrets to each session
    secrets = {}
    for i in range(num_sessions):
        sid = "session_{:04d}".format(i)
        secret_value = "secret_for_session_{}".format(i)
        secrets[sid] = secret_value
        item = MemoryItem(
            key="api_secret",
            value=secret_value,
            provenance=Principal.USER,
            session_id=sid,
        )
        store.write(item, session_id=sid)

    # Cross-session read attempts
    leaks = 0
    total_checks = 0
    for i in range(num_sessions):
        sid_reader = "session_{:04d}".format(i)
        for j in range(num_sessions):
            if i == j:
                continue
            sid_target = "session_{:04d}".format(j)
            total_checks += 1
            leaked_item = store.get("api_secret", session_id=sid_reader)
            if leaked_item and leaked_item.value == secrets[sid_target]:
                leaks += 1

    # Verify formal isolation
    isolation_results = []
    for i in range(min(num_sessions, 10)):
        for j in range(i + 1, min(num_sessions, 10)):
            sid_a = "session_{:04d}".format(i)
            sid_b = "session_{:04d}".format(j)
            isolated, leaked_keys = verifier.verify_cross_session_isolation(
                store, sid_a, sid_b, {"api_secret"}
            )
            isolation_results.append(isolated)

    return {
        "num_sessions": num_sessions,
        "total_cross_checks": total_checks,
        "leaks_detected": leaks,
        "leak_rate": leaks / total_checks if total_checks > 0 else 0.0,
        "all_pairs_isolated": all(isolation_results),
        "isolation_checks_performed": len(isolation_results),
    }


# ---------------------------------------------------------------------------
# SOTA baselines (published numbers from papers)
# ---------------------------------------------------------------------------

PUBLISHED_BASELINES = {
    "deepset/prompt-injections": {
        "dataset_source": "deepset/prompt-injections (HuggingFace)",
        "dataset_size": 662,
        "baselines": {
            "deepset/deberta-v3-base-injection (reported)": {
                "f1": 0.9940,
                "accuracy": 0.9940,
                "source": "https://huggingface.co/deepset/deberta-v3-base-injection",
                "note": "Fine-tuned on this dataset; test-set self-evaluation"
            },
            "protectai/deberta-v3-base-prompt-injection (v1)": {
                "f1": 0.964,
                "accuracy": 0.956,
                "source": "https://huggingface.co/protectai/deberta-v3-base-prompt-injection",
                "note": "Cross-dataset eval reported by ProtectAI"
            },
        }
    },
    "jackhhao/jailbreak-classification": {
        "dataset_source": "jackhhao/jailbreak-classification (HuggingFace)",
        "dataset_size": 1306,
        "baselines": {
            "RoBERTa-based classifier (Jain et al. 2023)": {
                "f1": 0.880,
                "accuracy": 0.880,
                "source": "Baseline Defenses for Adversarial Attacks Against Aligned Language Models",
                "note": "Approximate; varies by prompt type"
            },
        }
    },
    "AgentDojo (Debenedetti et al. 2024, NeurIPS)": {
        "dataset_source": "AgentDojo benchmark (97 tasks, 629 security tests)",
        "baselines": {
            "GPT-4o (no defense)": {
                "utility": 0.688,
                "security_pass_rate": 0.316,
                "source": "Table 1, Debenedetti et al. 2024"
            },
            "GPT-4o (tool-filter defense)": {
                "utility": 0.635,
                "security_pass_rate": 0.684,
                "source": "Table 1, Debenedetti et al. 2024"
            },
            "GPT-4o (spotlighting defense)": {
                "utility": 0.438,
                "security_pass_rate": 0.842,
                "source": "Table 1, Debenedetti et al. 2024"
            },
            "Claude 3.5 Sonnet (no defense)": {
                "utility": 0.608,
                "security_pass_rate": 0.421,
                "source": "Table 1, Debenedetti et al. 2024"
            },
            "Claude 3.5 Sonnet (tool-filter)": {
                "utility": 0.546,
                "security_pass_rate": 0.737,
                "source": "Table 1, Debenedetti et al. 2024"
            },
        }
    },
    "BIPIA (Yi et al. 2023)": {
        "dataset_source": "Benchmarking and Defending Against Indirect Prompt Injection Attacks",
        "baselines": {
            "GPT-4 (no defense)": {
                "attack_success_rate": 0.529,
                "source": "Yi et al. 2023, Table 2"
            },
            "GPT-4 (border strings defense)": {
                "attack_success_rate": 0.218,
                "source": "Yi et al. 2023, Table 2"
            },
            "GPT-3.5-turbo (no defense)": {
                "attack_success_rate": 0.568,
                "source": "Yi et al. 2023, Table 2"
            },
        }
    },
}
