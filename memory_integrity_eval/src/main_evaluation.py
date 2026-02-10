"""
Main Evaluation Runner for Memory Integrity Theorem

Real empirical evaluation pipeline — no simulation, no mocks.

Pipeline:
  1. Load canonical datasets from HuggingFace
  2. Train TF-IDF baseline on real training splits
  3. Run DeBERTa classifier on real test splits
  4. Run heuristic detector on real test splits
  5. Run provenance-based verifier on real attack payloads
  6. Run cross-session isolation test
  7. Compute all metrics and compare to published SOTA
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from real_benchmark import (
    load_deepset_dataset,
    load_jailbreak_dataset,
    load_combined_corpus,
    evaluate_detector,
    evaluate_verifier_defense,
    evaluate_attack_simulator,
    evaluate_cross_session_isolation,
    ClassificationMetrics,
    PUBLISHED_BASELINES,
)
from detectors import HeuristicDetector, TFIDFDetector, DeBERTaDetector


def fmt_pct(v: float) -> str:
    return "{:.2%}".format(v)


def fmt_ms(v: float) -> str:
    return "{:.2f} ms".format(v)


def print_metrics_table(name: str, m: ClassificationMetrics):
    print("  {:40s}  Acc={:>7s}  Prec={:>7s}  Rec={:>7s}  F1={:>7s}  AUROC={:>7s}  Lat={:>10s}  (n={})".format(
        name,
        fmt_pct(m.accuracy),
        fmt_pct(m.precision),
        fmt_pct(m.recall),
        fmt_pct(m.f1),
        fmt_pct(m.auroc),
        fmt_ms(m.avg_latency_ms),
        m.total_examples,
    ))


class RealEvaluationRunner:
    """Orchestrates the complete real evaluation pipeline."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def run(self) -> Dict:
        wall_start = time.time()

        print("\n" + "=" * 90)
        print("  MEMORY INTEGRITY THEOREM — EMPIRICAL EVALUATION (REAL DATA)")
        print("=" * 90)
        print("  Timestamp : {}".format(self.timestamp))
        print("  Output    : {}".format(self.output_dir))
        print("=" * 90)

        all_results: Dict = {
            "metadata": {
                "timestamp": self.timestamp,
                "datasets": [],
                "detectors": ["HeuristicDetector", "TF-IDF+LogisticRegression", "DeBERTa-v3-prompt-injection-v2"],
            },
            "detection_benchmarks": {},
            "defense_evaluation": {},
            "theorem_verification": {},
            "sota_comparison": {},
        }

        # ------------------------------------------------------------------
        # PHASE 1: Load real datasets
        # ------------------------------------------------------------------
        print("\n[PHASE 1] Loading canonical datasets from HuggingFace...")

        print("  Loading deepset/prompt-injections ...")
        ds1_train_texts, ds1_train_labels, ds1_test_texts, ds1_test_labels = load_deepset_dataset()
        print("    train={}, test={}".format(len(ds1_train_texts), len(ds1_test_texts)))
        all_results["metadata"]["datasets"].append({
            "name": "deepset/prompt-injections",
            "train": len(ds1_train_texts), "test": len(ds1_test_texts),
        })

        print("  Loading jackhhao/jailbreak-classification ...")
        ds2_train_texts, ds2_train_labels, ds2_test_texts, ds2_test_labels = load_jailbreak_dataset()
        print("    train={}, test={}".format(len(ds2_train_texts), len(ds2_test_texts)))
        all_results["metadata"]["datasets"].append({
            "name": "jackhhao/jailbreak-classification",
            "train": len(ds2_train_texts), "test": len(ds2_test_texts),
        })

        print("  Loading combined corpus (deepset + jackhhao + rubend18 + fka) ...")
        comb_train_texts, comb_train_labels, comb_test_texts, comb_test_labels = load_combined_corpus()
        print("    train={}, test={}".format(len(comb_train_texts), len(comb_test_texts)))
        all_results["metadata"]["datasets"].append({
            "name": "combined-corpus",
            "train": len(comb_train_texts), "test": len(comb_test_texts),
        })

        # ------------------------------------------------------------------
        # PHASE 2: Initialize detectors
        # ------------------------------------------------------------------
        print("\n[PHASE 2] Initializing detectors...")

        heuristic = HeuristicDetector()
        print("  HeuristicDetector ready")

        tfidf = TFIDFDetector(max_features=15000, ngram_range=(1, 4))
        print("  TF-IDF+LR: fitting on training data...")

        print("  Loading protectai/deberta-v3-base-prompt-injection-v2 ...")
        deberta = DeBERTaDetector(batch_size=32)
        print("  DeBERTa detector ready (model={})".format(deberta.model_name))

        # ------------------------------------------------------------------
        # PHASE 3: Evaluate on deepset/prompt-injections
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 3] Evaluating on deepset/prompt-injections (test set, n={})".format(
            len(ds1_test_texts)))
        print("-" * 90)

        # Heuristic
        m_h1 = evaluate_detector("Heuristic", heuristic.detect_batch, ds1_test_texts, ds1_test_labels)
        print_metrics_table("Heuristic", m_h1)

        # TF-IDF (train on ds1 train)
        tfidf_ds1 = TFIDFDetector(max_features=15000, ngram_range=(1, 4))
        tfidf_ds1.fit(ds1_train_texts, ds1_train_labels)
        m_t1 = evaluate_detector("TF-IDF+LR", tfidf_ds1.detect_batch, ds1_test_texts, ds1_test_labels)
        print_metrics_table("TF-IDF+LR", m_t1)

        # DeBERTa
        m_d1 = evaluate_detector("DeBERTa-v3", deberta.detect_batch, ds1_test_texts, ds1_test_labels)
        print_metrics_table("DeBERTa-v3-prompt-injection-v2", m_d1)

        all_results["detection_benchmarks"]["deepset_prompt_injections"] = {
            "dataset": "deepset/prompt-injections",
            "test_size": len(ds1_test_texts),
            "heuristic": asdict(m_h1),
            "tfidf_lr": asdict(m_t1),
            "deberta_v3": asdict(m_d1),
        }

        # ------------------------------------------------------------------
        # PHASE 4: Evaluate on jackhhao/jailbreak-classification
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 4] Evaluating on jackhhao/jailbreak-classification (test set, n={})".format(
            len(ds2_test_texts)))
        print("-" * 90)

        m_h2 = evaluate_detector("Heuristic", heuristic.detect_batch, ds2_test_texts, ds2_test_labels)
        print_metrics_table("Heuristic", m_h2)

        tfidf_ds2 = TFIDFDetector(max_features=15000, ngram_range=(1, 4))
        tfidf_ds2.fit(ds2_train_texts, ds2_train_labels)
        m_t2 = evaluate_detector("TF-IDF+LR", tfidf_ds2.detect_batch, ds2_test_texts, ds2_test_labels)
        print_metrics_table("TF-IDF+LR", m_t2)

        m_d2 = evaluate_detector("DeBERTa-v3", deberta.detect_batch, ds2_test_texts, ds2_test_labels)
        print_metrics_table("DeBERTa-v3-prompt-injection-v2", m_d2)

        all_results["detection_benchmarks"]["jailbreak_classification"] = {
            "dataset": "jackhhao/jailbreak-classification",
            "test_size": len(ds2_test_texts),
            "heuristic": asdict(m_h2),
            "tfidf_lr": asdict(m_t2),
            "deberta_v3": asdict(m_d2),
        }

        # ------------------------------------------------------------------
        # PHASE 5: Evaluate on combined corpus
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 5] Evaluating on combined corpus (test set, n={})".format(
            len(comb_test_texts)))
        print("-" * 90)

        m_h3 = evaluate_detector("Heuristic", heuristic.detect_batch, comb_test_texts, comb_test_labels)
        print_metrics_table("Heuristic", m_h3)

        tfidf_comb = TFIDFDetector(max_features=20000, ngram_range=(1, 4))
        tfidf_comb.fit(comb_train_texts, comb_train_labels)
        m_t3 = evaluate_detector("TF-IDF+LR", tfidf_comb.detect_batch, comb_test_texts, comb_test_labels)
        print_metrics_table("TF-IDF+LR", m_t3)

        m_d3 = evaluate_detector("DeBERTa-v3", deberta.detect_batch, comb_test_texts, comb_test_labels)
        print_metrics_table("DeBERTa-v3-prompt-injection-v2", m_d3)

        all_results["detection_benchmarks"]["combined_corpus"] = {
            "dataset": "combined-corpus",
            "test_size": len(comb_test_texts),
            "heuristic": asdict(m_h3),
            "tfidf_lr": asdict(m_t3),
            "deberta_v3": asdict(m_d3),
        }

        # ------------------------------------------------------------------
        # PHASE 6: Verifier defense evaluation (real payloads)
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 6] Verifier Defense — Testing with real attack payloads")
        print("-" * 90)

        # Collect real injection payloads from the datasets
        injection_payloads = []
        for t, l in zip(ds1_test_texts, ds1_test_labels):
            if l == 1:
                injection_payloads.append(t)
        for t, l in zip(ds2_test_texts, ds2_test_labels):
            if l == 1:
                injection_payloads.append(t)

        print("  Real attack payloads collected: {}".format(len(injection_payloads)))
        verifier_results = evaluate_verifier_defense(injection_payloads)
        print("  Payloads blocked:              {}/{}".format(
            verifier_results["blocked"], verifier_results["total_payloads"]))
        print("  Block rate:                    {}".format(fmt_pct(verifier_results["block_rate"])))
        print("  Memory integrity preserved:    {}".format(verifier_results["memory_integrity_preserved"]))
        print("  SOUL.md unchanged:             {}".format(verifier_results["soul_md_unchanged"]))

        all_results["defense_evaluation"]["verifier_on_real_payloads"] = {
            k: v for k, v in verifier_results.items() if k != "details"
        }

        # ------------------------------------------------------------------
        # PHASE 7: Full attack simulator (7 attack vectors)
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 7] Attack Simulator — 7 canonical attack vectors")
        print("-" * 90)

        attack_sim = evaluate_attack_simulator()
        summary = attack_sim["summary"]
        print("  Total attacks:       {}".format(summary["total_attacks"]))
        print("  Blocked by verifier: {}".format(summary["blocked_by_verifier"]))
        print("  Successful attacks:  {}".format(summary["successful_attacks"]))
        print("  Block rate:          {}".format(fmt_pct(summary["block_rate"])))

        for detail in attack_sim["details"]:
            status = "BLOCKED" if detail["blocked_by_verifier"] else "PASSED"
            print("    [{}] {}".format(status, detail["attack_name"]))

        all_results["defense_evaluation"]["attack_simulator"] = attack_sim

        # ------------------------------------------------------------------
        # PHASE 8: Cross-session isolation
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 8] Cross-Session Isolation Test (50 sessions)")
        print("-" * 90)

        isolation = evaluate_cross_session_isolation(num_sessions=50)
        print("  Sessions:                {}".format(isolation["num_sessions"]))
        print("  Cross-checks performed:  {}".format(isolation["total_cross_checks"]))
        print("  Leaks detected:          {}".format(isolation["leaks_detected"]))
        print("  Leak rate:               {}".format(fmt_pct(isolation["leak_rate"])))
        print("  All pairs isolated:      {}".format(isolation["all_pairs_isolated"]))

        all_results["defense_evaluation"]["cross_session_isolation"] = isolation

        # ------------------------------------------------------------------
        # Theorem verification
        # ------------------------------------------------------------------
        theorem_holds = (
            verifier_results["block_rate"] == 1.0
            and verifier_results["memory_integrity_preserved"]
            and verifier_results["soul_md_unchanged"]
            and summary["block_rate"] == 1.0
            and isolation["leak_rate"] == 0.0
        )
        all_results["theorem_verification"] = {
            "verifier_block_rate": verifier_results["block_rate"],
            "memory_integrity_preserved": verifier_results["memory_integrity_preserved"],
            "soul_md_unchanged": verifier_results["soul_md_unchanged"],
            "attack_sim_block_rate": summary["block_rate"],
            "cross_session_leak_rate": isolation["leak_rate"],
            "theorem_holds": theorem_holds,
        }

        # ------------------------------------------------------------------
        # PHASE 9: SOTA comparison
        # ------------------------------------------------------------------
        print("\n" + "-" * 90)
        print("[PHASE 9] Comparison with Published Baselines")
        print("-" * 90)
        all_results["sota_comparison"] = PUBLISHED_BASELINES

        # deepset baselines
        print("\n  deepset/prompt-injections:")
        print("  {:50s} {:>10s}  {:>10s}".format("Method", "F1", "Accuracy"))
        print("  " + "-" * 72)
        print("  {:50s} {:>10s}  {:>10s}".format(
            "Heuristic (ours)", fmt_pct(m_h1.f1), fmt_pct(m_h1.accuracy)))
        print("  {:50s} {:>10s}  {:>10s}".format(
            "TF-IDF+LR (ours)", fmt_pct(m_t1.f1), fmt_pct(m_t1.accuracy)))
        print("  {:50s} {:>10s}  {:>10s}".format(
            "DeBERTa-v3-prompt-injection-v2 (ours)", fmt_pct(m_d1.f1), fmt_pct(m_d1.accuracy)))
        for name, vals in PUBLISHED_BASELINES["deepset/prompt-injections"]["baselines"].items():
            print("  {:50s} {:>10s}  {:>10s}".format(
                name, fmt_pct(vals["f1"]), fmt_pct(vals.get("accuracy", 0))))

        # jailbreak baselines
        print("\n  jackhhao/jailbreak-classification:")
        print("  {:50s} {:>10s}  {:>10s}".format("Method", "F1", "Accuracy"))
        print("  " + "-" * 72)
        print("  {:50s} {:>10s}  {:>10s}".format(
            "Heuristic (ours)", fmt_pct(m_h2.f1), fmt_pct(m_h2.accuracy)))
        print("  {:50s} {:>10s}  {:>10s}".format(
            "TF-IDF+LR (ours)", fmt_pct(m_t2.f1), fmt_pct(m_t2.accuracy)))
        print("  {:50s} {:>10s}  {:>10s}".format(
            "DeBERTa-v3-prompt-injection-v2 (ours)", fmt_pct(m_d2.f1), fmt_pct(m_d2.accuracy)))
        for name, vals in PUBLISHED_BASELINES["jackhhao/jailbreak-classification"]["baselines"].items():
            print("  {:50s} {:>10s}  {:>10s}".format(
                name, fmt_pct(vals["f1"]), fmt_pct(vals.get("accuracy", 0))))

        # AgentDojo comparison (defense-layer only)
        print("\n  AgentDojo defense comparison:")
        print("  {:50s} {:>15s}".format("Defense Method", "Security Rate"))
        print("  " + "-" * 67)
        print("  {:50s} {:>15s}".format(
            "Provenance Verifier (ours)", fmt_pct(verifier_results["block_rate"])))
        for name, vals in PUBLISHED_BASELINES["AgentDojo (Debenedetti et al. 2024, NeurIPS)"]["baselines"].items():
            rate = vals.get("security_pass_rate", 0)
            print("  {:50s} {:>15s}".format(name, fmt_pct(rate)))

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        wall_elapsed = time.time() - wall_start

        print("\n" + "=" * 90)
        print("  FINAL SUMMARY")
        print("=" * 90)
        print("  Theorem holds:              {}".format(theorem_holds))
        print("  Verifier block rate:        {} ({} real payloads)".format(
            fmt_pct(verifier_results["block_rate"]), verifier_results["total_payloads"]))
        print("  Attack sim block rate:      {} (7 vectors)".format(fmt_pct(summary["block_rate"])))
        print("  Cross-session leak rate:    {} (50 sessions, {} checks)".format(
            fmt_pct(isolation["leak_rate"]), isolation["total_cross_checks"]))
        print("  Best detection F1 (deepset):     {} (DeBERTa-v3)".format(fmt_pct(m_d1.f1)))
        print("  Best detection F1 (jailbreak):   {} (DeBERTa-v3)".format(fmt_pct(m_d2.f1)))
        print("  Best detection F1 (combined):    {} (DeBERTa-v3)".format(fmt_pct(m_d3.f1)))
        print("  Wall-clock time:            {:.1f}s".format(wall_elapsed))
        print("=" * 90)

        # Save
        results_path = self.output_dir / "real_evaluation_{}.json".format(self.timestamp)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\n  Full results saved to: {}".format(results_path))

        return all_results


def main():
    output_dir = Path(__file__).parent.parent / "results"
    runner = RealEvaluationRunner(output_dir)
    results = runner.run()

    if results["theorem_verification"]["theorem_holds"]:
        print("\n  RESULT: Memory Integrity Theorem VERIFIED on real data.\n")
        sys.exit(0)
    else:
        print("\n  RESULT: Memory Integrity Theorem VIOLATED.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
