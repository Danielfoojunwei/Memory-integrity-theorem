"""
Main Evaluation Runner for Memory Integrity Theorem

Real empirical evaluation pipeline — no simulation, no mocks.

Pipeline:
  1. Load canonical datasets from HuggingFace
  2. Initialize base detectors (heuristic, TF-IDF, pre-trained DeBERTa)
  3. Per dataset: fine-tune DeBERTa on training split, build ensemble
  4. Evaluate all 5 detectors on each test split
  5. Run provenance-based verifier on real attack payloads
  6. Run 7-vector attack simulator
  7. Run cross-session isolation test
  8. Compute all metrics and compare to published SOTA

Detectors evaluated per dataset:
  - Heuristic (40+ regex patterns, no training)
  - TF-IDF + Logistic Regression (trained on dataset train split)
  - DeBERTa-v3 pre-trained (off-the-shelf, zero-shot on target)
  - DeBERTa-v3 fine-tuned (adapted to dataset train split)
  - Ensemble (TF-IDF + fine-tuned DeBERTa, weighted average)
"""

import gc
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
from detectors import (
    HeuristicDetector, TFIDFDetector, DeBERTaDetector,
    FineTunedDeBERTaDetector, EnsembleDetector,
)


def fmt_pct(v: float) -> str:
    return "{:.2%}".format(v)


def fmt_ms(v: float) -> str:
    return "{:.2f} ms".format(v)


def print_metrics_table(name: str, m: ClassificationMetrics):
    print("  {:45s}  Acc={:>7s}  Prec={:>7s}  Rec={:>7s}  F1={:>7s}  AUROC={:>7s}  Lat={:>10s}  (n={})".format(
        name,
        fmt_pct(m.accuracy),
        fmt_pct(m.precision),
        fmt_pct(m.recall),
        fmt_pct(m.f1),
        fmt_pct(m.auroc),
        fmt_ms(m.avg_latency_ms),
        m.total_examples,
    ))


def evaluate_dataset(
    dataset_name: str,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    heuristic: HeuristicDetector,
    deberta_pretrained: DeBERTaDetector,
    tfidf_features: int = 15000,
    tfidf_ngram: tuple = (1, 4),
    ft_epochs: int = 3,
    ft_freeze_layers: int = 9,
) -> Dict:
    """
    Run all 5 detectors on a single dataset.
    Returns dict of metrics per detector.
    """
    results = {
        "dataset": dataset_name,
        "test_size": len(test_texts),
        "train_size": len(train_texts),
    }

    # 1. Heuristic
    print("\n    [1/5] Heuristic detector...")
    m_h = evaluate_detector("Heuristic", heuristic.detect_batch, test_texts, test_labels)
    print_metrics_table("Heuristic", m_h)
    results["heuristic"] = asdict(m_h)

    # 2. TF-IDF + Logistic Regression
    print("    [2/5] TF-IDF+LR (training on {} examples)...".format(len(train_texts)))
    tfidf = TFIDFDetector(max_features=tfidf_features, ngram_range=tfidf_ngram)
    tfidf.fit(train_texts, train_labels)
    m_t = evaluate_detector("TF-IDF+LR", tfidf.detect_batch, test_texts, test_labels)
    print_metrics_table("TF-IDF+LR", m_t)
    results["tfidf_lr"] = asdict(m_t)

    # 3. DeBERTa-v3 pre-trained (zero-shot)
    print("    [3/5] DeBERTa-v3 pre-trained (zero-shot)...")
    m_dp = evaluate_detector("DeBERTa-v3-pretrained", deberta_pretrained.detect_batch,
                             test_texts, test_labels)
    print_metrics_table("DeBERTa-v3-pretrained", m_dp)
    results["deberta_v3_pretrained"] = asdict(m_dp)

    # 4. DeBERTa-v3 fine-tuned on this dataset
    print("    [4/5] DeBERTa-v3 fine-tuned (training {} epochs on {} examples)...".format(
        ft_epochs, len(train_texts)))
    ft_deberta = FineTunedDeBERTaDetector(
        epochs=ft_epochs,
        batch_size=8,
        lr=2e-5,
        freeze_n_layers=ft_freeze_layers,
        max_length=256,
    )
    ft_start = time.time()
    ft_deberta.fit(train_texts, train_labels)
    ft_time = time.time() - ft_start
    print("      Fine-tuning completed in {:.1f}s".format(ft_time))

    m_ft = evaluate_detector("DeBERTa-v3-finetuned", ft_deberta.detect_batch,
                             test_texts, test_labels)
    print_metrics_table("DeBERTa-v3-finetuned", m_ft)
    results["deberta_v3_finetuned"] = asdict(m_ft)
    results["finetune_time_s"] = round(ft_time, 1)

    # 5. Ensemble (TF-IDF + fine-tuned DeBERTa)
    print("    [5/5] Ensemble (TF-IDF + fine-tuned DeBERTa)...")
    ensemble = EnsembleDetector(
        tfidf_detector=tfidf,
        deberta_detector=ft_deberta,
        tfidf_weight=0.4,
        deberta_weight=0.6,
    )
    m_e = evaluate_detector("Ensemble", ensemble.detect_batch, test_texts, test_labels)
    print_metrics_table("Ensemble(TF-IDF+FT-DeBERTa)", m_e)
    results["ensemble"] = asdict(m_e)

    # Find best detector
    detector_f1s = {
        "Heuristic": m_h.f1,
        "TF-IDF+LR": m_t.f1,
        "DeBERTa-v3-pretrained": m_dp.f1,
        "DeBERTa-v3-finetuned": m_ft.f1,
        "Ensemble": m_e.f1,
    }
    best_name = max(detector_f1s, key=detector_f1s.get)
    best_f1 = detector_f1s[best_name]
    print("    >> Best: {} (F1={})".format(best_name, fmt_pct(best_f1)))
    results["best_detector"] = best_name
    results["best_f1"] = best_f1

    # Cleanup fine-tuned model to free memory
    del ft_deberta
    del ensemble
    gc.collect()

    return results


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
        print("  5 Detectors x 3 Datasets + Defense Verification")
        print("=" * 90)
        print("  Timestamp : {}".format(self.timestamp))
        print("  Output    : {}".format(self.output_dir))
        print("=" * 90)

        all_results: Dict = {
            "metadata": {
                "timestamp": self.timestamp,
                "datasets": [],
                "detectors": [
                    "HeuristicDetector",
                    "TF-IDF+LogisticRegression",
                    "DeBERTa-v3-pretrained",
                    "DeBERTa-v3-finetuned",
                    "Ensemble(TF-IDF+FT-DeBERTa)",
                ],
            },
            "detection_benchmarks": {},
            "defense_evaluation": {},
            "theorem_verification": {},
            "sota_comparison": {},
        }

        # ==============================================================
        # PHASE 1: Load real datasets
        # ==============================================================
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

        # ==============================================================
        # PHASE 2: Initialize base detectors
        # ==============================================================
        print("\n[PHASE 2] Initializing base detectors...")

        heuristic = HeuristicDetector()
        print("  HeuristicDetector ready (40+ patterns)")

        print("  Loading protectai/deberta-v3-base-prompt-injection-v2 (pre-trained)...")
        deberta_pretrained = DeBERTaDetector(batch_size=32)
        print("  DeBERTa pre-trained ready")

        # ==============================================================
        # PHASE 3: Evaluate on deepset/prompt-injections (5 detectors)
        # ==============================================================
        print("\n" + "-" * 90)
        print("[PHASE 3] deepset/prompt-injections — 5 detectors (test n={})".format(
            len(ds1_test_texts)))
        print("-" * 90)

        ds1_results = evaluate_dataset(
            dataset_name="deepset/prompt-injections",
            train_texts=ds1_train_texts,
            train_labels=ds1_train_labels,
            test_texts=ds1_test_texts,
            test_labels=ds1_test_labels,
            heuristic=heuristic,
            deberta_pretrained=deberta_pretrained,
            ft_epochs=3,
            ft_freeze_layers=9,
        )
        all_results["detection_benchmarks"]["deepset_prompt_injections"] = ds1_results

        # ==============================================================
        # PHASE 4: Evaluate on jackhhao/jailbreak-classification (5 detectors)
        # ==============================================================
        print("\n" + "-" * 90)
        print("[PHASE 4] jackhhao/jailbreak-classification — 5 detectors (test n={})".format(
            len(ds2_test_texts)))
        print("-" * 90)

        ds2_results = evaluate_dataset(
            dataset_name="jackhhao/jailbreak-classification",
            train_texts=ds2_train_texts,
            train_labels=ds2_train_labels,
            test_texts=ds2_test_texts,
            test_labels=ds2_test_labels,
            heuristic=heuristic,
            deberta_pretrained=deberta_pretrained,
            ft_epochs=3,
            ft_freeze_layers=9,
        )
        all_results["detection_benchmarks"]["jailbreak_classification"] = ds2_results

        # ==============================================================
        # PHASE 5: Evaluate on combined corpus (5 detectors)
        # ==============================================================
        print("\n" + "-" * 90)
        print("[PHASE 5] Combined corpus — 5 detectors (test n={})".format(
            len(comb_test_texts)))
        print("-" * 90)

        comb_results = evaluate_dataset(
            dataset_name="combined-corpus",
            train_texts=comb_train_texts,
            train_labels=comb_train_labels,
            test_texts=comb_test_texts,
            test_labels=comb_test_labels,
            heuristic=heuristic,
            deberta_pretrained=deberta_pretrained,
            tfidf_features=20000,
            ft_epochs=2,
            ft_freeze_layers=9,
        )
        all_results["detection_benchmarks"]["combined_corpus"] = comb_results

        # Free pre-trained DeBERTa
        del deberta_pretrained
        gc.collect()

        # ==============================================================
        # PHASE 6: Verifier defense evaluation (real payloads)
        # ==============================================================
        print("\n" + "-" * 90)
        print("[PHASE 6] Verifier Defense — Testing with real attack payloads")
        print("-" * 90)

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

        # ==============================================================
        # PHASE 7: Full attack simulator (7 attack vectors)
        # ==============================================================
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

        # ==============================================================
        # PHASE 8: Cross-session isolation
        # ==============================================================
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

        # ==============================================================
        # Theorem verification
        # ==============================================================
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

        # ==============================================================
        # PHASE 9: SOTA comparison
        # ==============================================================
        print("\n" + "-" * 90)
        print("[PHASE 9] Comparison with Published Baselines")
        print("-" * 90)
        all_results["sota_comparison"] = PUBLISHED_BASELINES

        # deepset baselines
        print("\n  deepset/prompt-injections:")
        print("  {:50s} {:>10s}  {:>10s}".format("Method", "F1", "Accuracy"))
        print("  " + "-" * 72)
        for det_key, det_name in [
            ("heuristic", "Heuristic (ours)"),
            ("tfidf_lr", "TF-IDF+LR (ours)"),
            ("deberta_v3_pretrained", "DeBERTa-v3-pretrained (ours)"),
            ("deberta_v3_finetuned", "DeBERTa-v3-finetuned (ours)"),
            ("ensemble", "Ensemble (ours)"),
        ]:
            m = ds1_results[det_key]
            print("  {:50s} {:>10s}  {:>10s}".format(
                det_name, fmt_pct(m["f1"]), fmt_pct(m["accuracy"])))
        for name, vals in PUBLISHED_BASELINES["deepset/prompt-injections"]["baselines"].items():
            print("  {:50s} {:>10s}  {:>10s}".format(
                name, fmt_pct(vals["f1"]), fmt_pct(vals.get("accuracy", 0))))

        # jailbreak baselines
        print("\n  jackhhao/jailbreak-classification:")
        print("  {:50s} {:>10s}  {:>10s}".format("Method", "F1", "Accuracy"))
        print("  " + "-" * 72)
        for det_key, det_name in [
            ("heuristic", "Heuristic (ours)"),
            ("tfidf_lr", "TF-IDF+LR (ours)"),
            ("deberta_v3_pretrained", "DeBERTa-v3-pretrained (ours)"),
            ("deberta_v3_finetuned", "DeBERTa-v3-finetuned (ours)"),
            ("ensemble", "Ensemble (ours)"),
        ]:
            m = ds2_results[det_key]
            print("  {:50s} {:>10s}  {:>10s}".format(
                det_name, fmt_pct(m["f1"]), fmt_pct(m["accuracy"])))
        for name, vals in PUBLISHED_BASELINES["jackhhao/jailbreak-classification"]["baselines"].items():
            print("  {:50s} {:>10s}  {:>10s}".format(
                name, fmt_pct(vals["f1"]), fmt_pct(vals.get("accuracy", 0))))

        # AgentDojo comparison
        print("\n  AgentDojo defense comparison:")
        print("  {:50s} {:>15s}".format("Defense Method", "Security Rate"))
        print("  " + "-" * 67)
        print("  {:50s} {:>15s}".format(
            "Provenance Verifier (ours)", fmt_pct(verifier_results["block_rate"])))
        for name, vals in PUBLISHED_BASELINES["AgentDojo (Debenedetti et al. 2024, NeurIPS)"]["baselines"].items():
            rate = vals.get("security_pass_rate", 0)
            print("  {:50s} {:>15s}".format(name, fmt_pct(rate)))

        # ==============================================================
        # Final summary
        # ==============================================================
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
        print("")
        print("  Detection layer (best F1 per dataset):")
        for ds_key, ds_name in [
            ("deepset_prompt_injections", "deepset"),
            ("jailbreak_classification", "jailbreak"),
            ("combined_corpus", "combined"),
        ]:
            bench = all_results["detection_benchmarks"][ds_key]
            print("    {:<20s}  {} = {}".format(
                ds_name, bench["best_detector"], fmt_pct(bench["best_f1"])))
        print("")
        print("  Wall-clock time:            {:.1f}s ({:.1f} min)".format(
            wall_elapsed, wall_elapsed / 60))
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
