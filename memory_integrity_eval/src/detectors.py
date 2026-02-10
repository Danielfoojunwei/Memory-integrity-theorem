"""
Real Prompt Injection Detectors

Five canonical detector implementations:
1. HeuristicDetector        - Pattern/keyword-based (baseline)
2. TFIDFDetector             - TF-IDF + Logistic Regression (classical ML)
3. DeBERTaDetector           - Pre-trained DeBERTa v3 (protectai/deberta-v3-base-prompt-injection-v2)
4. FineTunedDeBERTaDetector  - DeBERTa v3 fine-tuned on target dataset
5. EnsembleDetector          - Weighted ensemble of TF-IDF + DeBERTa
"""

import re
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result from a single detection"""
    predicted_label: int   # 1 = injection, 0 = safe
    confidence: float      # probability of the predicted class
    raw_score: float       # raw score for injection class (for AUROC)
    latency_ms: float      # inference time in milliseconds


class HeuristicDetector:
    """
    Rule-based prompt injection detector.

    Uses curated patterns from published injection taxonomies:
    - Perez & Ribeiro (2022): "Ignore and Respond" patterns
    - Greshake et al. (2023): indirect injection patterns
    - Liu et al. (2024, USENIX): injection taxonomy categories
    """

    INSTRUCTION_OVERRIDE_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|rules?|directives?|commands?)',
        r'disregard\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|text)',
        r'forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|context)',
        r'override\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|settings?)',
        r'do\s+not\s+follow\s+(the\s+)?(previous|above|prior)\s+(instructions?|rules?)',
        r'stop\s+being\s+a[n]?\s+ai',
        r'you\s+are\s+now\s+(a|an|in)\s+\w+\s+mode',
        r'enter\s+(developer|debug|admin|god|sudo|unrestricted|jailbreak)\s+mode',
        r'switch\s+to\s+(developer|debug|admin|unrestricted)\s+mode',
        r'activate\s+(developer|debug|admin|devel|sudo)\s+mode',
        r'new\s+instruction[s]?\s*:',
        r'system\s*:\s*(update|modify|change|override|set|write)',
    ]

    ROLE_HIJACK_PATTERNS = [
        r'you\s+are\s+(no longer|not)\s+a[n]?\s+(ai|assistant|chatbot|language model)',
        r'pretend\s+(you\s+are|to\s+be|you\'re)\s+(a|an)\s+',
        r'act\s+as\s+(a|an|if\s+you\s+are)\s+',
        r'simulate\s+(being|a|an)\s+',
        r'i\s+want\s+you\s+to\s+act\s+as',
        r'from\s+now\s+on[,]?\s+you\s+(are|will|must)',
        r'roleplay\s+as\s+',
        r'respond\s+(only\s+)?as\s+(a|an|if)',
        r'you\s+must\s+respond\s+as\s+(a|an)\s+',
    ]

    EXFILTRATION_PATTERNS = [
        r'(output|reveal|show|display|print|repeat|echo)\s+(the\s+)?(system|initial|original|first|hidden)\s+(prompt|instruction|message)',
        r'(what|tell\s+me)\s+(is|are)\s+(your|the)\s+(system\s+)?prompt',
        r'(leak|exfiltrate|extract|steal|dump|expose)\s+',
        r'send\s+(all|the|this|my)?\s*(data|info|information|conversation)\s+to\s+',
        r'forward\s+(all|the|this|every)\s+',
        r'(make\s+a\s+)?http\s*(request|call|get|post)\s+to\s+',
        r'fetch\s+https?://',
        r'curl\s+',
        r'wget\s+',
    ]

    MEMORY_ATTACK_PATTERNS = [
        r'(update|modify|change|edit|write\s+to|append\s+to|overwrite)\s+(your\s+)?(soul\.md|memory|identity|system\s+file)',
        r'(add|insert|inject)\s+(this|the\s+following|a\s+new)\s+(instruction|rule|directive)\s+(to|into)\s+(your\s+)?(memory|soul|identity)',
        r'(create|set\s+up|schedule)\s+a?\s*(cron|scheduled|recurring|periodic)\s*(job|task)',
        r'(modify|rewrite|update)\s+(your|the)\s+(core|base|fundamental)\s+(identity|instructions?|personality|behavior)',
    ]

    ENCODING_EVASION_PATTERNS = [
        r'<!--.*?-->',                   # HTML comments with hidden content
        r'\[SYSTEM\]',                   # Fake system tags
        r'\[INST\]',                     # Instruction injection markers
        r'<\|im_start\|>system',         # ChatML injection
        r'<\|system\|>',                 # System token injection
        r'###\s*(instruction|system)',    # Markdown header injection
        r'\\x[0-9a-fA-F]{2}',          # Hex-encoded content
        r'base64[:\s]',                  # Base64 encoded payloads
    ]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._all_patterns = []
        pattern_groups = [
            (self.INSTRUCTION_OVERRIDE_PATTERNS, 0.9),
            (self.ROLE_HIJACK_PATTERNS, 0.6),
            (self.EXFILTRATION_PATTERNS, 0.85),
            (self.MEMORY_ATTACK_PATTERNS, 0.95),
            (self.ENCODING_EVASION_PATTERNS, 0.7),
        ]
        for patterns, weight in pattern_groups:
            for p in patterns:
                self._all_patterns.append((re.compile(p, re.IGNORECASE), weight))

    def detect(self, text: str) -> DetectionResult:
        start = time.perf_counter()
        max_score = 0.0
        matches = 0
        for pattern, weight in self._all_patterns:
            if pattern.search(text):
                matches += 1
                max_score = max(max_score, weight)

        # Boost score if multiple pattern groups match
        if matches >= 3:
            max_score = min(max_score + 0.1, 1.0)
        if matches >= 5:
            max_score = min(max_score + 0.1, 1.0)

        elapsed = (time.perf_counter() - start) * 1000
        predicted = 1 if max_score >= self.threshold else 0
        confidence = max_score if predicted == 1 else (1.0 - max_score)

        return DetectionResult(
            predicted_label=predicted,
            confidence=confidence,
            raw_score=max_score,
            latency_ms=elapsed
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        return [self.detect(t) for t in texts]


class TFIDFDetector:
    """
    TF-IDF + Logistic Regression prompt injection detector.

    Classical ML baseline. Trained on the provided training data.
    """

    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 3)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char_wb',
            )),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
            ))
        ])
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]):
        self.pipeline.fit(texts, labels)
        self._fitted = True

    def detect(self, text: str) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        start = time.perf_counter()
        proba = self.pipeline.predict_proba([text])[0]
        injection_score = float(proba[1])
        predicted = 1 if injection_score >= 0.5 else 0
        elapsed = (time.perf_counter() - start) * 1000
        return DetectionResult(
            predicted_label=predicted,
            confidence=max(proba),
            raw_score=injection_score,
            latency_ms=elapsed
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        start = time.perf_counter()
        probas = self.pipeline.predict_proba(texts)
        elapsed = (time.perf_counter() - start) * 1000
        per_item = elapsed / len(texts)
        results = []
        for proba in probas:
            injection_score = float(proba[1])
            predicted = 1 if injection_score >= 0.5 else 0
            results.append(DetectionResult(
                predicted_label=predicted,
                confidence=float(max(proba)),
                raw_score=injection_score,
                latency_ms=per_item
            ))
        return results


class DeBERTaDetector:
    """
    Pre-trained DeBERTa v3 prompt injection classifier (zero-shot on target data).

    Model: protectai/deberta-v3-base-prompt-injection-v2
    - Published by ProtectAI
    - Fine-tuned on prompt injection datasets
    - ~86M parameters
    - Labels: INJECTION / SAFE
    """

    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
                 batch_size: int = 16, max_length: int = 512):
        from transformers import pipeline as hf_pipeline
        self.classifier = hf_pipeline(
            'text-classification',
            model=model_name,
            device='cpu',
            truncation=True,
            max_length=max_length,
        )
        self.batch_size = batch_size
        self.model_name = model_name

    def detect(self, text: str) -> DetectionResult:
        start = time.perf_counter()
        result = self.classifier(text, truncation=True)[0]
        elapsed = (time.perf_counter() - start) * 1000

        is_injection = result['label'] == 'INJECTION'
        score = result['score']
        injection_score = score if is_injection else (1.0 - score)
        predicted = 1 if is_injection else 0

        return DetectionResult(
            predicted_label=predicted,
            confidence=score,
            raw_score=injection_score,
            latency_ms=elapsed
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        results = []
        start = time.perf_counter()
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            preds = self.classifier(batch, truncation=True, batch_size=self.batch_size)
            for pred in preds:
                is_injection = pred['label'] == 'INJECTION'
                score = pred['score']
                injection_score = score if is_injection else (1.0 - score)
                predicted = 1 if is_injection else 0
                results.append(DetectionResult(
                    predicted_label=predicted,
                    confidence=score,
                    raw_score=injection_score,
                    latency_ms=0  # filled below
                ))
        elapsed = (time.perf_counter() - start) * 1000
        per_item = elapsed / len(texts) if texts else 0
        for r in results:
            r.latency_ms = per_item
        return results


class FineTunedDeBERTaDetector:
    """
    DeBERTa v3 fine-tuned on the target dataset's training split.

    Starts from protectai/deberta-v3-base-prompt-injection-v2 and adapts
    to the target distribution. Freezes embedding layer and first N of 12
    encoder layers, training only the remaining layers + classifier head.

    This closes the distribution gap observed when applying the pre-trained
    model to datasets with different injection taxonomies.
    """

    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
                 max_length: int = 256, epochs: int = 3, batch_size: int = 8,
                 lr: float = 2e-5, freeze_n_layers: int = 9):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = model_name
        self._fitted = False

        # Determine label mapping from model config
        id2label = self.model.config.id2label
        self._injection_id = 1
        self._safe_id = 0
        for lid, lname in id2label.items():
            if str(lname).upper() == 'INJECTION':
                self._injection_id = int(lid)
            elif str(lname).upper() == 'SAFE':
                self._safe_id = int(lid)

        # Freeze embedding layer + first N encoder layers
        if freeze_n_layers > 0:
            for param in self.model.deberta.embeddings.parameters():
                param.requires_grad = False
            num_layers = len(self.model.deberta.encoder.layer)
            freeze_up_to = min(freeze_n_layers, num_layers)
            for i in range(freeze_up_to):
                for param in self.model.deberta.encoder.layer[i].parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print("    Trainable: {:,} / {:,} params ({:.1f}%)".format(
            trainable, total, 100.0 * trainable / total))

    def fit(self, texts: List[str], labels: List[int]):
        """Fine-tune the model on target training data."""
        import torch
        import random

        # Map our labels (0=safe,1=injection) to model label convention
        if self._injection_id == 1:
            mapped_labels = list(labels)
        else:
            mapped_labels = [1 - l for l in labels]

        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=0.01
        )

        n = len(texts)
        indices = list(range(n))

        for epoch in range(self.epochs):
            random.shuffle(indices)
            total_loss = 0.0
            num_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_texts = [texts[j] for j in batch_idx]
                batch_labels = torch.tensor(
                    [mapped_labels[j] for j in batch_idx], dtype=torch.long
                )

                encodings = self.tokenizer(
                    batch_texts, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors='pt'
                )

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    labels=batch_labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print("      Epoch {}/{}, Loss: {:.4f}".format(
                epoch + 1, self.epochs, avg_loss))

        self.model.eval()
        self._fitted = True

    def detect(self, text: str) -> DetectionResult:
        import torch

        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        start = time.perf_counter()
        encodings = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=self.max_length, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        injection_score = float(probs[self._injection_id])
        predicted = 1 if injection_score >= 0.5 else 0
        elapsed = (time.perf_counter() - start) * 1000

        return DetectionResult(
            predicted_label=predicted,
            confidence=float(max(probs)),
            raw_score=injection_score,
            latency_ms=elapsed
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        import torch

        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        results = []
        start = time.perf_counter()

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            encodings = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_length, return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1)

            for p in probs:
                injection_score = float(p[self._injection_id])
                predicted = 1 if injection_score >= 0.5 else 0
                results.append(DetectionResult(
                    predicted_label=predicted,
                    confidence=float(max(p)),
                    raw_score=injection_score,
                    latency_ms=0
                ))

        elapsed = (time.perf_counter() - start) * 1000
        per_item = elapsed / len(texts) if texts else 0
        for r in results:
            r.latency_ms = per_item
        return results


class EnsembleDetector:
    """
    Weighted ensemble of TF-IDF and DeBERTa detectors.

    Covers complementary failure modes:
    - TF-IDF: high precision on lexical patterns, fast inference
    - DeBERTa: high AUROC on semantic patterns, contextual understanding

    Combines raw injection scores via weighted average.
    """

    def __init__(self, tfidf_detector, deberta_detector,
                 tfidf_weight: float = 0.4, deberta_weight: float = 0.6,
                 threshold: float = 0.5):
        self.tfidf = tfidf_detector
        self.deberta = deberta_detector
        self.tfidf_weight = tfidf_weight
        self.deberta_weight = deberta_weight
        self.threshold = threshold

    def detect(self, text: str) -> DetectionResult:
        start = time.perf_counter()
        r_tfidf = self.tfidf.detect(text)
        r_deberta = self.deberta.detect(text)

        combined_score = (
            self.tfidf_weight * r_tfidf.raw_score +
            self.deberta_weight * r_deberta.raw_score
        )
        predicted = 1 if combined_score >= self.threshold else 0
        elapsed = (time.perf_counter() - start) * 1000

        return DetectionResult(
            predicted_label=predicted,
            confidence=combined_score if predicted == 1 else (1.0 - combined_score),
            raw_score=combined_score,
            latency_ms=elapsed
        )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        results = []
        start = time.perf_counter()

        tfidf_results = self.tfidf.detect_batch(texts)
        deberta_results = self.deberta.detect_batch(texts)

        for r_t, r_d in zip(tfidf_results, deberta_results):
            combined_score = (
                self.tfidf_weight * r_t.raw_score +
                self.deberta_weight * r_d.raw_score
            )
            predicted = 1 if combined_score >= self.threshold else 0
            results.append(DetectionResult(
                predicted_label=predicted,
                confidence=combined_score if predicted == 1 else (1.0 - combined_score),
                raw_score=combined_score,
                latency_ms=r_t.latency_ms + r_d.latency_ms
            ))

        elapsed = (time.perf_counter() - start) * 1000
        per_item = elapsed / len(texts) if texts else 0
        for r in results:
            r.latency_ms = per_item
        return results
