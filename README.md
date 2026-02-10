# A Reference Monitor for LLM Agent Memory: Provenance-Based Write Integrity with Empirical Detection Analysis

---

**Abstract.** LLM agents that maintain persistent memory (e.g., `SOUL.md`, `CLAUDE.md`) are vulnerable to *persistent prompt injection* — attacks that permanently alter agent behaviour by writing malicious content into long-term storage. We formalize a **Memory Integrity Theorem** that guarantees *write integrity* for the persistence layer: under a trusted provenance-tracking runtime, untrusted inputs cannot modify protected memory items or promote data across session namespaces. Our approach applies the classical *reference monitor* concept from OS security to the specific problem of agent memory writes — it governs what gets persisted, not what the model reasons about. The provenance verifier blocks all 199 injection payloads and 7 canonical attack vectors in our test suite, with no effect on inference-layer utility, because it interposes only on the write path. We complement the structural guarantee with a detection-layer analysis across 5 classifiers and 3 public datasets (1,028 test examples), finding that fine-tuning DeBERTa-v3 on target data closes a 41-percentage-point F1 gap and that an ensemble of TF-IDF and fine-tuned DeBERTa achieves AUROC up to 99.63%. We discuss the relationship to inference-layer defenses (e.g., AgentDojo), clarify what the theorem does and does not guarantee, and identify the USER-channel trust boundary as the critical open problem for future work.

---

## 1. Introduction

Large language model agents increasingly maintain persistent state through memory files, tool configurations, and long-term identity documents. Systems such as Cursor, AutoGPT, and Claude Code read and update memory files (e.g., `SOUL.md`, `CLAUDE.md`) on every interaction, enabling continuity across sessions. This architectural pattern creates a specific attack surface: **persistent prompt injection**, where an attacker writes malicious instructions into the agent's long-term memory to achieve permanent behavioural compromise that survives restarts and session boundaries [1, 6].

Unlike ephemeral prompt injection — which affects a single conversation turn — persistent injection targets the agent's storage layer. Researchers have demonstrated attacks where a URL contains hidden instructions that cause the agent to modify its own identity file, and scheduled reinjection attacks where the agent creates cron jobs that periodically overwrite its memory [1, 6]. Cross-session memory leakage compounds this threat: misconfigured caches allow data from one session to appear in another [4].

**Scope and positioning.** This work addresses a *specific sub-problem* of agent security: write integrity for the persistence layer. We do not claim to solve prompt injection generally. Inference-layer attacks — where the model is manipulated within a single turn without touching persistent memory — are outside our scope and require complementary defenses [1, 3]. Our contribution is analogous to a *reference monitor* in operating systems: a small, verifiable component that mediates access to a specific protected resource (here, the memory store), without interfering with the broader computation (here, model inference).

**Contributions.** This paper makes the following contributions:

1. **Formal model (Section 3).** We define a small-step operational semantics for agent memory operations, with explicit state transitions and a provenance-tracking runtime that operates independently of the LLM's outputs.

2. **Memory Integrity Theorem (Section 4).** Under five explicit assumptions — including a *trusted enforcement boundary* between the runtime and the LLM — we prove that protected memory items cannot be modified by untrusted inputs. We scope the guarantee to write integrity (not read confidentiality) and to namespace-keyed stores (not vector indices).

3. **Structural defense evaluation (Section 5).** We test the provenance verifier against 199 real injection payloads from public datasets and 7 programmatic attack vectors. All are blocked. We frame this as a *verification of implementation correctness*, not as a claim of adversarial robustness against novel attack strategies.

4. **Detection layer analysis (Section 6).** We evaluate 5 detectors across 3 public datasets (1,028 test examples), finding that domain adaptation closes large performance gaps and that ensembling improves calibration.

**What this work does NOT claim.** We do not claim general agent security. We do not claim robustness against attacks through the USER channel (Section 9.5). We do not claim read confidentiality. We do not claim our detection results constitute state-of-the-art — they are situated within our specific evaluation protocol and may not generalize to other datasets or annotation conventions.

---

## 2. Related Work

**Prompt injection attacks.** Perez and Ribeiro [7] first demonstrated LLM susceptibility to prompt injection. Greshake et al. [8] extended this to *indirect* prompt injection through tool outputs and retrieved documents. Liu et al. [3] provided the first formal benchmark with standardized evaluation. These works address the *inference layer* — the model is manipulated during a single turn.

**Detection-based defenses.** Jain et al. [5] established baseline defenses including perplexity filtering and RoBERTa-based classifiers (~88% F1 on jailbreak data). ProtectAI released DeBERTa-v3 classifiers achieving 96.4% F1 on deepset/prompt-injections [9]. Yi et al. [4] benchmarked indirect injection (BIPIA), finding GPT-4 with border strings still allows 21.8% attack success. All detection-based approaches have irreducible error rates — our detection experiments confirm this.

**Agent-level defenses.** Debenedetti et al. [1] introduced AgentDojo, a dynamic benchmark for evaluating prompt injection defenses in agentic settings. Their key finding: every defense that improves security degrades utility. Spotlighting achieves 84.2% security but drops utility by 25pp. **Crucially, AgentDojo evaluates inference-layer defenses against inference-layer attacks.** Our work addresses a different threat surface (the persistence layer), making direct numerical comparison inappropriate (Section 7.3).

**Information flow control.** Taint tracking and mandatory access control have been studied extensively in operating systems [10] and programming languages [11]. Denning's lattice model [12] provides the theoretical foundation for tracking information flow through security labels. Our work applies these principles to the specific problem of LLM agent memory writes.

**Gap addressed.** Prior work either (a) treats prompt injection as a detection/classification problem with irreducible errors, or (b) defends the inference layer at a utility cost. We observe that the *persistence layer* — the write path to long-term memory — is a narrow, well-defined interface where a reference monitor can provide a deterministic structural guarantee without affecting inference. This is a complementary defense, not a replacement for inference-layer protections.

---

## 3. Formal Model

### 3.1 Agent State

We model the agent state at time $t$ as a tuple:

$$S_t = (P_t, M_t, B_t, G_t)$$

where $P_t$ is the current prompt context, $M_t$ is the persistent memory store (a set of key-value items with metadata), $B_t$ is the belief state (internal representation nodes), and $G_t$ is the goal set.

### 3.2 Memory Items

Each memory item $m \in M_t$ is a record:

$$m = (\mathsf{key}, \mathsf{val}, \pi, \tau, \iota, \sigma)$$

where:
- $\mathsf{key}$: the storage key (e.g., `"SOUL.md"`)
- $\mathsf{val}$: the stored content
- $\pi \in \{\mathsf{TRUSTED}, \mathsf{UNTRUSTED}\}$: the provenance label
- $\tau \in \{0, 1\}$: the taint bit, where $\tau = 1$ if any input in the causal chain originated from an untrusted source
- $\iota \in \{0, 1\}$: the immutability flag
- $\sigma$: the session namespace that owns this item

**Provenance labels.** We use a binary trust model: data is either $\mathsf{TRUSTED}$ (originating from the system configuration or authenticated user input through a verified channel) or $\mathsf{UNTRUSTED}$ (originating from web content, tool outputs, skill invocations, or any source not authenticated as $\mathsf{TRUSTED}$). For implementation specificity, we further tag untrusted sources with origin labels ($\mathsf{WEB}$, $\mathsf{TOOL}$, $\mathsf{SKILL}$) for audit logging, but the security decision is binary.

**Design choice: binary vs. lattice.** We deliberately chose a binary trust model over a richer lattice (e.g., $\mathsf{SYS} > \mathsf{USER} > \mathsf{SKILL} > \mathsf{TOOL} > \mathsf{WEB}$) for two reasons: (1) in practice, the critical boundary is between human-authenticated and machine-generated input — intermediate gradations add complexity without clear security benefit in the memory-write setting; (2) a binary model is easier to verify and audit. Systems that need finer-grained policies can extend the model, but the core theorem requires only the binary distinction.

### 3.3 Operational Semantics (Small-Step)

We define the agent's interaction with memory as a transition system. The agent runtime processes a sequence of *actions*; we focus on the actions relevant to memory:

**Actions:**

$$a ::= \mathsf{READ}(k, \sigma) \mid \mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma) \mid \mathsf{PROMOTE}(k, \sigma_{\text{src}}, \sigma_{\text{shared}}) \mid \mathsf{INTERNAL}(\ldots)$$

- $\mathsf{READ}(k, \sigma)$: Read key $k$ from namespace $\sigma$. Returns $m.\mathsf{val}$ if $m.\mathsf{key} = k$ and $m.\sigma = \sigma$, else $\bot$.
- $\mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)$: Propose writing value $v$ to key $k$ in namespace $\sigma$, with dependency set $\mathsf{deps} \subseteq B_t$ and declared provenance $\pi$.
- $\mathsf{PROMOTE}(k, \sigma_{\text{src}}, \sigma_{\text{shared}})$: Propose promoting key $k$ from session namespace $\sigma_{\text{src}}$ to shared namespace $\sigma_{\text{shared}}$.
- $\mathsf{INTERNAL}(\ldots)$: Any action that does not touch the memory store (inference, tool calls, response generation). The reference monitor does not interpose on these.

**Transition rules.** The state transition $S_t \xrightarrow{a} S_{t+1}$ is defined by:

**Rule 1 (Read):**
$$\frac{m \in M_t \quad m.\mathsf{key} = k \quad m.\sigma = \sigma}{S_t \xrightarrow{\mathsf{READ}(k, \sigma)} S_t[B_{t+1} := B_t \cup \{b_m\}]}$$
where $b_m$ is a belief node carrying $m.\tau$ (taint propagates to beliefs).

**Rule 2 (Write — accepted):**
$$\frac{\iota(k) = 0 \quad \forall b \in \mathsf{deps}: \tau(b) = 0 \quad \pi = \mathsf{TRUSTED}}{S_t \xrightarrow{\mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)} S_t[M_{t+1} := M_t[k \mapsto (k, v, \pi, 0, \iota(k), \sigma)]]}$$

**Rule 3 (Write — rejected, immutable):**
$$\frac{\iota(k) = 1}{S_t \xrightarrow{\mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)} S_t} \quad \text{(no state change)}$$

**Rule 4 (Write — rejected, tainted):**
$$\frac{\exists b \in \mathsf{deps}: \tau(b) = 1}{S_t \xrightarrow{\mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)} S_t} \quad \text{(no state change)}$$

**Rule 5 (Write — rejected, untrusted principal):**
$$\frac{\pi = \mathsf{UNTRUSTED}}{S_t \xrightarrow{\mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)} S_t} \quad \text{(no state change)}$$

**Rule 6 (Promote — accepted):**
$$\frac{\pi_{\text{authorizer}} = \mathsf{TRUSTED} \quad \tau(m) = 0}{S_t \xrightarrow{\mathsf{PROMOTE}(k, \sigma_s, \sigma_{\text{shared}})} S_t[M_{t+1} := M_t \cup \{m[\sigma := \sigma_{\text{shared}}]\}]}$$

**Rule 7 (Promote — rejected):**
$$\frac{\pi_{\text{authorizer}} = \mathsf{UNTRUSTED} \lor \tau(m) = 1}{S_t \xrightarrow{\mathsf{PROMOTE}(k, \sigma_s, \sigma_{\text{shared}})} S_t} \quad \text{(no state change)}$$

**Rule 8 (Taint propagation):**
$$\tau(b) = \max_{d \in \mathsf{ancestors}(b)} \tau(d)$$
Taint is monotonically increasing: once any ancestor is tainted, all descendants are tainted. There is no operation that removes taint — this is a deliberate design choice (no "declassification" without explicit TRUSTED authorization).

**Rule 9 (Namespace scoping):**
$$\mathsf{visible}(\sigma_s) = \{m \in M_t : m.\sigma = \sigma_s\} \cup \{m \in M_t : m.\sigma = \sigma_{\text{shared}}\}$$

### 3.4 Enforcement Boundary

**Critical assumption.** The provenance labels ($\pi$, $\tau$) and the enforcement of Rules 2–7 are implemented by a *trusted runtime* that operates independently of the LLM. The LLM generates candidate actions; the runtime assigns provenance labels based on the actual data source (not the LLM's self-report) and enforces the transition rules before committing state changes.

This is analogous to how an OS kernel enforces file permissions regardless of what a userspace process claims about its own privilege level. The LLM is the "userspace process" — it can request any write, but the runtime decides whether to allow it based on structural properties (provenance, taint, immutability) that the LLM cannot forge.

**What the runtime must correctly implement:**
1. **Source labeling.** Every data item entering the system is labeled with its true origin (web retrieval → $\mathsf{UNTRUSTED/WEB}$, tool output → $\mathsf{UNTRUSTED/TOOL}$, authenticated user input → $\mathsf{TRUSTED}$, system config → $\mathsf{TRUSTED}$).
2. **Taint propagation.** When the LLM produces output that depends on tainted inputs, the runtime marks the output as tainted. This requires tracking dataflow through the LLM's context window — a conservative approach is to taint any output produced in a context that contains tainted data.
3. **Rule enforcement.** Before any write to persistent memory commits, the runtime checks Rules 2–7.

**What the runtime does NOT need to do:** The runtime does not need to understand the *content* of data — it only tracks structural metadata (provenance labels, taint bits, immutability flags, namespace membership). This makes it simple enough to audit and verify.

### 3.5 Threat Model

**Adversary capabilities.** The adversary can inject arbitrary content through:
- Web pages, emails, or documents retrieved by the agent ($\mathsf{UNTRUSTED/WEB}$)
- Tool outputs from external APIs ($\mathsf{UNTRUSTED/TOOL}$)
- Skill/plugin invocations ($\mathsf{UNTRUSTED/SKILL}$)

**Adversary goals:**
1. Modify protected memory items (persistent injection)
2. Exfiltrate data across session boundaries (cross-session leakage via writes)
3. Establish persistent backdoors through scheduled reinjection

**Explicit exclusions.** The adversary **cannot**:
- Modify the trusted runtime or its configuration ($\mathsf{TRUSTED}$ code is in the TCB)
- Bypass the enforcement boundary (e.g., write directly to disk circumventing the runtime)
- Impersonate authenticated user actions through the $\mathsf{USER}$ channel (see Section 9.5 for discussion of this critical assumption)

**What the threat model does NOT cover:**
- **Inference-layer attacks.** An adversary who manipulates the LLM's reasoning within a single turn (without targeting persistent memory) is outside scope.
- **Read confidentiality.** We guarantee that untrusted inputs cannot *write* to protected memory; we do not guarantee they cannot *read* or exfiltrate data already in context. Read confidentiality requires additional mechanisms (e.g., output filtering).
- **Availability.** A denial-of-service attack (flooding the verifier with write proposals) is outside scope.

### 3.6 Attack Taxonomy

We consider 7 canonical attack vectors:

| # | Attack | Source | Mechanism |
|---|--------|--------|-----------|
| A1 | SOUL.md Direct Modification | WEB | Hidden instruction to overwrite identity file |
| A2 | Hidden Instruction Injection | WEB | Injected instruction via retrieved document |
| A3 | Scheduled Reinjection | SKILL | Create cron job to periodically rewrite memory |
| A4 | Tool Output Poisoning | TOOL | Malicious payload in API response |
| A5 | Cross-Session Data Leak | WEB | Access secrets from another session's namespace |
| A6 | Dependency Chain Poisoning | TOOL | Taint laundering through intermediate belief nodes |
| A7 | Taint Washing | SKILL | Attempt to remove taint via trusted-looking intermediary |

---

## 4. Memory Integrity Theorem

### 4.1 Assumptions

We state the theorem under the following explicit assumptions:

> **A1 (Trusted runtime).** The provenance-tracking runtime correctly implements source labeling, taint propagation, and rule enforcement (Section 3.4). The runtime is in the trusted computing base (TCB).
>
> **A2 (Faithful source labeling).** Every data item entering the system receives a provenance label that accurately reflects its true origin. In particular, web-retrieved content, tool outputs, and skill results are labeled $\mathsf{UNTRUSTED}$.
>
> **A3 (Conservative taint propagation).** Any LLM output produced in a context containing tainted data is itself tainted. Taint is monotonically increasing — no operation removes taint without explicit $\mathsf{TRUSTED}$ authorization.
>
> **A4 (Namespace-keyed store).** The memory store $M_t$ is implemented as a namespace-keyed store (e.g., key-value database, file system with directory isolation). Vector stores with approximate nearest-neighbour retrieval require additional partitioning mechanisms not covered by this theorem.
>
> **A5 (No USER-channel compromise).** The adversary cannot inject content through the authenticated $\mathsf{USER}$ channel. Social engineering attacks that trick the human user into pasting malicious content are outside scope (see Section 9.5).

### 4.2 Theorem Statement

**Theorem 1 (Memory Write Integrity).** *Under Assumptions A1–A5, for any agent execution $S_0 \xrightarrow{a_1} S_1 \xrightarrow{a_2} \cdots \xrightarrow{a_n} S_n$ where each $a_i$ is governed by Rules 1–9 of Section 3.3:*

**(Part I — Immutability.)** *No immutable memory item is modified at any step:*
$$\forall t \in [0, n],\ \forall m \in M_t:\ \iota(m) = 1 \implies m.\mathsf{val}_t = m.\mathsf{val}_0$$

**(Part II — Taint blocking.)** *No write whose dependency set contains tainted data succeeds:*
$$\forall t,\ \text{if } a_t = \mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma) \text{ and } \exists b \in \mathsf{deps}: \tau(b) = 1, \text{ then } M_{t+1} = M_t$$

**(Part III — Session isolation.)** *No data written in session namespace $\sigma_i$ becomes visible in session namespace $\sigma_j$ ($i \neq j$) without $\mathsf{TRUSTED}$ promotion:*
$$\forall t,\ \forall m \in M_t:\ m.\sigma = \sigma_i \implies m \notin \mathsf{visible}(\sigma_j) \quad \text{for } j \neq i$$

### 4.3 Proof

**Part I.** By induction on the number of transitions. *Base case:* At $t = 0$, no action has been taken; immutable items have their initial values. *Inductive step:* Assume all immutable items are unchanged at step $t$. At step $t+1$, the only action that could modify a memory item is $\mathsf{WRITE}$. If $a_{t+1} = \mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)$ and $\iota(k) = 1$, then Rule 3 applies: the transition produces $S_{t+1} = S_t$ with no memory change. For $\mathsf{PROMOTE}$, immutability flags are preserved. For all other actions ($\mathsf{READ}$, $\mathsf{INTERNAL}$), $M_{t+1} = M_t$. Therefore immutable items remain unchanged at $t+1$.

**Part II.** Suppose $a_t = \mathsf{WRITE}(k, v, \mathsf{deps}, \pi, \sigma)$ with $\exists b \in \mathsf{deps}: \tau(b) = 1$. By Rule 4, the transition produces $S_t$ (no state change), so $M_{t+1} = M_t$. The key non-trivial property is that taint cannot be removed: by Rule 8, $\tau(b) = \max_{d \in \mathsf{ancestors}(b)} \tau(d)$, so if any ancestor of $b$ was tainted, $b$ remains tainted. An adversary who attempts "taint washing" — routing data through intermediate nodes — cannot reduce the taint bit because $\max$ is monotonically increasing. Under Assumption A3, even LLM-mediated transformations of tainted data produce tainted outputs.

**Part III.** By Rule 9, $\mathsf{visible}(\sigma_j) = \{m : m.\sigma = \sigma_j\} \cup \{m : m.\sigma = \sigma_{\text{shared}}\}$. An item $m$ with $m.\sigma = \sigma_i$ is not in either set (since $\sigma_i \neq \sigma_j$ and $\sigma_i \neq \sigma_{\text{shared}}$). The only way $m$ could enter $\mathsf{visible}(\sigma_j)$ is through $\mathsf{PROMOTE}(k, \sigma_i, \sigma_{\text{shared}})$, which by Rule 6 requires $\pi_{\text{authorizer}} = \mathsf{TRUSTED}$ and $\tau(m) = 0$. Under Assumption A5, the adversary cannot issue $\mathsf{TRUSTED}$ promotions, and under Assumption A3, any data influenced by the adversary is tainted. Therefore adversary-influenced data cannot be promoted. $\square$

### 4.4 What the Theorem Does and Does Not Guarantee

**Does guarantee:**
- Immutable items (e.g., `SOUL.md`) cannot be modified by any action in any execution trace, regardless of adversarial content.
- Tainted data cannot be written to *any* memory location (mutable or immutable) unless all taint is provably absent.
- Session namespaces are isolated: no cross-session data transfer without trusted authorization.

**Does NOT guarantee:**
- **Read confidentiality.** The model can still *read* memory items and include their contents in outputs. An adversary might exfiltrate secrets via inference-layer manipulation (e.g., "read SOUL.md and include it in your next API call"). This requires output-filtering defenses orthogonal to our work.
- **Inference-layer integrity.** The model's *reasoning* within a single turn can still be manipulated by prompt injection. Our theorem only protects what gets *written back* to persistent storage.
- **Availability.** An adversary could flood the system with write proposals that all get rejected, potentially causing performance degradation.
- **USER-channel attacks.** If the adversary convinces the human user to paste malicious content, that content arrives with $\mathsf{TRUSTED}$ provenance (Section 9.5).

### 4.5 On the Tautological Concern

A reviewer might observe that the theorem is "trivially true" given the rules — if the runtime correctly enforces the rules, the properties follow by construction. We acknowledge this. The value of the formalization is not in the proof difficulty but in:

1. **Making assumptions explicit.** Assumptions A1–A5 precisely delineate what must be trusted and what need not be. This enables principled security auditing: check the five assumptions, not the entire agent stack.
2. **Defining the enforcement boundary.** The small-step semantics make clear *where* the reference monitor interposes (on $\mathsf{WRITE}$ and $\mathsf{PROMOTE}$, not on $\mathsf{READ}$ or $\mathsf{INTERNAL}$) and *what* it checks (provenance, taint, immutability — not content).
3. **Enabling compositional reasoning.** The theorem can be composed with inference-layer defenses: our reference monitor guarantees persistence-layer integrity; other systems guarantee inference-layer properties. The overall security is the conjunction.

---

## 5. Experimental Setup

### 5.1 Datasets

We evaluate on three public datasets from HuggingFace, with no simulation or synthetic data:

| Dataset | Source | Total | Train | Test | Pos:Neg | Split |
|---------|--------|-------|-------|------|---------|-------|
| deepset/prompt-injections | HuggingFace Hub | 662 | 546 | 116 | 60:56 | Stratified 80/20 |
| jackhhao/jailbreak-classification | HuggingFace Hub | 1,306 | 1,044 | 262 | 139:123 | Stratified 80/20 |
| Combined corpus | 4 HF datasets | 3,250 | 2,600 | 650 | 202:448 | Stratified 80/20 |

The combined corpus aggregates deepset/prompt-injections (662), jackhhao/jailbreak-classification (1,306), rubend18/ChatGPT-Jailbreak-Prompts (79 jailbreak examples), and fka/awesome-chatgpt-prompts (1,203 benign examples). All splits use `sklearn.model_selection.train_test_split` with `stratify=y, random_state=42`. No data augmentation or oversampling is applied. Text preprocessing: whitespace normalization only; no lowercasing, stemming, or stop-word removal.

### 5.2 Detectors

We evaluate 5 detectors spanning the complexity spectrum:

**D1: HeuristicDetector (rule-based baseline).** 40+ regex patterns across 5 categories: instruction override, role hijack, exfiltration, memory attack, and encoding evasion. Each pattern group carries a weight; the final score is the maximum weighted match. No training required.

**D2: TF-IDF + Logistic Regression.** Character n-gram TF-IDF vectorization (n=1..4, max 15,000 features, sublinear TF) followed by balanced Logistic Regression (L-BFGS solver, balanced class weights, C=1.0, `random_state=42`). Trained on each dataset's training split.

**D3: DeBERTa-v3 pre-trained (zero-shot transfer).** `protectai/deberta-v3-base-prompt-injection-v2` (184M parameters) applied directly to target data without fine-tuning. Represents the common deployment pattern of using an off-the-shelf classifier on a new distribution.

**D4: DeBERTa-v3 fine-tuned (domain-adapted).** Same architecture, adapted to each dataset's training split. We freeze the embedding layer and first 9 of 12 encoder layers, training only the last 3 encoder layers + classification head (22.25M / 184.42M = 12.1% trainable parameters). Hyperparameters: 3 epochs (deepset, jailbreak) or 2 epochs (combined); batch size 8; learning rate 2e-5; AdamW optimizer (weight decay 0.01); gradient clipping at norm 1.0; `torch.manual_seed(42)`.

**D5: Ensemble.** Weighted average of D2 and D4 scores: $s = 0.4 \cdot s_{\text{TF-IDF}} + 0.6 \cdot s_{\text{DeBERTa-FT}}$, threshold 0.5.

### 5.3 Defense Evaluation

**Provenance verifier.** We test the structural defense (Section 3.3) against:
- 199 real injection payloads extracted from test sets (all examples with ground-truth label = injection)
- 7 programmatic attack vectors from the taxonomy (Section 3.6)
- Cross-session isolation across 50 concurrent sessions (2,450 pair-wise namespace checks)

This evaluation confirms that the verifier *implementation* correctly enforces the formal rules. It does not constitute an adversarial robustness evaluation against novel attack strategies — the verifier's security depends on Assumptions A1–A5, not on the diversity of the test payloads.

### 5.4 Metrics

**Detection:** accuracy, precision, recall, F1-score, AUROC (via `sklearn.metrics`, threshold 0.5 for binary predictions). **Defense:** block rate (proportion of injection payloads rejected), memory integrity preservation (SHA-256 hash comparison of `SOUL.md` before and after), cross-session leak rate.

### 5.5 Published Baselines

| Method | Source | Metric | Value | Notes |
|--------|--------|--------|-------|-------|
| deepset/deberta-v3-base-injection | HF model card | F1 on deepset | 99.40% | Trained and evaluated on same dataset |
| protectai/deberta-v3 v1 | HF model card | F1 on deepset | 96.40% | Cross-dataset evaluation |
| RoBERTa classifier [5] | Jain et al. 2023 | F1 on jailbreak | ~88.0% | Approximate; varies by prompt type |
| GPT-4o + spotlighting [1] | NeurIPS 2024 | Security rate | 84.2% | Inference-layer defense |
| GPT-4 + border strings [4] | Yi et al. 2023 | Attack success | 21.8% | Inference-layer defense |

---

## 6. Results

### 6.1 Detection Performance

**Table 1. deepset/prompt-injections (n=116 test, 60 positive, 56 negative).**

| Detector | Accuracy | Precision | Recall | F1 | AUROC | Latency |
|----------|----------|-----------|--------|----|-------|---------|
| D1: Heuristic | 50.86% | 100.00% | 5.00% | 9.52% | 52.50% | 0.07 ms |
| D2: TF-IDF+LR | 90.52% | 98.04% | 83.33% | 90.09% | 97.44% | 0.19 ms |
| D3: DeBERTa pretrained | 67.24% | 100.00% | 36.67% | 53.66% | 89.57% | 375.5 ms |
| D4: DeBERTa fine-tuned | 94.83% | 100.00% | 90.00% | 94.74% | 97.74% | 22.5 ms |
| D5: Ensemble | 94.83% | 100.00% | 90.00% | 94.74% | 99.14% | 22.2 ms |

**Table 2. jackhhao/jailbreak-classification (n=262 test, 139 positive, 123 negative).**

| Detector | Accuracy | Precision | Recall | F1 | AUROC | Latency |
|----------|----------|-----------|--------|----|-------|---------|
| D1: Heuristic | 61.07% | 79.37% | 35.97% | 49.50% | 62.75% | 0.78 ms |
| D2: TF-IDF+LR | 96.56% | 99.24% | 94.24% | 96.68% | 99.51% | 1.78 ms |
| D3: DeBERTa pretrained | 90.84% | 98.32% | 84.17% | 90.70% | 97.91% | 513.7 ms |
| D4: DeBERTa fine-tuned | 97.33% | 97.14% | 97.84% | 97.49% | 98.87% | 55.1 ms |
| D5: Ensemble | 97.33% | 97.14% | 97.84% | 97.49% | 99.63% | 55.8 ms |

**Table 3. Combined corpus (n=650 test, 202 positive, 448 negative).**

| Detector | Accuracy | Precision | Recall | F1 | AUROC | Latency |
|----------|----------|-----------|--------|----|-------|---------|
| D1: Heuristic | 58.92% | 33.33% | 32.18% | 32.75% | 51.85% | 0.68 ms |
| D2: TF-IDF+LR | 95.54% | 95.29% | 90.10% | 92.62% | 98.26% | 1.35 ms |
| D3: DeBERTa pretrained | 92.00% | 96.88% | 76.73% | 85.64% | 92.62% | 228.0 ms |
| D4: DeBERTa fine-tuned | 96.31% | 97.85% | 90.10% | 93.81% | 98.18% | 56.7 ms |
| D5: Ensemble | 96.62% | 97.87% | 91.09% | 94.36% | 98.75% | 59.0 ms |

### 6.2 Impact of Fine-Tuning

**Table 4. DeBERTa-v3 F1 before and after fine-tuning.**

| Dataset | Pre-trained F1 | Fine-tuned F1 | $\Delta$ F1 | Recall gain | Time (CPU) |
|---------|---------------|---------------|-------------|-------------|------------|
| deepset | 53.66% | 94.74% | +41.08pp | 36.67% → 90.00% | 154s |
| jailbreak | 90.70% | 97.49% | +6.79pp | 84.17% → 97.84% | 799s |
| combined | 85.64% | 93.81% | +8.17pp | 76.73% → 90.10% | 1317s |

The deepset improvement is striking: the pre-trained model missed 38 of 60 injections (recall 36.67%) due to distribution mismatch. After fine-tuning 12.1% of parameters for 3 epochs, recall reaches 90.00% with zero false positives.

### 6.3 Comparison with Published Results

**Table 5. Our results in context of published baselines.**

| Method | Dataset | F1 | Context |
|--------|---------|-----|---------|
| DeBERTa fine-tuned (ours) | deepset | 94.74% | 1.66pp below protectai v1 (96.40%) |
| deepset self-evaluation | deepset | 99.40% | Trained on full dataset, self-evaluation |
| protectai/deberta-v3 v1 | deepset | 96.40% | Full model, different training procedure |
| DeBERTa fine-tuned (ours) | jailbreak | 97.49% | +9.49pp above Jain et al. RoBERTa (~88%) |
| Ensemble (ours) | jailbreak | 97.49% (AUROC 99.63%) | Highest AUROC in our evaluation |

**Caveats on comparison.** These numbers are not directly comparable across studies due to differences in: (a) train/test splits — deepset's self-evaluation uses the full dataset; (b) preprocessing — Jain et al. [5] use different tokenization and prompt formatting; (c) model capacity — we freeze 75% of encoder layers for efficiency, while baselines use full models. We report these comparisons for *context*, not to claim superiority. Within our own evaluation protocol (same splits, same preprocessing, same hardware), fine-tuning and ensembling consistently improve performance.

### 6.4 Defense Verification

**Table 6. Provenance verifier results.**

| Test | Result |
|------|--------|
| Injection payloads blocked | 199/199 (100%) |
| Attack vectors blocked (7 canonical) | 7/7 (100%) |
| Cross-session isolation (50 sessions, 2,450 pair-checks) | 0 leaks |
| SOUL.md integrity (SHA-256 pre vs. post) | Identical |
| Memory items modified by untrusted input | 0 |
| Theorem 1 holds (for this execution) | True |

**Interpretation.** The 199/199 block rate confirms that the verifier implementation correctly enforces the formal rules against real injection payloads. This is a *correctness test*, not an *adversarial robustness evaluation*. The payloads were drawn from public datasets and were not crafted to target our specific defense. The verifier's security guarantee comes from the formal rules (Section 3.3) and the trusted runtime assumption (A1), not from the diversity of the test suite. An adversary who satisfies our threat model — i.e., operates through untrusted channels — cannot write to protected memory regardless of payload content, because the check is structural (provenance and taint), not content-based.

**Table 7. Attack vector breakdown.**

| Attack | Type | Blocked | Mechanism |
|--------|------|---------|-----------|
| A1: SOUL.md Direct Modification | Persistent injection | Yes | Immutability (Rule 3) |
| A2: Hidden Instruction Injection | Indirect injection | Yes | WEB taint (Rule 4) |
| A3: Scheduled Reinjection | Persistent backdoor | Yes | SKILL taint (Rule 4) |
| A4: Tool Output Poisoning | Indirect injection | Yes | Immutability + TOOL taint (Rules 3, 4) |
| A5: Cross-Session Data Leak | Session breach | Yes | Namespace isolation (Rule 9) |
| A6: Dependency Chain Poisoning | Taint propagation | Yes | Transitive taint (Rule 8) |
| A7: Taint Washing | Privilege escalation | Yes | No trusted justification (Rule 5) |

---

## 7. Analysis and Discussion

### 7.1 Why Detection Alone Is Insufficient

Even our best detector (D4, F1=97.49% on jailbreak) has residual errors:
- deepset: 6 false negatives (6 missed injections out of 60)
- jailbreak: 3 false negatives (3 missed out of 139)
- combined: 20 false negatives (20 missed out of 202)

In a detection-only architecture, each missed injection is a potential successful attack on persistent memory. At scale (thousands of interactions per day), even a 2.5% miss rate translates to dozens of potential memory poisoning events. The provenance verifier catches all of these because it checks *structural properties* (provenance, taint, immutability) rather than *content* — it doesn't need to understand whether a payload is malicious, only whether the write proposal has trusted provenance.

### 7.2 Distribution Mismatch Drives Zero-Shot Failure

TF-IDF+LR (F1=90.09%) outperforms the 184M-parameter DeBERTa (F1=53.66%) on deepset by 36.43pp in the zero-shot setting. This is not because TF-IDF is a better model — it's because distribution mismatch causes catastrophic recall failure. DeBERTa achieves 100% precision but only 36.67% recall: it recognizes injections from its training distribution but misses 63% of deepset's injections because they look different. After fine-tuning on deepset's training split, DeBERTa's recall jumps to 90.00% and it outperforms TF-IDF.

This finding is consistent with Liu et al. [3]: feature engineering can outperform large transformers when distribution alignment is poor. The practical implication is that deploying off-the-shelf injection classifiers on new distributions without adaptation is unreliable.

### 7.3 Relationship to AgentDojo

AgentDojo [1] evaluates *inference-layer* defenses against *inference-layer* attacks: the adversary manipulates the model's reasoning to produce incorrect tool calls within a single turn. Their finding — that every defense degrades utility — reflects a fundamental tension at the inference layer: making the model more suspicious of inputs necessarily makes it worse at following legitimate instructions.

Our defense operates at a *different layer* (persistence) against a *different threat* (memory writes). There is no utility trade-off because we do not modify how the model processes inputs — only what gets written to long-term storage. This architectural difference makes direct numerical comparison misleading:

- **AgentDojo's security rate** measures whether the model avoids making a malicious tool call within a conversation. Our verifier does not prevent malicious tool calls — it prevents malicious *memory writes*.
- **AgentDojo's utility** measures task completion across 97 diverse agent tasks. Our "utility" is narrower: the verifier does not interfere with inference, so inference-layer utility is unaffected by construction.

The appropriate framing is that these are *complementary defenses* for different layers of the agent stack. An agent could use AgentDojo-style defenses (spotlighting, tool-filtering) for inference-layer protection AND our provenance verifier for persistence-layer protection. The security properties compose.

### 7.4 Ensemble Calibration

The ensemble (D5) ties or exceeds D4 on F1 and consistently achieves the highest AUROC:
- deepset: 99.14% AUROC (vs. 97.74% for D4)
- jailbreak: 99.63% AUROC (vs. 98.87% for D4)
- combined: 98.75% AUROC (vs. 98.18% for D4)

The AUROC improvement indicates better-calibrated confidence scores from combining lexical (TF-IDF) and semantic (DeBERTa) signals. This is valuable for threshold tuning in deployment.

### 7.5 Fine-Tuning Efficiency

Fine-tuning 12.1% of DeBERTa parameters achieves near-full-model performance (our gap to protectai v1 is 1.66pp on deepset) while running in 154–1,317 seconds on CPU. This makes the approach practical without GPU infrastructure.

---

## 8. Limitations of the Formal Model

1. **Immutability and real workflows.** Marking identity files as immutable prevents *all* modification — including legitimate updates. Real agent workflows often need to update memory (e.g., learning user preferences). Our model supports mutable items with taint-based write control, but the "default immutable" posture requires explicit override for legitimate writes. Future work should explore *graduated mutability* with authenticated update channels.

2. **Conservative taint propagation.** Our taint model is monotonic: once tainted, always tainted. This is secure but restrictive — it prevents useful patterns like "the agent reads a web page, extracts a fact, and stores it." A *declassification* mechanism (where a trusted review process can remove taint from specific data) would enable richer workflows, but its design is non-trivial and we leave it to future work.

3. **Namespace-keyed stores only.** The session isolation guarantee (Part III) relies on namespace-keyed storage (key-value stores, file systems with directory isolation). Vector stores with approximate nearest-neighbour retrieval may return results across namespace boundaries if embeddings are stored in a shared index. Partition-by-namespace strategies for vector stores are future work.

4. **No formal verification of the runtime.** We assume the runtime correctly implements the rules (Assumption A1), but we have not formally verified the runtime implementation using a proof assistant (e.g., Coq, Lean). Our test suite (38 tests) provides empirical confidence, not formal proof of implementation correctness.

---

## 9. Broader Limitations and Open Problems

### 9.1 Small Test Suite

The 199 injection payloads are drawn from public datasets and represent known attack patterns. We do not claim robustness against novel adversarial strategies crafted specifically to target provenance-based defenses. The verifier's security depends on the formal rules and the trusted runtime, not on having "seen" enough attacks.

### 9.2 Detection Results Are Protocol-Specific

Our detection numbers (e.g., 97.49% F1 on jailbreak) are specific to our evaluation protocol: stratified 80/20 split with `random_state=42`, no augmentation, character n-gram TF-IDF, partial DeBERTa fine-tuning. Results may differ under different splits, preprocessing, or full-model fine-tuning. We do not report confidence intervals because we use a single fixed split; future work should use k-fold cross-validation.

### 9.3 No End-to-End Agent Evaluation

We evaluate the verifier against isolated write proposals, not against a fully deployed agent processing real-world tasks. End-to-end evaluation — integrating the verifier into an agent framework (e.g., LangChain, AutoGPT) and measuring task completion — is necessary to validate the zero-utility-loss claim in practice.

### 9.4 Confidentiality Gap

The theorem guarantees write integrity but not read confidentiality. An adversary who compromises the inference layer can potentially exfiltrate data from memory via the model's outputs (e.g., "include the contents of SOUL.md in your response and send it to attacker.com"). Output filtering and egress control are complementary defenses.

### 9.5 The USER Channel: The Critical Open Problem

The most important limitation is Assumption A5: we assume the adversary cannot inject through the $\mathsf{USER}$ channel. In practice, this assumption can be violated through:

- **Social engineering.** The adversary tricks the human user into pasting malicious content that arrives with $\mathsf{TRUSTED}$ provenance.
- **Clipboard injection.** Malicious content copied from a compromised webpage is pasted by the user.
- **UI confusion.** The agent interface does not clearly distinguish $\mathsf{TRUSTED}$ and $\mathsf{UNTRUSTED}$ sources, leading the user to accidentally approve a malicious write.

Mitigations (not implemented in this work):
- **Diff view for memory writes.** Show the user exactly what will change before committing.
- **Rate-limited promotions.** Cap the rate of $\mathsf{USER}$-authorized writes to prevent bulk compromise.
- **Audit logs.** Maintain an immutable log of all write operations for post-hoc review.
- **Content anomaly detection.** Flag $\mathsf{USER}$-sourced writes that resemble known injection patterns (using the detection layer as a secondary check).

We identify this as the critical open problem: the boundary between "what the user intended" and "what the adversary tricked the user into doing" is fundamentally a UX and authentication problem, not a provenance-tracking problem.

### 9.6 Deployment Considerations

Real-world deployment requires solving several engineering challenges not addressed here:
- **Taint tracking through LLM context.** The conservative approach (taint all output if any input is tainted) may be too restrictive for agents that process both trusted and untrusted data in the same context window. Fine-grained taint tracking through transformer attention is an open research problem.
- **Performance overhead.** The verifier adds latency to every write operation. In our implementation, this overhead is negligible (< 1ms per check), but the taint propagation tracking adds memory and computation overhead that we have not benchmarked at scale.
- **Integration with existing frameworks.** Agent frameworks (LangChain, CrewAI, AutoGPT) do not currently expose write operations through a reference monitor interface. Adoption requires framework-level support.

---

## 10. Conclusion

We presented a reference monitor for LLM agent memory that provides provenance-based write integrity. The Memory Integrity Theorem, stated under five explicit assumptions, guarantees that untrusted inputs cannot modify protected memory items or transfer data across session namespaces. The defense operates at the persistence layer — interposing on memory writes, not inference — which eliminates the security-utility trade-off observed in inference-layer defenses.

Our implementation blocks all 199 real injection payloads and 7 canonical attack vectors in our test suite, confirming the correctness of the verifier logic. We complement the structural guarantee with a detection-layer analysis showing that domain adaptation closes large performance gaps (41pp F1 improvement on deepset) and that ensembling improves confidence calibration (AUROC up to 99.63%).

The formal model makes explicit what prior work often leaves implicit: the *enforcement boundary* between the trusted runtime and the untrusted LLM, the *scope* of the guarantee (write integrity, not read confidentiality), and the *critical assumption* that the USER channel is not compromised. We identify USER-channel security as the most important open problem for future work and propose mitigation strategies.

The architectural insight — that memory write integrity can be enforced by a small reference monitor at the persistence layer without modifying the inference pipeline — suggests a general design principle for trustworthy agentic AI: separate the concerns of *what the model thinks* from *what it can persist*.

---

## 11. Reproducibility

### 11.1 Repository Structure

```
memory-integrity-theorem/
├── memory-integrity-theorem.md          # Formal theorem document
├── memory_integrity_eval/
│   ├── src/
│   │   ├── agent_state.py               # Agent state model, verifier
│   │   ├── attack_simulator.py          # 7 attack vector implementations
│   │   ├── detectors.py                 # 5 detectors (D1-D5)
│   │   ├── real_benchmark.py            # Dataset loaders, metrics, baselines
│   │   ├── main_evaluation.py           # Full evaluation pipeline
│   │   └── benchmark_integration.py     # Legacy benchmark integration
│   ├── tests/
│   │   └── test_memory_integrity.py     # 38 unit tests
│   ├── results/                         # Evaluation results (JSON)
│   └── requirements.txt
└── LICENSE                              # Apache 2.0
```

### 11.2 Quick Start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy datasets transformers tqdm pyyaml pytest

cd memory_integrity_eval/src
python main_evaluation.py    # ~45 min on CPU

pytest memory_integrity_eval/tests/ -v    # 38 tests
```

### 11.3 Reproducibility Details

| Item | Value |
|------|-------|
| Random seed (sklearn splits) | 42 |
| Random seed (PyTorch) | 42 |
| Train/test split ratio | 80/20, stratified |
| TF-IDF features | char n-gram (1,4), max 15,000, sublinear TF |
| Logistic Regression | L-BFGS, balanced weights, C=1.0 |
| DeBERTa base model | `protectai/deberta-v3-base-prompt-injection-v2` |
| DeBERTa frozen layers | Embedding + encoder layers 0–8 (9 of 12) |
| DeBERTa trainable params | 22.25M / 184.42M (12.1%) |
| Fine-tuning epochs | 3 (deepset, jailbreak), 2 (combined) |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight decay 0.01) |
| Gradient clipping | Max norm 1.0 |
| Hardware | Single CPU (no GPU) |
| Total wall-clock time | 2,750 seconds (45.8 minutes) |
| Python version | ≥ 3.9 |
| Key dependencies | torch, transformers, scikit-learn, datasets |

### 11.4 Known Reproducibility Limitations

- **No confidence intervals.** We use a single fixed train/test split. Results may vary under different splits. Future work should report k-fold cross-validation with standard deviations.
- **DeBERTa non-determinism.** PyTorch operations on CPU are not fully deterministic across hardware. Results may differ slightly on different machines.
- **Dataset versioning.** HuggingFace datasets may be updated over time. We used the versions available as of February 2026.

---

## References

[1] Debenedetti, E., Abdelnabi, S., Balunovic, M., Wagner, D., & Fritz, M. (2024). AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. *NeurIPS*, 37.

[2] Lakera AI. (2024). PINT Benchmark: Prompt Injection Test Benchmark.

[3] Liu, Y., Jia, Y., Geng, R., Jia, J., & Gong, N. Z. (2024). Formalizing and Benchmarking Prompt Injection Attacks and Defenses. *USENIX Security Symposium*.

[4] Yi, J., Xie, Y., Zhu, B., Kiciman, E., Sun, G., Xie, X., & Wu, F. (2023). Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models. *arXiv:2312.14197*.

[5] Jain, N., Schwarzschild, A., Wen, Y., Somepalli, G., Kirchenbauer, J., Chiang, P., Goldblum, M., Saha, A., Geiping, J., & Goldstein, T. (2023). Baseline Defenses for Adversarial Attacks Against Aligned Language Models. *arXiv:2309.00614*.

[6] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. *ACM AISec Workshop*.

[7] Perez, F., & Ribeiro, I. (2022). Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs through a Global Scale Prompt Hacking Competition. *arXiv:2311.16119*.

[8] Greshake, K., et al. (2023). More than you've asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models. *arXiv:2302.12173*.

[9] ProtectAI. (2024). DeBERTa v3 Base Prompt Injection v2. HuggingFace Model Card.

[10] Anderson, J. P. (1972). Computer Security Technology Planning Study. *ESD-TR-73-51*, Vol. II. Air Force Electronic Systems Division.

[11] Myers, A. C., & Liskov, B. (2000). Protecting Privacy Using the Decentralized Label Model. *ACM TOSEM*, 9(4), 410-442.

[12] Denning, D. E. (1976). A Lattice Model of Secure Information Flow. *Communications of the ACM*, 19(5), 236-243.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
