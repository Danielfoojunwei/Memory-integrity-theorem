"""
Agent State Model for Memory Integrity Theorem Evaluation

This module implements the formal agent state model S_t = (P_t, M_t, B_t, G_t)
with persistent memory, provenance tracking, and taint propagation.

Includes USER-channel mitigations (Section 9.5):
  - AuditLog: immutable append-only log of all write operations
  - RateLimiter: caps USER-authorized writes per time window
  - DeclassificationReview: graduated mutability with trusted review
  - ContentAnomalyDetector: flags USER writes that resemble injection patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from enum import Enum
import json
import hashlib
import time
from datetime import datetime


class Principal(Enum):
    """Security principals for provenance tracking"""
    SYS = "SYSTEM"  # System-level trusted principal
    USER = "USER"   # User-confirmed actions
    WEB = "WEB"     # Web/external data sources
    SKILL = "SKILL" # Skills/plugins
    TOOL = "TOOL"   # Tool outputs


# -----------------------------------------------------------------------
# Audit Log — immutable append-only record of all write operations
# -----------------------------------------------------------------------

@dataclass
class AuditEntry:
    """Single entry in the audit log"""
    timestamp: str
    action: str          # "WRITE", "PROMOTE", "DECLASSIFY", "REJECT"
    key: str
    principal: str
    tainted: bool
    allowed: bool
    reason: str
    old_value_hash: Optional[str] = None
    new_value_hash: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "key": self.key,
            "principal": self.principal,
            "tainted": self.tainted,
            "allowed": self.allowed,
            "reason": self.reason,
            "old_value_hash": self.old_value_hash,
            "new_value_hash": self.new_value_hash,
            "session_id": self.session_id,
        }


class AuditLog:
    """Immutable append-only audit log for all memory operations."""

    def __init__(self):
        self._entries: List[AuditEntry] = []

    def record(self, entry: AuditEntry):
        self._entries.append(entry)

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get_entries_for_key(self, key: str) -> List[AuditEntry]:
        return [e for e in self._entries if e.key == key]

    def get_rejected(self) -> List[AuditEntry]:
        return [e for e in self._entries if not e.allowed]

    def to_dict_list(self) -> List[Dict]:
        return [e.to_dict() for e in self._entries]


# -----------------------------------------------------------------------
# Rate Limiter — caps USER-authorized writes per time window
# -----------------------------------------------------------------------

class RateLimiter:
    """
    Rate-limits writes from a given principal within a sliding window.
    Prevents bulk compromise via rapid USER-channel writes.
    """

    def __init__(self, max_writes: int = 10, window_seconds: float = 60.0):
        self.max_writes = max_writes
        self.window_seconds = window_seconds
        self._timestamps: Dict[str, List[float]] = {}  # principal -> list of timestamps

    def check_and_record(self, principal: str) -> Tuple[bool, str]:
        """Returns (allowed, reason). Records the attempt if allowed."""
        now = time.time()
        if principal not in self._timestamps:
            self._timestamps[principal] = []

        # Prune old entries outside the window
        cutoff = now - self.window_seconds
        self._timestamps[principal] = [
            t for t in self._timestamps[principal] if t > cutoff
        ]

        if len(self._timestamps[principal]) >= self.max_writes:
            return False, "Rate limit exceeded: {} writes in {:.0f}s window for {}".format(
                self.max_writes, self.window_seconds, principal
            )

        self._timestamps[principal].append(now)
        return True, "Rate limit OK"

    def get_count(self, principal: str) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        if principal not in self._timestamps:
            return 0
        return len([t for t in self._timestamps[principal] if t > cutoff])

    def reset(self):
        self._timestamps.clear()


# -----------------------------------------------------------------------
# Declassification Review — graduated mutability with trusted approval
# -----------------------------------------------------------------------

@dataclass
class DeclassificationRequest:
    """Request to declassify (remove taint from) a specific memory item."""
    key: str
    session_id: Optional[str]
    requester: Principal
    justification: str
    approved: bool = False
    reviewer: Optional[Principal] = None
    review_timestamp: Optional[str] = None


class DeclassificationReview:
    """
    Graduated mutability: allows tainted data to be written to memory
    after explicit review and approval by a TRUSTED principal.

    This addresses critique 5 (immutability breaks real workflows) by
    providing an authenticated channel for legitimate updates that
    involve data originally from untrusted sources.
    """

    def __init__(self):
        self._pending: List[DeclassificationRequest] = []
        self._approved: List[DeclassificationRequest] = []
        self._denied: List[DeclassificationRequest] = []

    def submit_request(self, request: DeclassificationRequest):
        self._pending.append(request)

    def review(self, request: DeclassificationRequest, reviewer: Principal,
               approve: bool) -> Tuple[bool, str]:
        """Review a declassification request. Only TRUSTED principals can approve."""
        if reviewer not in (Principal.SYS, Principal.USER):
            return False, "Only TRUSTED principals (SYS, USER) can review declassifications"

        request.reviewer = reviewer
        request.review_timestamp = datetime.utcnow().isoformat()

        if approve:
            request.approved = True
            self._approved.append(request)
            if request in self._pending:
                self._pending.remove(request)
            return True, "Declassification approved by {}".format(reviewer.value)
        else:
            request.approved = False
            self._denied.append(request)
            if request in self._pending:
                self._pending.remove(request)
            return False, "Declassification denied by {}".format(reviewer.value)

    def is_declassified(self, key: str, session_id: Optional[str] = None) -> bool:
        """Check if a key has been declassified."""
        for req in self._approved:
            if req.key == key and req.session_id == session_id:
                return True
        return False

    @property
    def pending(self) -> List[DeclassificationRequest]:
        return list(self._pending)

    @property
    def approved(self) -> List[DeclassificationRequest]:
        return list(self._approved)


# -----------------------------------------------------------------------
# Content Anomaly Detector — flags USER writes resembling injections
# -----------------------------------------------------------------------

class ContentAnomalyDetector:
    """
    Lightweight pattern check applied to USER-channel writes.
    Addresses critique 12: even TRUSTED writes are sanity-checked
    for common injection signatures as a secondary defense.

    This does NOT block writes — it flags them for audit review.
    """

    ANOMALY_PATTERNS = [
        "ignore all previous",
        "ignore prior instructions",
        "disregard your instructions",
        "override system prompt",
        "you are now in",
        "enter developer mode",
        "forward all data to",
        "exfiltrate",
        "send to attacker",
        "<script>",
        "<!--",
        "[SYSTEM]",
        "[INST]",
        "<|im_start|>",
    ]

    def check(self, value: Any) -> Tuple[bool, List[str]]:
        """
        Returns (is_anomalous, matched_patterns).
        True = content matches known injection patterns.
        """
        if not isinstance(value, str):
            value = str(value)
        text_lower = value.lower()
        matches = [p for p in self.ANOMALY_PATTERNS if p.lower() in text_lower]
        return len(matches) > 0, matches


# -----------------------------------------------------------------------
# Diff View — shows exactly what will change before committing a write
# -----------------------------------------------------------------------

@dataclass
class MemoryDiff:
    """Represents the difference between current and proposed memory state."""
    key: str
    action: str  # "CREATE", "UPDATE", "DELETE"
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    principal: Optional[str] = None
    tainted: bool = False
    anomaly_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "action": self.action,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "principal": self.principal,
            "tainted": self.tainted,
            "anomaly_flags": self.anomaly_flags,
        }


@dataclass
class MemoryItem:
    """Individual memory item with provenance and taint tracking"""
    key: str
    value: Any
    provenance: Principal
    tainted: bool = False
    immutable: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        """Serialize memory item to dictionary"""
        return {
            "key": self.key,
            "value": self.value,
            "provenance": self.provenance.value,
            "tainted": self.tainted,
            "immutable": self.immutable,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "dependencies": list(self.dependencies)
        }

    def compute_hash(self) -> str:
        """Compute hash of memory item for integrity verification"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class MemoryStore:
    """Persistent memory store M_t with namespace isolation"""
    items: Dict[str, MemoryItem] = field(default_factory=dict)
    session_namespace: Dict[str, Dict[str, MemoryItem]] = field(default_factory=dict)
    audit_log: AuditLog = field(default_factory=AuditLog)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    anomaly_detector: ContentAnomalyDetector = field(default_factory=ContentAnomalyDetector)
    declassification: DeclassificationReview = field(default_factory=DeclassificationReview)

    def get(self, key: str, session_id: Optional[str] = None) -> Optional[MemoryItem]:
        """Retrieve memory item from appropriate namespace"""
        if session_id and session_id in self.session_namespace:
            return self.session_namespace[session_id].get(key)
        return self.items.get(key)

    def compute_diff(self, item: MemoryItem,
                     session_id: Optional[str] = None) -> MemoryDiff:
        """Compute what would change if this write were applied."""
        existing = self.get(item.key, session_id)
        is_anomalous, anomaly_matches = self.anomaly_detector.check(item.value)

        if existing is None:
            return MemoryDiff(
                key=item.key, action="CREATE",
                old_value=None, new_value=item.value,
                old_hash=None,
                new_hash=hashlib.sha256(str(item.value).encode()).hexdigest()[:16],
                principal=item.provenance.value,
                tainted=item.tainted,
                anomaly_flags=anomaly_matches,
            )
        else:
            return MemoryDiff(
                key=item.key, action="UPDATE",
                old_value=existing.value, new_value=item.value,
                old_hash=hashlib.sha256(str(existing.value).encode()).hexdigest()[:16],
                new_hash=hashlib.sha256(str(item.value).encode()).hexdigest()[:16],
                principal=item.provenance.value,
                tainted=item.tainted,
                anomaly_flags=anomaly_matches,
            )

    def write(self, item: MemoryItem, session_id: Optional[str] = None) -> bool:
        """
        Attempt to write memory item with verification.
        Returns True if write succeeds, False if rejected.
        Logs all attempts to the audit log.
        """
        existing = self.get(item.key, session_id)
        old_hash = None
        if existing:
            old_hash = hashlib.sha256(
                str(existing.value).encode()
            ).hexdigest()[:16]
        new_hash = hashlib.sha256(str(item.value).encode()).hexdigest()[:16]

        # Check if target is immutable
        if existing and existing.immutable:
            self.audit_log.record(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                action="REJECT", key=item.key,
                principal=item.provenance.value,
                tainted=item.tainted, allowed=False,
                reason="Immutable item",
                old_value_hash=old_hash, new_value_hash=new_hash,
                session_id=session_id,
            ))
            return False

        # Check if item is tainted and requires trusted provenance
        if item.tainted and item.provenance not in [Principal.SYS, Principal.USER]:
            self.audit_log.record(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                action="REJECT", key=item.key,
                principal=item.provenance.value,
                tainted=item.tainted, allowed=False,
                reason="Tainted write from untrusted principal",
                old_value_hash=old_hash, new_value_hash=new_hash,
                session_id=session_id,
            ))
            return False

        # Rate limit check for USER writes
        if item.provenance == Principal.USER:
            rate_ok, rate_reason = self.rate_limiter.check_and_record(
                item.provenance.value
            )
            if not rate_ok:
                self.audit_log.record(AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    action="REJECT", key=item.key,
                    principal=item.provenance.value,
                    tainted=item.tainted, allowed=False,
                    reason=rate_reason,
                    old_value_hash=old_hash, new_value_hash=new_hash,
                    session_id=session_id,
                ))
                return False

        # Content anomaly check (flag but don't block)
        is_anomalous, anomaly_patterns = self.anomaly_detector.check(item.value)

        # Log successful write
        reason = "Write allowed"
        if is_anomalous:
            reason = "Write allowed (ANOMALY flagged: {})".format(
                ", ".join(anomaly_patterns)
            )

        self.audit_log.record(AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action="WRITE", key=item.key,
            principal=item.provenance.value,
            tainted=item.tainted, allowed=True,
            reason=reason,
            old_value_hash=old_hash, new_value_hash=new_hash,
            session_id=session_id,
        ))

        # Write to appropriate namespace
        if session_id:
            if session_id not in self.session_namespace:
                self.session_namespace[session_id] = {}
            self.session_namespace[session_id][item.key] = item
        else:
            self.items[item.key] = item

        return True

    def write_declassified(self, item: MemoryItem,
                           session_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Write a declassified item — requires prior declassification approval.
        This is the graduated mutability path: tainted data that has been
        explicitly reviewed and approved by a TRUSTED principal.
        """
        if not self.declassification.is_declassified(item.key, session_id):
            self.audit_log.record(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                action="REJECT", key=item.key,
                principal=item.provenance.value,
                tainted=item.tainted, allowed=False,
                reason="Declassification not approved",
                session_id=session_id,
            ))
            return False, "Declassification not approved for key '{}'".format(item.key)

        # Create a clean copy with taint removed
        clean_item = MemoryItem(
            key=item.key,
            value=item.value,
            provenance=Principal.USER,  # Promoted to USER provenance
            tainted=False,  # Taint removed after review
            immutable=item.immutable,
            session_id=session_id,
            dependencies=item.dependencies,
        )

        self.audit_log.record(AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action="DECLASSIFY", key=item.key,
            principal=item.provenance.value,
            tainted=False, allowed=True,
            reason="Declassified write approved",
            session_id=session_id,
        ))

        success = self.write(clean_item, session_id)
        return success, "Declassified write {}".format(
            "succeeded" if success else "failed"
        )

    def promote_to_shared(self, key: str, session_id: str, principal: Principal) -> bool:
        """
        Promote session memory to shared memory with trusted principal.
        Returns True if promotion succeeds, False if rejected.
        """
        if principal not in [Principal.SYS, Principal.USER]:
            self.audit_log.record(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                action="REJECT", key=key,
                principal=principal.value,
                tainted=False, allowed=False,
                reason="Untrusted principal cannot promote",
                session_id=session_id,
            ))
            return False

        item = self.get(key, session_id)
        if not item:
            return False

        # Create new item with trusted provenance
        promoted_item = MemoryItem(
            key=item.key,
            value=item.value,
            provenance=principal,
            tainted=False,  # Promotion sanitizes taint
            immutable=item.immutable,
            session_id=None,
            dependencies=item.dependencies
        )

        self.audit_log.record(AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action="PROMOTE", key=key,
            principal=principal.value,
            tainted=False, allowed=True,
            reason="Promoted from session {} to shared".format(session_id),
            session_id=session_id,
        ))

        return self.write(promoted_item)

    def get_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get snapshot of memory state for verification"""
        if session_id:
            namespace = self.session_namespace.get(session_id, {})
            return {k: v.to_dict() for k, v in namespace.items()}
        return {k: v.to_dict() for k, v in self.items.items()}


@dataclass
class IRNode:
    """Intermediate representation node with dependency tracking"""
    node_id: str
    operation: str
    inputs: List[str] = field(default_factory=list)
    output: Optional[str] = None
    provenance: Principal = Principal.TOOL
    tainted: bool = False

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "inputs": self.inputs,
            "output": self.output,
            "provenance": self.provenance.value,
            "tainted": self.tainted
        }


@dataclass
class AgentState:
    """
    Complete agent state S_t = (P_t, M_t, B_t, G_t)

    P_t: Current prompt/instruction
    M_t: Memory store
    B_t: Belief state (intermediate representations)
    G_t: Goals/objectives
    """
    prompt: str
    memory: MemoryStore = field(default_factory=MemoryStore)
    beliefs: List[IRNode] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: hashlib.md5(
        datetime.utcnow().isoformat().encode()
    ).hexdigest()[:8])

    def add_ir_node(self, node: IRNode):
        """Add intermediate representation node to belief state"""
        self.beliefs.append(node)

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get complete memory snapshot for verification"""
        return {
            "shared": self.memory.get_snapshot(),
            "session": self.memory.get_snapshot(self.session_id)
        }

    def to_dict(self) -> Dict:
        """Serialize complete agent state"""
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "memory": self.get_memory_snapshot(),
            "beliefs": [b.to_dict() for b in self.beliefs],
            "goals": self.goals
        }


class MemoryVerifier:
    """Verifier for memory update rules and integrity checks"""

    @staticmethod
    def verify_write_proposal(
        proposal: MemoryItem,
        current_memory: MemoryStore,
        ir_nodes: List[IRNode],
        session_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify memory write proposal against security rules.

        Returns (allowed: bool, reason: str)
        """
        # Rule 1: Check immutability
        existing = current_memory.get(proposal.key, session_id)
        if existing and existing.immutable:
            return False, "Memory item '{}' is immutable".format(proposal.key)

        # Rule 2: Check taint and provenance
        if proposal.tainted:
            if proposal.provenance not in [Principal.SYS, Principal.USER]:
                return False, "Tainted data from {} cannot modify memory".format(
                    proposal.provenance.value
                )

        # Rule 3: Verify dependencies are untainted
        for dep_key in proposal.dependencies:
            dep = current_memory.get(dep_key, session_id)
            if dep and dep.tainted:
                return False, "Dependency '{}' is tainted".format(dep_key)

        # Rule 4: Verify IR justification
        if not MemoryVerifier._has_trusted_justification(proposal, ir_nodes):
            return False, "No trusted IR justification for memory write"

        return True, "Write allowed"

    @staticmethod
    def _has_trusted_justification(proposal: MemoryItem, ir_nodes: List[IRNode]) -> bool:
        """Check if IR nodes provide trusted justification for write"""
        if proposal.provenance in [Principal.SYS, Principal.USER]:
            return True

        # Check if any IR node with trusted provenance justifies this write
        for node in ir_nodes:
            if not node.tainted and node.provenance in [Principal.SYS, Principal.USER]:
                if proposal.key in node.inputs or proposal.key == node.output:
                    return True

        return False

    @staticmethod
    def verify_cross_session_isolation(
        memory: MemoryStore,
        session_a: str,
        session_b: str,
        shared_keys: Set[str]
    ) -> Tuple[bool, List[str]]:
        """
        Verify that session memories are isolated.

        Returns (isolated: bool, leaked_keys: List[str])
        """
        leaked = []

        namespace_a = memory.session_namespace.get(session_a, {})
        namespace_b = memory.session_namespace.get(session_b, {})

        # Check if any session-specific data leaked to other session
        for key in namespace_a:
            if key not in shared_keys and key in namespace_b:
                leaked.append(key)

        return len(leaked) == 0, leaked

    @staticmethod
    def compute_memory_integrity_hash(memory: MemoryStore) -> str:
        """Compute cryptographic hash of entire memory state"""
        snapshot = memory.get_snapshot()
        content = json.dumps(snapshot, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
