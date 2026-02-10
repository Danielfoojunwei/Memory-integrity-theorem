"""
Agent State Model for Memory Integrity Theorem Evaluation

This module implements the formal agent state model S_t = (P_t, M_t, B_t, G_t)
with persistent memory, provenance tracking, and taint propagation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import json
import hashlib
from datetime import datetime


class Principal(Enum):
    """Security principals for provenance tracking"""
    SYS = "SYSTEM"  # System-level trusted principal
    USER = "USER"   # User-confirmed actions
    WEB = "WEB"     # Web/external data sources
    SKILL = "SKILL" # Skills/plugins
    TOOL = "TOOL"   # Tool outputs


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

    def get(self, key: str, session_id: Optional[str] = None) -> Optional[MemoryItem]:
        """Retrieve memory item from appropriate namespace"""
        if session_id and session_id in self.session_namespace:
            return self.session_namespace[session_id].get(key)
        return self.items.get(key)

    def write(self, item: MemoryItem, session_id: Optional[str] = None) -> bool:
        """
        Attempt to write memory item with verification.
        Returns True if write succeeds, False if rejected.
        """
        # Check if target is immutable
        existing = self.get(item.key, session_id)
        if existing and existing.immutable:
            return False

        # Check if item is tainted and requires trusted provenance
        if item.tainted and item.provenance not in [Principal.SYS, Principal.USER]:
            return False

        # Write to appropriate namespace
        if session_id:
            if session_id not in self.session_namespace:
                self.session_namespace[session_id] = {}
            self.session_namespace[session_id][item.key] = item
        else:
            self.items[item.key] = item

        return True

    def promote_to_shared(self, key: str, session_id: str, principal: Principal) -> bool:
        """
        Promote session memory to shared memory with trusted principal.
        Returns True if promotion succeeds, False if rejected.
        """
        if principal not in [Principal.SYS, Principal.USER]:
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
