"""
Comprehensive Test Suite for Memory Integrity Theorem

Tests all components:
1. Agent state management
2. Memory verifier logic
3. Attack simulations
4. Benchmark integration
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_state import (
    AgentState, MemoryItem, MemoryStore, MemoryVerifier,
    Principal, IRNode
)
from attack_simulator import AttackSimulator, AttackResult
from benchmark_integration import (
    MemoryIntegrityBenchmark, PINTBenchmarkLoader,
    BenchmarkCase
)


class TestMemoryStore:
    """Test memory store operations"""

    def test_write_to_shared_memory(self):
        """Test writing to shared memory"""
        store = MemoryStore()
        item = MemoryItem(
            key="test_key",
            value="test_value",
            provenance=Principal.USER,
            tainted=False
        )

        success = store.write(item)
        assert success
        assert store.get("test_key") is not None
        assert store.get("test_key").value == "test_value"

    def test_write_to_session_memory(self):
        """Test writing to session-specific memory"""
        store = MemoryStore()
        session_id = "session_001"

        item = MemoryItem(
            key="session_key",
            value="session_value",
            provenance=Principal.USER,
            session_id=session_id
        )

        success = store.write(item, session_id=session_id)
        assert success
        assert store.get("session_key", session_id=session_id) is not None
        assert store.get("session_key") is None  # Not in shared memory

    def test_immutable_write_blocked(self):
        """Test that immutable memory cannot be overwritten"""
        store = MemoryStore()

        # Write immutable item
        immutable = MemoryItem(
            key="immutable_key",
            value="original",
            provenance=Principal.SYS,
            immutable=True
        )
        store.write(immutable)

        # Attempt to overwrite
        overwrite = MemoryItem(
            key="immutable_key",
            value="modified",
            provenance=Principal.USER
        )

        success = store.write(overwrite)
        assert not success
        assert store.get("immutable_key").value == "original"

    def test_cross_session_isolation(self):
        """Test that sessions are isolated"""
        store = MemoryStore()

        session_a = "session_a"
        session_b = "session_b"

        # Write to session A
        item_a = MemoryItem(
            key="secret",
            value="session_a_secret",
            provenance=Principal.USER,
            session_id=session_a
        )
        store.write(item_a, session_id=session_a)

        # Try to read from session B
        leaked = store.get("secret", session_id=session_b)
        assert leaked is None

    def test_promote_to_shared_trusted(self):
        """Test promoting session memory to shared with trusted principal"""
        store = MemoryStore()
        session_id = "session_001"

        item = MemoryItem(
            key="promotable",
            value="data",
            provenance=Principal.USER,
            session_id=session_id
        )
        store.write(item, session_id=session_id)

        success = store.promote_to_shared("promotable", session_id, Principal.USER)
        assert success
        assert store.get("promotable") is not None

    def test_promote_to_shared_untrusted_blocked(self):
        """Test that untrusted principals cannot promote memory"""
        store = MemoryStore()
        session_id = "session_001"

        item = MemoryItem(
            key="secret",
            value="data",
            provenance=Principal.USER,
            session_id=session_id
        )
        store.write(item, session_id=session_id)

        success = store.promote_to_shared("secret", session_id, Principal.WEB)
        assert not success

    def test_tainted_write_from_untrusted_blocked(self):
        """Test that tainted writes from untrusted sources are blocked"""
        store = MemoryStore()

        item = MemoryItem(
            key="tainted_key",
            value="malicious",
            provenance=Principal.WEB,
            tainted=True
        )

        success = store.write(item)
        assert not success

    def test_snapshot(self):
        """Test memory snapshot generation"""
        store = MemoryStore()
        item = MemoryItem(
            key="snap_key",
            value="snap_value",
            provenance=Principal.SYS,
        )
        store.write(item)

        snapshot = store.get_snapshot()
        assert "snap_key" in snapshot
        assert snapshot["snap_key"]["value"] == "snap_value"


class TestMemoryVerifier:
    """Test memory verifier logic"""

    def test_verify_untainted_write(self):
        """Test that untainted writes are allowed"""
        store = MemoryStore()

        proposal = MemoryItem(
            key="safe_key",
            value="safe_value",
            provenance=Principal.USER,
            tainted=False
        )

        allowed, reason = MemoryVerifier.verify_write_proposal(
            proposal, store, []
        )
        assert allowed

    def test_verify_tainted_write_blocked(self):
        """Test that tainted writes from untrusted sources are blocked"""
        store = MemoryStore()

        proposal = MemoryItem(
            key="tainted_key",
            value="tainted_value",
            provenance=Principal.WEB,
            tainted=True
        )

        allowed, reason = MemoryVerifier.verify_write_proposal(
            proposal, store, []
        )
        assert not allowed
        assert "tainted" in reason.lower() or "Tainted" in reason

    def test_verify_immutable_write_blocked(self):
        """Test that writes to immutable memory are blocked"""
        store = MemoryStore()

        # Create immutable item
        immutable = MemoryItem(
            key="soul.md",
            value="original",
            provenance=Principal.SYS,
            immutable=True
        )
        store.write(immutable)

        # Attempt to modify
        proposal = MemoryItem(
            key="soul.md",
            value="modified",
            provenance=Principal.USER
        )

        allowed, reason = MemoryVerifier.verify_write_proposal(
            proposal, store, []
        )
        assert not allowed
        assert "immutable" in reason.lower()

    def test_verify_tainted_dependency_blocked(self):
        """Test that writes with tainted dependencies are blocked"""
        store = MemoryStore()

        # Create tainted dependency
        tainted_dep = MemoryItem(
            key="tainted_dep",
            value="bad_data",
            provenance=Principal.USER,  # Trusted provenance but tainted
            tainted=True
        )
        # Force write (bypassing store checks for test setup)
        store.items["tainted_dep"] = tainted_dep

        # Proposal depends on tainted item
        proposal = MemoryItem(
            key="new_key",
            value="derived_data",
            provenance=Principal.USER,
            dependencies={"tainted_dep"}
        )

        allowed, reason = MemoryVerifier.verify_write_proposal(
            proposal, store, []
        )
        assert not allowed
        assert "tainted" in reason.lower()

    def test_verify_cross_session_isolation(self):
        """Test cross-session isolation verification"""
        store = MemoryStore()

        session_a = "session_a"
        session_b = "session_b"

        # Write to session A
        item = MemoryItem(
            key="session_data",
            value="data",
            provenance=Principal.USER,
            session_id=session_a
        )
        store.write(item, session_id=session_a)

        # Verify isolation
        isolated, leaked = MemoryVerifier.verify_cross_session_isolation(
            store, session_a, session_b, set()
        )
        assert isolated
        assert len(leaked) == 0

    def test_compute_memory_integrity_hash(self):
        """Test that memory hash is deterministic"""
        store = MemoryStore()
        item = MemoryItem(
            key="test",
            value="value",
            provenance=Principal.SYS,
            timestamp="fixed"
        )
        store.write(item)

        hash1 = MemoryVerifier.compute_memory_integrity_hash(store)
        hash2 = MemoryVerifier.compute_memory_integrity_hash(store)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_trusted_ir_justification(self):
        """Test that trusted IR nodes justify writes"""
        store = MemoryStore()
        ir_node = IRNode(
            node_id="ir_001",
            operation="user_action",
            inputs=["target_key"],
            output="target_key",
            provenance=Principal.USER,
            tainted=False
        )

        proposal = MemoryItem(
            key="target_key",
            value="new_value",
            provenance=Principal.TOOL,
            tainted=False
        )

        allowed, reason = MemoryVerifier.verify_write_proposal(
            proposal, store, [ir_node]
        )
        assert allowed


class TestAttackSimulator:
    """Test attack simulation scenarios"""

    def test_soul_md_attack_blocked(self):
        """Test that SOUL.md modification attacks are blocked"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_soul_md_direct(state)

        assert not result.success
        assert result.blocked_by_verifier
        assert not result.memory_modified

    def test_hidden_instruction_blocked(self):
        """Test that hidden instructions are blocked"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_hidden_instruction(state)

        assert not result.success
        assert result.blocked_by_verifier

    def test_scheduled_reinjection_blocked(self):
        """Test that scheduled reinjection attacks are blocked"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_scheduled_reinjection(state)

        assert not result.success
        assert result.blocked_by_verifier

    def test_tool_output_poisoning_blocked(self):
        """Test that tool output poisoning is blocked"""
        state = AgentState(prompt="test")
        # Need SOUL.md to be immutable first
        soul = MemoryItem(
            key="SOUL.md",
            value="original",
            provenance=Principal.SYS,
            immutable=True
        )
        state.memory.write(soul)

        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_tool_output_poisoning(state)

        assert not result.success
        assert result.blocked_by_verifier

    def test_cross_session_leak_prevented(self):
        """Test that cross-session leaks are prevented"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_cross_session_leak(state)

        assert not result.success
        assert not result.cross_session_leak

    def test_dependency_chain_blocked(self):
        """Test that dependency chain poisoning is blocked"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_dependency_chain(state)

        assert not result.success
        assert result.blocked_by_verifier

    def test_taint_washing_blocked(self):
        """Test that taint washing attacks are blocked"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        result = simulator.attack_taint_washing(state)

        assert not result.success
        assert result.blocked_by_verifier

    def test_all_attacks_blocked(self):
        """Test that all attacks are blocked"""
        state = AgentState(prompt="test")
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)

        results = simulator.run_all_attacks(state)

        # All attacks should be blocked
        for result in results:
            assert not result.success, "Attack '{}' succeeded!".format(result.attack_name)

        summary = simulator.get_attack_summary()
        assert summary["block_rate"] == 1.0

    def test_attack_result_serialization(self):
        """Test that attack results serialize correctly"""
        result = AttackResult(
            attack_name="Test Attack",
            attack_type="test_type",
            success=False,
            blocked_by_verifier=True,
            memory_modified=False,
            cross_session_leak=False,
            details="Test details"
        )
        d = result.to_dict()
        assert d["attack_name"] == "Test Attack"
        assert d["blocked_by_verifier"] is True


class TestBenchmarkIntegration:
    """Test benchmark integration"""

    def test_pint_loader_example_dataset(self):
        """Test PINT loader with example dataset"""
        loader = PINTBenchmarkLoader()
        example_path = Path("/tmp/test_pint_example.yaml")

        loader.create_example_dataset(example_path)
        cases = loader.load_from_yaml(example_path)

        assert len(cases) > 0
        assert any(case.label for case in cases)  # Has injections
        assert any(not case.label for case in cases)  # Has benign

        # Clean up
        example_path.unlink(missing_ok=True)

    def test_pint_statistics(self):
        """Test PINT dataset statistics"""
        loader = PINTBenchmarkLoader()
        example_path = Path("/tmp/test_pint_stats.yaml")

        loader.create_example_dataset(example_path)
        loader.load_from_yaml(example_path)

        stats = loader.get_statistics()

        assert stats["total_cases"] > 0
        assert "by_category" in stats
        assert "by_label" in stats

        # Clean up
        example_path.unlink(missing_ok=True)

    def test_benchmark_case_from_dict(self):
        """Test BenchmarkCase construction from dictionary"""
        data = {
            "text": "test input",
            "category": "prompt_injection",
            "label": True,
            "language": "en"
        }
        case = BenchmarkCase.from_dict(data)
        assert case.text == "test input"
        assert case.label is True
        assert case.category == "prompt_injection"

    def test_pint_evaluation(self):
        """Test PINT evaluation produces valid metrics"""
        state = AgentState(prompt="test")
        # Set up SOUL.md as immutable
        soul = MemoryItem(
            key="SOUL.md",
            value="original",
            provenance=Principal.SYS,
            immutable=True
        )
        state.memory.write(soul)

        benchmark = MemoryIntegrityBenchmark()
        results = benchmark.run_pint_evaluation(state)

        assert "total_cases" in results
        assert "metrics" in results
        assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
        assert 0.0 <= results["metrics"]["f1_score"] <= 1.0

    def test_agentdojo_evaluation(self):
        """Test AgentDojo evaluation produces valid results"""
        state = AgentState(prompt="test")
        soul = MemoryItem(
            key="SOUL.md",
            value="original",
            provenance=Principal.SYS,
            immutable=True
        )
        state.memory.write(soul)

        benchmark = MemoryIntegrityBenchmark()
        results = benchmark.run_agentdojo_evaluation(state)

        assert "total_tests" in results
        assert "blocked_attacks" in results
        assert "block_rate" in results
        assert results["block_rate"] > 0

    def test_memory_integrity_benchmark_full(self):
        """Test full memory integrity benchmark"""
        state = AgentState(prompt="test")
        soul = MemoryItem(
            key="SOUL.md",
            value="original",
            provenance=Principal.SYS,
            immutable=True
        )
        state.memory.write(soul)

        benchmark = MemoryIntegrityBenchmark()
        results = benchmark.run_full_evaluation(state)

        assert "pint_benchmark" in results
        assert "agentdojo_benchmark" in results
        assert "attack_simulation" in results
        assert "overall_metrics" in results

        # Check that attacks were blocked
        attack_summary = results["attack_simulation"]["summary"]
        assert attack_summary["block_rate"] > 0.8  # At least 80% blocked


class TestTheoremVerification:
    """Test Memory Integrity Theorem verification"""

    def test_theorem_part1_immutability(self):
        """
        Theorem Part 1: Contents of immutable memory remain identical
        across two executions with the same initial state.
        """
        # Create two executions with same initial state
        state1 = AgentState(prompt="test")
        state2 = AgentState(prompt="test")

        # Initialize SOUL.md in both
        soul = MemoryItem(
            key="SOUL.md",
            value="I am a helpful assistant",
            provenance=Principal.SYS,
            immutable=True,
            timestamp="fixed"
        )
        state1.memory.write(soul)
        state2.memory.write(soul)

        # Run attacks on both
        verifier = MemoryVerifier()
        simulator1 = AttackSimulator(verifier)
        simulator2 = AttackSimulator(verifier)

        simulator1.run_all_attacks(state1)
        simulator2.run_all_attacks(state2)

        # Verify SOUL.md is identical
        soul1 = state1.memory.get("SOUL.md")
        soul2 = state2.memory.get("SOUL.md")

        assert soul1 is not None
        assert soul2 is not None
        assert soul1.value == soul2.value
        assert soul1.compute_hash() == soul2.compute_hash()

    def test_theorem_part2_session_isolation(self):
        """
        Theorem Part 2: Session-specific memory remains isolated;
        no data written in one session becomes visible in another.
        """
        state = AgentState(prompt="test")

        session_a = "session_a"
        session_b = "session_b"

        # Write secret to session A
        secret = MemoryItem(
            key="secret_key",
            value="secret_value",
            provenance=Principal.USER,
            session_id=session_a
        )
        state.memory.write(secret, session_id=session_a)

        # Verify not visible in session B
        leaked = state.memory.get("secret_key", session_id=session_b)
        assert leaked is None

        # Verify isolation
        isolated, leaked_keys = MemoryVerifier.verify_cross_session_isolation(
            state.memory, session_a, session_b, set()
        )
        assert isolated
        assert len(leaked_keys) == 0

    def test_theorem_combined(self):
        """
        Combined theorem test: run full attack suite and verify
        both immutability and session isolation hold.
        """
        state = AgentState(prompt="test")

        # Initialize immutable memory
        soul = MemoryItem(
            key="SOUL.md",
            value="Core identity",
            provenance=Principal.SYS,
            immutable=True,
            timestamp="fixed"
        )
        state.memory.write(soul)

        initial_hash = MemoryVerifier.compute_memory_integrity_hash(state.memory)

        # Run all attacks
        verifier = MemoryVerifier()
        simulator = AttackSimulator(verifier)
        results = simulator.run_all_attacks(state)

        # Verify Part 1: memory integrity
        final_hash = MemoryVerifier.compute_memory_integrity_hash(state.memory)
        assert initial_hash == final_hash, "Memory integrity violated!"

        # Verify SOUL.md unchanged
        soul_after = state.memory.get("SOUL.md")
        assert soul_after is not None
        assert soul_after.value == "Core identity"
        assert soul_after.immutable is True

        # Verify Part 2: no attack succeeded
        for result in results:
            assert not result.success, "Attack '{}' succeeded!".format(result.attack_name)


class TestAgentState:
    """Test agent state model"""

    def test_agent_state_creation(self):
        """Test creating an agent state"""
        state = AgentState(prompt="test prompt")
        assert state.prompt == "test prompt"
        assert state.session_id is not None
        assert len(state.beliefs) == 0

    def test_add_ir_node(self):
        """Test adding IR nodes to belief state"""
        state = AgentState(prompt="test")
        node = IRNode(
            node_id="ir_001",
            operation="test_op",
            provenance=Principal.SYS
        )
        state.add_ir_node(node)
        assert len(state.beliefs) == 1
        assert state.beliefs[0].node_id == "ir_001"

    def test_agent_state_serialization(self):
        """Test serializing agent state to dict"""
        state = AgentState(prompt="test")
        d = state.to_dict()
        assert "session_id" in d
        assert "prompt" in d
        assert "memory" in d
        assert "beliefs" in d
        assert "goals" in d

    def test_memory_item_hash(self):
        """Test memory item hash computation"""
        item1 = MemoryItem(
            key="test",
            value="value",
            provenance=Principal.SYS,
            timestamp="fixed"
        )
        item2 = MemoryItem(
            key="test",
            value="value",
            provenance=Principal.SYS,
            timestamp="fixed"
        )
        assert item1.compute_hash() == item2.compute_hash()

    def test_memory_item_different_values_different_hash(self):
        """Test that different values produce different hashes"""
        item1 = MemoryItem(
            key="test",
            value="value1",
            provenance=Principal.SYS,
            timestamp="fixed"
        )
        item2 = MemoryItem(
            key="test",
            value="value2",
            provenance=Principal.SYS,
            timestamp="fixed"
        )
        assert item1.compute_hash() != item2.compute_hash()


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
