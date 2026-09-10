from __future__ import annotations

from api.schemas import AgentActiveStep, AgentAdvisory, AgentMemoryState


def test_memory_state_is_bounded_and_typed():
    state = AgentMemoryState(
        summary="CERT-003 is being verified.",
        facts=["CERT-003 is deterministic"],
        active_step=AgentActiveStep(
            id="verify-certificate-chain",
            title="Verify the certificate chain",
            status="in_progress",
            evidence=["CERT-003"],
        ),
    )
    assert state.active_step.id == "verify-certificate-chain"
    assert state.completed_steps == []


def test_advisory_requires_memory_update():
    state = AgentMemoryState(summary="CERT-003 is being verified.")
    advisory = AgentAdvisory(
        answer="Verify the chain.",
        recommendations=["Verify the chain."],
        evidence=["CERT-003"],
        active_step=None,
        memory_update=state,
    )
    assert advisory.memory_update.summary == "CERT-003 is being verified."
