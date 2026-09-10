from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import Base
from api.schemas import AgentActiveStep, AgentAdvisory, AgentMemoryState
from api.agent.memory import load_memory, save_memory
from api.agent.policy import apply_focus_policy, is_out_of_scope_request


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


@pytest.mark.anyio
async def test_memory_round_trips_without_chat_transcript():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db_session:
        state = AgentMemoryState(summary="Verify CERT-003.")
        revision = await save_memory(db_session, "analysis-1", state, 0)
        loaded, loaded_revision = await load_memory(db_session, "analysis-1")
    await engine.dispose()
    assert loaded == state
    assert loaded_revision == revision
    assert not hasattr(loaded, "messages")


def test_confirmation_keeps_the_current_step():
    current = AgentMemoryState(
        summary="Verify the chain.",
        active_step=AgentActiveStep(
            id="verify-chain", title="Verify the chain", status="in_progress", evidence=["CERT-003"]
        ),
    )
    proposed = AgentMemoryState(
        summary="Now review model disagreement.",
        active_step=AgentActiveStep(
            id="review-models", title="Review model disagreement", status="in_progress", evidence=[]
        ),
    )
    result = apply_focus_policy(current, proposed, "yes, start with step 1", {"CERT-003"})
    assert result.active_step.id == "verify-chain"


def test_python_request_is_out_of_scope():
    assert is_out_of_scope_request("write a Python script to parse this") is True
    assert is_out_of_scope_request("how do I verify CERT-003?") is False
