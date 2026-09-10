from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AgentMemoryRecord
from api.schemas import AgentMemoryState


class AgentMemoryConflict(RuntimeError):
    """The memory revision changed while this request was running."""


async def load_memory(
    db: AsyncSession,
    analysis_request_id: str,
) -> tuple[AgentMemoryState, int]:
    row = await db.scalar(
        select(AgentMemoryRecord).where(
            AgentMemoryRecord.analysis_request_id == analysis_request_id
        )
    )
    if row is None:
        return AgentMemoryState(), 0
    return AgentMemoryState.model_validate(row.memory), row.revision


async def save_memory(
    db: AsyncSession,
    analysis_request_id: str,
    state: AgentMemoryState,
    revision: int,
) -> int:
    row = await db.scalar(
        select(AgentMemoryRecord)
        .where(AgentMemoryRecord.analysis_request_id == analysis_request_id)
        .with_for_update()
    )
    next_revision = revision + 1
    now = datetime.now(timezone.utc)
    memory = state.model_dump(mode="json")
    if row is None:
        db.add(
            AgentMemoryRecord(
                analysis_request_id=analysis_request_id,
                memory=memory,
                revision=next_revision,
                created_at=now,
                updated_at=now,
            )
        )
    else:
        if row.revision != revision:
            raise AgentMemoryConflict("memory_revision_conflict")
        row.memory = memory
        row.revision = next_revision
        row.updated_at = now
    await db.flush()
    return next_revision
