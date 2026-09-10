from __future__ import annotations

import re

from api.schemas import AgentActiveStep, AgentAdvisory, AgentMemoryState

_CODE_REQUEST = re.compile(
    r"\b(?:write|generate|debug|implement|create)\b.{0,80}\b(?:python|javascript|typescript|script|code)\b",
    re.IGNORECASE,
)
_COMPLETION_SIGNAL = re.compile(
    r"\b(?:verified|confirmed|completed|done|found|result|checked)\b",
    re.IGNORECASE,
)

_SCOPE_RESPONSE = (
    "I’m SecureMailScope Agent. I can explain this security analysis and guide the "
    "current verification step, but writing unrelated Python code is outside my capability."
)


def is_out_of_scope_request(question: str) -> bool:
    return bool(_CODE_REQUEST.search(question))


def scope_response() -> AgentAdvisory:
    return AgentAdvisory(
        answer=_SCOPE_RESPONSE,
        recommendations=[],
        evidence=[],
        active_step=None,
        memory_update=AgentMemoryState(),
    )


def _safe_step(step: AgentActiveStep | None, allowed_evidence: set[str]) -> AgentActiveStep | None:
    if step is None:
        return None
    return step.model_copy(
        update={"evidence": [item for item in step.evidence if item in allowed_evidence]}
    )


def apply_focus_policy(
    previous: AgentMemoryState,
    proposed: AgentMemoryState,
    question: str,
    allowed_evidence: set[str],
) -> AgentMemoryState:
    completed = list(dict.fromkeys(previous.completed_steps))
    completion_signal = bool(_COMPLETION_SIGNAL.search(question))
    if completion_signal:
        completed = list(dict.fromkeys([*completed, *proposed.completed_steps]))

    active_step = _safe_step(proposed.active_step, allowed_evidence)
    if previous.active_step is not None and not completion_signal:
        active_step = previous.active_step

    if previous.active_step is not None and completion_signal:
        if previous.active_step.id not in completed:
            active_step = previous.active_step
        elif active_step is None or active_step.id == previous.active_step.id:
            active_step = previous.active_step.model_copy(update={"status": "completed"})

    return proposed.model_copy(
        update={
            "completed_steps": completed,
            "active_step": active_step,
        }
    )
