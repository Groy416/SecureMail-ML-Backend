from __future__ import annotations

from api.agent.policy import is_out_of_scope_request


def test_python_request_is_out_of_scope():
    assert is_out_of_scope_request("write a Python script to parse this") is True
    assert is_out_of_scope_request("how do I verify CERT-003?") is False
