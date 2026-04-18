"""P0-3: Idempotency store — duplicate calls return the cached result."""

import pytest

from koa.tenant_gate.idempotency import InMemoryIdempotencyStore, make_idempotency_key


def test_make_key_prefers_caller_key():
    k1 = make_idempotency_key(
        tenant_id="t",
        session_id="s",
        tool_name="send",
        arguments={"to": "a"},
        caller_key="abc",
    )
    assert k1.startswith("caller:t:abc")


def test_make_key_order_independent():
    k1 = make_idempotency_key(
        tenant_id="t", session_id="s", tool_name="send", arguments={"a": 1, "b": 2}
    )
    k2 = make_idempotency_key(
        tenant_id="t", session_id="s", tool_name="send", arguments={"b": 2, "a": 1}
    )
    assert k1 == k2


@pytest.mark.asyncio
async def test_begin_returns_existing_on_duplicate():
    store = InMemoryIdempotencyStore()
    new, existing = await store.begin("k1")
    assert new is True and existing is None
    await store.complete("k1", {"status": "sent"})

    new2, existing2 = await store.begin("k1")
    assert new2 is False
    assert existing2 is not None and existing2.result == {"status": "sent"}


@pytest.mark.asyncio
async def test_fail_releases_key():
    store = InMemoryIdempotencyStore()
    await store.begin("k")
    await store.fail("k")
    new, _ = await store.begin("k")
    assert new is True  # retry allowed
