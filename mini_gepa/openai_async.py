from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type


_client: Optional[AsyncOpenAI] = None
_client_loop: Optional[asyncio.AbstractEventLoop] = None


async def get_client() -> AsyncOpenAI:
    """Return a process-global AsyncOpenAI client bound to the current event loop.

    Recreates the client if the loop changed or if the client was reset.
    """
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is not None and _client_loop is loop:
        return _client

    _client = AsyncOpenAI(
        max_retries=5, 
        timeout=1800, 
        # base_url="http://localhost:8000/v1"
    )
    _client.responses.create
    _client_loop = loop
    return _client


def reset_client() -> None:
    """Reset the cached client to force recreation on next use."""
    global _client, _client_loop
    _client = None
    _client_loop = None


T = TypeVar("T")


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random(min=5, max=10),
    retry=retry_if_exception_type((openai.APITimeoutError, openai.BadRequestError)),
    reraise=True,
)
async def call_with_client(coro_factory: Callable[[AsyncOpenAI], Awaitable[T]]) -> T:
    """Execute an async API call with retry; recreate client on timeouts and specific errors."""
    try:
        client = await get_client()
        return await coro_factory(client)
    except openai.APITimeoutError:
        reset_client()
        raise
    except openai.BadRequestError as e:
        # Check for the specific error message about 'NoneType' object has no attribute 'startswith'
        if "'NoneType' object has no attribute 'startswith'" in str(e):
            reset_client()
        raise


async def responses_create(**kwargs: Any) -> Any:
    """Thin wrapper around client.responses.create using the global client."""
    return await call_with_client(lambda client: client.responses.create(**kwargs))


