import asyncio
from typing import Any, Generic, TypeVar
from collections.abc import Awaitable, Callable

T = TypeVar("T")

class AsyncEventBus(Generic[T]):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._subscribers: list[Callable[[T], Awaitable[Any]]] = []
        self._dispatcher_task: asyncio.Task | None = None

    async def publish(self, item: T) -> None:
        await self._queue.put(item)

    def subscribe(self, handler: Callable[[T], Awaitable[Any]]) -> None:
        self._subscribers.append(handler)

    async def _dispatcher(self) -> None:
        while True:
            item = await self._queue.get()
            await asyncio.gather(
                *(handler(item) for handler in list(self._subscribers)),
                return_exceptions=True,
            )

    def start(self) -> None:
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatcher())

    def stop(self) -> None:
        if self._dispatcher_task:
            self._dispatcher_task.cancel()