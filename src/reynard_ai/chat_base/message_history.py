
import asyncio
from .message_snapshot import MessageSnapshot

class MessageSnapshotHistory:
    def __init__(self, initial_history: list[MessageSnapshot]  | None = None, memory_length: int = 14):
        if initial_history is None:
            self._memory = []
        else:
            self._memory = [msg for msg in initial_history]
            self._memory = self._memory[-memory_length:]
        self.MEMORY_LENGTH = memory_length

    async def add(self, message: MessageSnapshot):
        self._memory.append(message)
        if len(self._memory) > self.MEMORY_LENGTH:
            self._memory.pop(0)

    async def add_after(self, id: int, new_message: MessageSnapshot) -> bool:
        for index, msg in enumerate(self._memory):
            if msg.message_id == id:
                self._memory.insert(index + 1, new_message)
                if len(self._memory) > self.MEMORY_LENGTH:
                    self._memory.pop(0)
                return True
        return False 
    
    async def remove(self, message: MessageSnapshot):
        self._memory = [mem_msg for mem_msg in self._memory if mem_msg.message_id != message.message_id]

    async def remove_id(self, message_id: int):
        self._memory = [mem_msg for mem_msg in self._memory if mem_msg.message_id != message_id]

    def clone(self):
        new_instance = MessageSnapshotHistory(
            initial_history=[m for m in self._memory], 
            memory_length=self.MEMORY_LENGTH
        )
        return new_instance

    def as_list(self) -> list[MessageSnapshot]:
        return self._memory
    
    def __str__(self) -> str:
        ret = ""
        for msg in self._memory:
            id = msg.message_id
            bot_str = "" if not msg.is_bot else "(BOT)"
            ret += f"[ID {id} | {msg.sent}] <{msg.nick}{bot_str}> {msg.text}\n"
        return ret

class SynchronizedMessageHistory:
    def __init__(self, *, max_length: int, initial_history: MessageSnapshotHistory | None = None):
        if initial_history is None:
            self._backing_history = MessageSnapshotHistory()
        else:
            self._backing_history = initial_history
        self._pending_message_ids: set[int] = set()
        self._lock = asyncio.Lock()

    async def add(self, message: MessageSnapshot, *, pending=False):
        async with self._lock:
            await self._backing_history.add(message)
            if pending:
                self._pending_message_ids.add(message.message_id)

    async def remove(self, message_id: int):
        async with self._lock:
            await self._backing_history.remove_id(message_id)

    async def add_after(self, id: int, message: MessageSnapshot, *, pending=False):
        async with self._lock:
            await self._backing_history.add_after(id, message)
            if pending:
                self._pending_message_ids.add(message.message_id)

    async def mark_finalized(self, message_id: int):
        async with self._lock:
            if message_id in self._pending_message_ids:
                self._pending_message_ids.remove(message_id)
            else:
                raise ValueError(f"Cannot mark non-pending message {message_id} as finalized")
        
    def is_pending(self, message_id: int) -> bool:
        return message_id in self._pending_message_ids

    async def get_finalized_message_history(self) -> MessageSnapshotHistory:
        ret_msgs = []

        async with self._lock:
            for msg in self._backing_history._memory:
                if not self.is_pending(msg.message_id):
                    ret_msgs.append(msg)
            return MessageSnapshotHistory(ret_msgs)
        
    def __str__(self) -> str:
        ret = ""
        for msg in self._backing_history._memory:
            id = msg.message_id
            pending_str = "" if not self.is_pending(id) else "(PENDING)"
            bot_str = "" if not msg.is_bot else "(BOT)"
            ret += f"{pending_str}[ID {id} | {msg.sent}] <{msg.nick}{bot_str}> {msg.text}\n"
        return ret
        