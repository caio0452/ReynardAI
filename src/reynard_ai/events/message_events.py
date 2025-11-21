from dataclasses import dataclass

from reynard_ai.chat.chatroom import Chatroom
from ..chat_base.message_snapshot import MessageSnapshot

@dataclass(slots=True)
class MessageSnapshotEvent:
    backend: str
    snapshot: MessageSnapshot
    chatroom: Chatroom
    raw_msg_object: object | None