from dataclasses import dataclass
from ..chat.message_snapshot import MessageSnapshot

@dataclass(slots=True)
class MessageSnapshotEvent:
    backend: str
    snapshot: MessageSnapshot
    raw_msg_object: object | None