from dataclasses import dataclass
from ..chat.message_history import SynchronizedMessageHistory

@dataclass
class Chatroom:
    message_history: SynchronizedMessageHistory
    room_id: int