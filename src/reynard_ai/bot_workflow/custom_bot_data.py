from abc import ABC
from ..ai_apis import providers
from ..bot_workflow.profile_loader import Profile
from ..bot_workflow.knowledge import KnowledgeIndex, LongTermMemoryIndex
from ..chat.message_history import MessageSnapshotHistory, SynchronizedMessageHistory

class AIBotData(ABC):
    def __init__(self, name: str, recent_memory: MessageSnapshotHistory | None = None):
        self.name = name
        self.memory = recent_memory

class CustomBotData(AIBotData):
    def __init__(self,
                 *,
                 name: str,
                 profile: Profile,
                 provider_store: providers.ProviderDataStore,
                 knowledge: KnowledgeIndex,
                 long_term_memory: LongTermMemoryIndex | None,
                 discord_bot_id: int,
                 memory_length: int = 50
                ):
        super().__init__(name, MessageSnapshotHistory(memory_length=memory_length))
        self.profile = profile
        self.provider_store = provider_store
        self.discord_bot_id = discord_bot_id # TODO: tight coupling
        self.long_term_memory = long_term_memory
        self.full_history = SynchronizedMessageHistory()
        self.knowledge = knowledge 
