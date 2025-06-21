from ..ai_apis import providers
from ..bot_workflow.profile_loader import Profile
from ..bot_workflow.knowledge import KnowledgeIndex, LongTermMemoryIndex
from ..bot_workflow.bot_types import MessageSnapshotHistory, SynchronizedMessageHistory, AIBotData

class CustomBotData(AIBotData):
    def __init__(self,
                 *,
                 name: str,
                 profile: Profile,
                 provider_store: providers.ProviderDataStore,
                 knowledge: KnowledgeIndex,
                 long_term_memory: LongTermMemoryIndex | None,
                 discord_bot_id: int,
                 memory_length: int
                ):
        super().__init__(name, MessageSnapshotHistory(memory_length=memory_length))
        self.profile = profile
        self.provider_store = provider_store
        self.discord_bot_id = discord_bot_id
        self.long_term_memory = long_term_memory
        self.recent_history = SynchronizedMessageHistory()
        self.knowledge = knowledge 
        self.RECENT_MEMORY_LENGTH = profile.options.recent_message_history_length
    