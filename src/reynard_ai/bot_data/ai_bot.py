from abc import ABC

from ..ai_apis import providers
from discord.ext import commands
from .bot_profile import Profile
from ..ai_apis.api_types import Prompt
from ..ai_apis.client import LLMClient
from ..events.event_bus import AsyncEventBus
from ..chat.base_chat_handler import BaseChatHandler
from ..chat.discord_events_bridge import DiscordBridge
from ..chat.discord_chat_handler import DiscordChatHandler
from ..bot_data.knowledge import KnowledgeIndex, LongTermMemoryIndex
from ..chat.message_history import MessageSnapshotHistory, SynchronizedMessageHistory

class AbstractReynardAIBotData(ABC):
    def __init__(self, name: str, recent_memory: MessageSnapshotHistory | None = None):
        self.name = name
        self.memory = recent_memory

class ReynardAIBotData(AbstractReynardAIBotData):
    def __init__(self,
                 *,
                 profile: Profile,
                 provider_store: providers.ProviderDataStore,
                 knowledge: KnowledgeIndex,
                 long_term_memory: LongTermMemoryIndex | None,
                 account_id: int,
                 memory_length: int = 50
                ):
        super().__init__(self.profile.options.botname, MessageSnapshotHistory(memory_length=memory_length))
        self.profile = profile
        self.provider_store = provider_store
        self.account_id = account_id
        self.long_term_memory = long_term_memory
        self.short_term_memory = SynchronizedMessageHistory(max_length=profile.memory_settings.medium_term_history_length)
        self.medium_term_memory : SynchronizedMessageHistory | None = None
        if self.profile.memory_settings.enable_long_term_memory:
            self.medium_term_memory : SynchronizedMessageHistory | None = SynchronizedMessageHistory(
                max_length=self.profile.memory_settings.medium_term_history_length
            )
        self.knowledge = knowledge 

    async def send_llm_request(self, *, provider_name: str, prompt: Prompt, parameter_set_name: str | None = None):
        if parameter_set_name is None:
            parameter_set_name = provider_name
        params = self.profile.get_request_params(provider_name)
        provider: providers.ProviderData = self.profile.get_provider(parameter_set_name)
        client: LLMClient = LLMClient.from_provider(provider)
        return await client.send_request(prompt=prompt, params=params)
    
class ReynardChatBot():
    def __init__(self, bot_data: ReynardAIBotData, chat_handler: BaseChatHandler):
        self.bot_data = bot_data
        self.event_bus = AsyncEventBus()
        self.chat_handler = chat_handler 
        self.event_bus.start()

    @classmethod
    async def create_discord_bot(cls, discord_bot: commands.Bot, ai_bot_data: ReynardAIBotData):
        event_bus = AsyncEventBus()
        chat_handler = DiscordChatHandler(
            bus=event_bus,
            ai_bot=ai_bot_data
        )
        discord_bridge = DiscordBridge(
            discord_bot, 
            bus=event_bus, 
            known_chatrooms=[]
        )
        await discord_bot.add_cog(discord_bridge)
        return cls(ai_bot_data, chat_handler)