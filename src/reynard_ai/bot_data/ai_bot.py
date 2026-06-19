from abc import ABC
from typing import Any

from ..ai_apis import providers
from .bot_profile import Profile
from ..ai_apis.api_types import Prompt
from ..ai_apis.client import LLMClient
from ..bot_data.knowledge import KnowledgeIndex, LongTermMemoryIndex
from ..chat_base.message_history import MessageSnapshotHistory, SynchronizedMessageHistory
from .modules import BaseModule

class AbstractReynardAIBotData(ABC):
    def __init__(self, name: str, recent_memory: MessageSnapshotHistory | None = None) -> None:
        self.name: str = name
        self.memory: MessageSnapshotHistory | None = recent_memory

class ReynardAIBotData(AbstractReynardAIBotData):
    def __init__(self,
                 *,
                 profile: Profile | None = None,
                 provider_store: providers.ProviderDataStore | None = None,
                 knowledge: KnowledgeIndex | None = None,
                 long_term_memory: LongTermMemoryIndex | None = None,
                 account_id: int = 0,
                 memory_length: int = 50,
                 api_key: str | None = None,
                 api_base: str = "https://api.openai.com/v1",
                 modules: list[BaseModule] | None = None
                ) -> None:
        if profile is None:
            profile = Profile()
            if api_key:
                profile.providers["openai"] = providers.ProviderData(
                    provider_name="openai",
                    api_base=api_base,
                    api_key=api_key
                )
        self.profile: Profile = profile

        if provider_store is None:
            provider_store = providers.ProviderDataStore(
                providers=list(self.profile.providers.values())
            )
        self.provider_store: providers.ProviderDataStore = provider_store

        super().__init__(self.profile.options.botname, MessageSnapshotHistory(memory_length=memory_length))
        
        self.account_id: int = account_id
        self.long_term_memory: LongTermMemoryIndex | None = long_term_memory
        self.knowledge: KnowledgeIndex | None = knowledge
        
        self.short_term_memory: SynchronizedMessageHistory = None
        self.medium_term_memory: SynchronizedMessageHistory | None = None
        self.update_memory_structures()

        if modules:
            for module in modules:
                module.register(self)

    def update_memory_structures(self) -> None:
        self.short_term_memory = SynchronizedMessageHistory(
            max_length=self.profile.memory_settings.short_term_history_length
        )
        self.medium_term_memory = None
        if self.profile.memory_settings.enable_medium_term_memory:
            self.medium_term_memory = SynchronizedMessageHistory(
                max_length=self.profile.memory_settings.medium_term_history_length
            )

    async def send_llm_request(self, *, provider_name: str, prompt: Prompt, parameter_set_name: str | None = None) -> Any:
        if parameter_set_name is None:
            parameter_set_name = provider_name
        params = self.profile.get_request_params(provider_name)
        provider: providers.ProviderData = self.profile.get_provider(parameter_set_name)
        client: LLMClient = LLMClient.from_provider(provider)
        return await client.send_request(prompt=prompt, params=params)

ReynardAIBot = ReynardAIBotData
