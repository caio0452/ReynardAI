from __future__ import annotations
from typing import override, TYPE_CHECKING
from .knowledge import KnowledgeIndex, LongTermMemoryIndex

if TYPE_CHECKING:
    from .ai_bot import ReynardAIBotData

class BaseModule:
    def register(self, bot: ReynardAIBotData) -> None:
        raise NotImplementedError("register")

class MemoryModule(BaseModule):
    def __init__(self, *, 
                 short_term_length: int = 50, 
                 enable_medium_term: bool = False, 
                 medium_term_length: int = 20, 
                 enable_long_term: bool = False,
                 long_term_memory_index: LongTermMemoryIndex | None = None) -> None:
        self.short_term_length: int = short_term_length
        self.enable_medium_term: bool = enable_medium_term
        self.medium_term_length: int = medium_term_length
        self.enable_long_term: bool = enable_long_term
        self.long_term_memory_index: LongTermMemoryIndex | None = long_term_memory_index

    @override
    def register(self, bot: ReynardAIBotData) -> None:
        bot.profile.memory_settings.short_term_history_length = self.short_term_length
        bot.profile.memory_settings.enable_medium_term_memory = self.enable_medium_term
        bot.profile.memory_settings.medium_term_history_length = self.medium_term_length
        bot.profile.memory_settings.enable_long_term_memory = self.enable_long_term
        if self.long_term_memory_index is not None:
            bot.long_term_memory = self.long_term_memory_index
        bot.update_memory_structures()

class KnowledgeModule(BaseModule):
    def __init__(self, *, 
                 enable_retrieval: bool = True, 
                 enable_llm_summarization: bool = False, 
                 knowledge_index: KnowledgeIndex | None = None) -> None:
        self.enable_retrieval: bool = enable_retrieval
        self.enable_summarization: bool = enable_llm_summarization
        self.knowledge_index: KnowledgeIndex | None = knowledge_index

    @override
    def register(self, bot: ReynardAIBotData) -> None:
        bot.profile.options.enable_knowledge_retrieval = self.enable_retrieval
        bot.profile.options.enable_knowledge_summarization = self.enable_summarization
        if self.knowledge_index is not None:
            bot.knowledge = self.knowledge_index

class ModerationModule(BaseModule):
    def __init__(self, *, 
                 enabled: bool = True, 
                 moderator_prompt: str | None = None,
                 moderator_provider_name: str = "MODERATOR") -> None:
        self.enabled: bool = enabled
        self.moderator_prompt: str | None = moderator_prompt
        self.moderator_provider_name: str = moderator_provider_name

    @override
    def register(self, bot: ReynardAIBotData) -> None:
        bot.profile.options.enable_moderation = self.enabled
        if self.moderator_prompt:
            from ..ai_apis.api_types import Prompt
            bot.profile.prompts[self.moderator_provider_name] = Prompt(messages=(Prompt.system_msg(self.moderator_prompt),))

class ImageGenModule(BaseModule):
    def __init__(self, *, 
                 enabled: bool = True, 
                 api_key: str = "", 
                 model_name: str = "") -> None:
        self.enabled: bool = enabled
        self.api_key: str = api_key
        self.model_name: str = model_name

    @override
    def register(self, bot: ReynardAIBotData) -> None:
        bot.profile.fal_image_gen_config.enabled = self.enabled
        bot.profile.fal_image_gen_config.api_key = self.api_key
        bot.profile.fal_image_gen_config.model_name = self.model_name
