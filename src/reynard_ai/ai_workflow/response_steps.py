import time

from ..bot_data.ai_bot import AIBot
from abc import ABC, abstractmethod
from ..ai_workflow.response_logs import SimpleDebugLogger

class ResponseStep(ABC):
    def __init__(self, logger: SimpleDebugLogger):
        self.finished = False
        self.elapsed_ms: float | None = None
        self.logger = logger
    
    async def execute(self, bot_data: AIBot, message: str) -> str | None:
        self.bot_data: AIBot = bot_data
        self.message = message
        start = time.perf_counter()
        ret = await self._run()
        end = time.perf_counter()
        self.elapsed_ms = 1000 * (end - start)
        self.finished = True
        self.logger.verbose(f"Finished {self.get_name()} step in {self.elapsed_ms} ms", category="STEP FINISHED")
        return ret

    def _get_prompt(self, name: str):
        if name not in self.bot_data.profile.prompts:
            available_prompts = ", ".join(p for p in self.bot_data.profile.prompts.keys())
            raise ValueError(f"Failed to fetch required prompt {name}, it doesn't exist. Available prompts: {available_prompts}")
        else:
            return self.bot_data.profile.prompts[name]

    @abstractmethod
    async def _run(self) -> str | None:
        raise NotImplementedError("_run() for ResponseStep")
    
    @abstractmethod
    def get_name(self) -> str | None:
        raise NotImplementedError("get_name() for ResponseStep")
    
class PersonalityRewriteStep(ResponseStep):
    async def _run(self):
        NAME = "PERSONALITY_REWRITE"
        rewriter_prompt = self._get_prompt(NAME)
        prompt = rewriter_prompt.replace({
            "message": self.message
        })
        response = await self.bot_data.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {prompt}\nResposne: {response}", category=NAME) 
        return response.message.content
    
    def get_name(self) -> str | None:
        return "personality rewriter"
    
class UserQueryRephraseStep(ResponseStep):
    async def _run(self):
        NAME = "USER_QUERY_REPHRASE"
        recent_history_list = self.bot_data.short_term_memory.backing_history.as_list()
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in recent_history_list]
        )
        last_user = recent_history_list[-1].nick
        prompt = self._get_prompt(NAME).replace({
            "user_query": user_prompt_str, 
            "last_user": last_user
        })
        response = await self.bot_data.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content
    
    def get_name(self) -> str | None:
        return "query rephraser"
    
class HistorySummarizerStep(ResponseStep):
    async def _run(self):
        NAME = "HISTORY_SUMMARIZE"
        medium_term_memory_len = self.bot_data.profile.memory_settings.medium_term_history_length
        if self.bot_data.medium_term_memory is None:
            raise RuntimeError("Cannot summarize medium-term memory because this bot has the option disabled")
        msgs_to_summarize = self.bot_data.medium_term_memory.backing_history.as_list()[-medium_term_memory_len:]
        msgs_to_summarize_str = "\n".join(
            [memorized_message.text for memorized_message in msgs_to_summarize]
        )
        prompt = self.bot_data.profile.prompts[NAME].replace({
            "messages": msgs_to_summarize_str
        })
        response = await self.bot_data.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content
    
    def get_name(self) -> str | None:
        return "history summarizer"
    
class RelevantInfoSelectStep(ResponseStep):
    def __init__(self, *, logger: SimpleDebugLogger, user_query: str):
        super().__init__(logger)
        self.user_query = user_query

    async def _run(self):
        NAME = "INFO_SELECT"
        available_info = ""
        hits_list = await self.bot_data.knowledge.retrieve(self.user_query)

        if len(hits_list) == 0:
            return None

        for hits in hits_list:
            for hit in hits:
                available_info += hit["text"] + "\n"

        prompt = self._get_prompt(NAME) \
            .replace({
                "user_query": self.user_query,
                "available_info": available_info
            })
        response = await self.bot_data.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content
    
    def get_name(self) -> str | None:
        return "info selector"