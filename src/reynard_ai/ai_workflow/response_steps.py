import time
import json
import base64
import requests

from ..bot_data.ai_bot import AIBot
from abc import ABC, abstractmethod
from ..ai_workflow.response_logs import SimpleDebugLogger

class ResponseStep(ABC):
    def __init__(self, logger: SimpleDebugLogger):
        self.finished = False
        self.elapsed_ms: float | None = None
        self.logger = logger
    
    async def execute(self, ai_bot: AIBot, message: str) -> str | None:
        self.ai_bot: AIBot = ai_bot
        self.message = message
        start = time.perf_counter()
        ret = await self._run()
        end = time.perf_counter()
        self.elapsed_ms = 1000 * (end - start)
        self.finished = True
        self.logger.verbose(f"Finished {self.get_name()} step in {self.elapsed_ms} ms", category="STEP FINISHED")
        return ret

    def _get_prompt(self, name: str):
        if name not in self.ai_bot.profile.prompts:
            available_prompts = ", ".join(p for p in self.ai_bot.profile.prompts.keys())
            raise ValueError(f"Failed to fetch required prompt {name}, it doesn't exist. Available prompts: {available_prompts}")
        else:
            return self.ai_bot.profile.prompts[name]

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
        prompt = rewriter_prompt.replace_or_throw({
            "message": self.message
        })
        response = await self.ai_bot.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {json.dumps(prompt.messages, indent=4)}\nResponse: {response}", category=NAME) 
        return response.message.content
    
    def get_name(self) -> str | None:
        return "personality rewriter"
    
class UserQueryRephraseStep(ResponseStep):
    async def _run(self):
        NAME = "USER_QUERY_REPHRASE"
        recent_history_list = self.ai_bot.short_term_memory.backing_history.as_list()
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in recent_history_list]
        )
        last_user = recent_history_list[-1].nick
        prompt = self._get_prompt(NAME).replace({
            "user_query": user_prompt_str, 
            "last_user": last_user
        })
        response = await self.ai_bot.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {json.dumps(prompt.messages, indent=4)}\nResponse: {response}", category=NAME)
        return response.message.content
    
    def get_name(self) -> str | None:
        return "query rephraser"
    
class HistorySummarizerStep(ResponseStep):
    async def _run(self):
        NAME = "HISTORY_SUMMARIZE"
        medium_term_memory_len = self.ai_bot.profile.memory_settings.medium_term_history_length
        if self.ai_bot.medium_term_memory is None:
            raise RuntimeError("Cannot summarize medium-term memory because this bot has the option disabled")
        msgs_to_summarize = self.ai_bot.medium_term_memory.backing_history.as_list()[-medium_term_memory_len:]
        msgs_to_summarize_str = "\n".join(
            [memorized_message.text for memorized_message in msgs_to_summarize]
        )
        prompt = self.ai_bot.profile.prompts[NAME].replace_or_throw({
            "messages": msgs_to_summarize_str
        })
        response = await self.ai_bot.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {json.dumps(prompt.messages, indent=4)}\nResponse: {response}", category=NAME)
        return response.message.content
    
    def get_name(self) -> str | None:
        return "history summarizer"
    
class AttachmentDescribeStep(ResponseStep):
    def __init__(self, *, logger: SimpleDebugLogger, attachment_urls: list[str]):
        super().__init__(logger=logger)
        self.attachment_urls = attachment_urls

    async def _run(self):
        NAME = "ATTACHMENT_DESCRIBE"
        attachment_urls = self.attachment_urls
        
        if len(attachment_urls) == 0:
            raise RuntimeError("Cannot run attachment description step with no attachments")
        if len(attachment_urls) > 1:
            return "I cannot view multiple attachments, please attach at most one."
        
        attachment_url = attachment_urls[0]
        try:
            response = requests.get(attachment_url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.logger.verbose(f"Failed to download attachment from {attachment_url}: {e}", category=NAME)
            return "Failed to download image, so you are unable to see it. Please warn the user about this"
            
        ctype = response.headers.get("Content-Type", "").lower()
        if ctype not in ["image/png", "image/jpeg"]:
            self.logger.verbose(f"Unsupported content type '{ctype}', ignoring attachment description for {attachment_url}", category=NAME)
            return "Unsupported image format, so you are unable to see it. Please warn the user about this"
        
        image_data = response.content
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:{ctype};base64,{base64_data}"

        prompt = self._get_prompt(NAME).replace(
            {"attachment_url": data_url}
        )
        self.logger.verbose(f"Prompt (messages truncated): {json.dumps(prompt.messages, indent=4)}")
        response = await self.ai_bot.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        return response.message.content
    
    def get_name(self) -> str | None:
        return "attachment describer"

class RelevantInfoSelectStep(ResponseStep):
    def __init__(self, *, logger: SimpleDebugLogger, user_query: str):
        super().__init__(logger)
        self.user_query = user_query

    async def _run(self):
        NAME = "INFO_SELECT"
        available_info = ""
        hits_list = await self.ai_bot.knowledge.retrieve(self.user_query)

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
        response = await self.ai_bot.send_llm_request(
            provider_name=NAME,
            prompt=prompt,
            parameter_set_name=NAME
        )
        self.logger.verbose(f"Prompt: {json.dumps(prompt.messages, indent=4)}\nResponse: {response}", category=NAME)
        return response.message.content
    
    def get_name(self) -> str | None:
        return "info selector"