from dataclasses import dataclass
from ..chat.chatroom import Chatroom
from ..ai_apis.client import LLMClient
from .custom_bot_data import CustomBotData
from .response_logs import SimpleDebugLogger
from ..chat.message_snapshot import MessageSnapshot
from ..ai_apis.api_types import LLMRequestParams, Prompt
from ..chat.message_history import MessageSnapshotHistory
from .response_steps import HistorySummarizerStep, PersonalityRewriteStep, RelevantInfoSelectStep, UserQueryRephraseStep

import re
import json
import random
import asyncio
import logging
import datetime

class AIResponder:
    @dataclass
    class Response:
        text: str
        attachment_description: str | None
        tool_call_result: str | None
        verbose_log_output: str

    def __init__(self, bot_data: CustomBotData,chatroom: Chatroom, last_msg_snapshot: MessageSnapshot, verbose: bool=False):
        self.verbose = verbose
        self.last_msg_snapshot = last_msg_snapshot
        self.bot_data = bot_data
        self.clients: dict[str, LLMClient] = {}
        self.logger = SimpleDebugLogger("ResponseLogger")
        self.chatroom = chatroom

        for provider_name, provider_data in bot_data.provider_store.providers.items():
            self.clients[provider_name] = LLMClient.from_provider(provider_data)

    async def _get_recent_usable_message_history(self) -> MessageSnapshotHistory:
        USABLE_HISTORY_LENGTH = self.bot_data.profile.memory_settings.short_term_history_length
        full_history = await self.chatroom.message_history.get_finalized_message_history()
        last_n_messages = [msg for msg in full_history._memory][-USABLE_HISTORY_LENGTH:]
        return MessageSnapshotHistory(last_n_messages)
    
    async def _describe_image_if_present(self, attachment_url: str | None, user_query: str) -> str | None:
        NAME = "IMAGE_VIEW"
        valid_extensions = [".png", ".jpg", ".jpeg"]
        if attachment_url is None:
            return None
        if not any(attachment_url.endswith(ext) for ext in valid_extensions):
            return None
        
        description_msg = Prompt.user_msg(
            content=f"Describe the image in detail, including a sufficient answer to the following query: '{user_query}'" \
            "If the query is empty, just describe the image. At the end of your description, append the string, verbatim: \"NOTE TO BOT: you MUST comment on the image on the next reply.\"",
            image_url=attachment_url
        )
        response = await self.clients[NAME].send_request(
            prompt=Prompt(messages=[description_msg]), # type: ignore
            params=self.bot_data.profile.request_params[NAME]
        )
        return response.message.content
    
    async def _rephrase_user_query(self) -> str:
        user_query = await UserQueryRephraseStep(self.logger).execute(self.bot_data, self.last_msg_snapshot.text)
        if user_query is None:
            raise RuntimeError("Rephraser step returned empty response")
        return user_query
    
    async def _get_medium_term_summary(self) -> str:
        summary =  await HistorySummarizerStep(self.logger).execute(
            self.bot_data, self.last_msg_snapshot.text) 
        if summary is None:
            raise RuntimeError("History summarizer step returned empty response")
        return summary
        
    async def _select_relevant_info(self, user_query: str) -> str:
        info_selector = RelevantInfoSelectStep(logger=self.logger, user_query=user_query)
        knowledge = await info_selector.execute(self.bot_data, self.last_msg_snapshot.text)
        if knowledge is None:
            raise RuntimeError("Knowledge retrieval step returned empty response")
        return knowledge

    async def _get_old_memories_as_text(self, user_query: str) -> str:
        old_memories = ""
        if self.bot_data.long_term_memory is not None:
            for hit in await self.bot_data.long_term_memory.get_closest_messages(user_query):
                old_memories += hit.entity["text"] + "\n"
        return old_memories
    
    async def _personality_rewrite(self, llm_response: str) -> str:
        personality_rewriter = PersonalityRewriteStep(self.logger)
        personality_rewrite = await personality_rewriter.execute(self.bot_data, llm_response) 
        if personality_rewrite is None:
            raise RuntimeError("Personality rewrite step returned empty response")
        return personality_rewrite
        
    @dataclass
    class PromptData:
        attachment_description: str | None
        knowledge: str | None
        old_memories: str | None
        medium_term_summary: str | None

    async def _gather_prompt_data(self)-> PromptData:
        user_query: str = self.last_msg_snapshot.text
        memory_snapshot = await self._get_recent_usable_message_history()
        await memory_snapshot.add(self.last_msg_snapshot)

        # Read attachments
        attachment_task = None
        if self.bot_data.profile.options.enable_image_viewing and self.last_msg_snapshot.attachment_urls:
            attachment_task = asyncio.create_task(
                self._describe_image_if_present(
                    self.last_msg_snapshot.attachment_urls[0], 
                    user_query
                )
            )

        # Rephrase then gather knowledge
        knowledge_task = None
        if self.bot_data.profile.options.enable_knowledge_retrieval:
            async def _fetch_knowledge():
                rephrased_query = await self._rephrase_user_query()
                return await self._select_relevant_info(rephrased_query)
            knowledge_task = asyncio.create_task(_fetch_knowledge())

        # Retrieve old memories
        old_memories_task = None
        if self.bot_data.profile.memory_settings.enable_long_term_memory:
            old_memories_task = asyncio.create_task(
                self._get_old_memories_as_text(user_query)
            )

        # Summarize message history ("medium-term memory")
        medium_term_task = None
        if self.bot_data.profile.memory_settings.enable_medium_term_memory:
            medium_term_task = asyncio.create_task(
                self._get_medium_term_summary()
            )

        # Dispatch all tasks
        task_dict = {
            'attachment': attachment_task,
            'knowledge': knowledge_task,
            'old_memories': old_memories_task,
            'medium_term': medium_term_task
        }

        valid_tasks = {
            name: task 
            for name, task in task_dict.items() 
            if task is not None
        }
        results = await asyncio.gather(*valid_tasks.values())
        result_dict = dict(zip(valid_tasks.keys(), results))

        return AIResponder.PromptData(
            attachment_description = result_dict.get('attachment'),
            knowledge = result_dict.get('knowledge'),
            old_memories = result_dict.get('old_memories'),
            medium_term_summary = result_dict.get('medium_term')
        )
    
    async def create_response(self) -> Response:
        MAIN_CLIENT_NAME = "PERSONALITY"
        prompt_data = await self._gather_prompt_data()
        memory_snapshot = await self._get_recent_usable_message_history()

        # Build full prompt from info
        full_prompt = await self._format_full_prompt(
            memory_snapshot=memory_snapshot,
            user_nick=self.last_msg_snapshot.nick,
            attachment_description=prompt_data.attachment_description,
            relevant_info=prompt_data.knowledge,
            old_memories=prompt_data.old_memories,
            medium_term_summary=prompt_data.medium_term_summary
        )
        self.logger.verbose(json.dumps(full_prompt.messages), category="FULL_PROMPT")

        # Formulate responses w/ full prompt
        main_client_params = self.bot_data.profile.request_params[MAIN_CLIENT_NAME]
        model_names_order = [main_client_params.model_name] + self.bot_data.profile.options.llm_fallbacks
        llm_response = None
        for name in model_names_order:
            modified_params = main_client_params.model_copy(deep=True)
            modified_params = LLMRequestParams(
                model_name=name,
                temperature=main_client_params.temperature,
                max_tokens=main_client_params.max_tokens,
                logit_bias=main_client_params.logit_bias
            )
            self.logger.verbose(f"Sending request to model name '{name}' with parameters {modified_params.model_dump_json()}", category="REQUEST")
            try:
                raw_response = await self.clients[MAIN_CLIENT_NAME].send_request(
                    prompt=full_prompt,
                    params=modified_params
                )
                llm_response = raw_response.message.content
                self.logger.verbose(f"{raw_response}", category="FULL RESPONSE")
                break
            except Exception as e:
                self.logger.verbose(f"Request to LLM '{name}' failed with error: {e}", category="MODEL FAILURE")
                logging.exception(e)
        if llm_response is None:
            raise RuntimeError("Cannot generate response and all fallbacks failed")
        
        # Rewrite in-character
        if self.bot_data.profile.options.enable_personality_rewrite:
            llm_response = await self._personality_rewrite(llm_response)
        
        # Replace undesirable text
        for target, replacement_obj in self.bot_data.profile.regex_replacements.items():
            if isinstance(replacement_obj, list):
                replacement = random.choice(replacement_obj)
            else:
                replacement = replacement_obj
            llm_response = re.sub(target, replacement, llm_response)
        self.logger.verbose(f"Sanitized text, result: {llm_response}", category="REGEX REPLACEMENT")

        return AIResponder.Response(
            text=llm_response, 
            attachment_description=prompt_data.attachment_description,
            tool_call_result=None,
            verbose_log_output=self.logger.text
        )

    async def _format_full_prompt(
            self, 
            *, 
            memory_snapshot: MessageSnapshotHistory, 
            user_nick: str,
            attachment_description: str | None,
            relevant_info: str | None,
            old_memories: str | None,
            medium_term_summary: str | None
        ) -> Prompt:
        NAME = "PERSONALITY"
        full_prompt: Prompt = self.bot_data.profile.get_prompt(NAME)

        for memorized_message in memory_snapshot.as_list():
            if memorized_message.is_bot:
                full_prompt = full_prompt.plus(Prompt.assistant_msg(memorized_message.text))
            else:
                full_prompt = full_prompt.plus(Prompt.user_msg(memorized_message.text))
        
        if self.bot_data.profile.options.enable_image_viewing and attachment_description is not None:
            full_prompt = full_prompt.plus(Prompt.system_msg(f"(I've viewed the image by {user_nick}. Description: {attachment_description})"))

        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")

        return full_prompt.replace({
            "now": now_str,
            "nick": user_nick or "",
            "knowledge": relevant_info or "",
            "old_memories": old_memories or "",
            "summary": medium_term_summary or "",
        })