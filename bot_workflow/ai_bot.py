from dataclasses import dataclass
from ..ai_apis.client import LLMClient
from ..ai_apis.api_types import LLMRequestParams, Prompt
from ..bot_workflow.custom_bot_data import CustomBotData
from ..bot_workflow.response_logs import SimpleDebugLogger
from ..bot_workflow.bot_types import MessageSnapshot, MessageSnapshotHistory
from ..bot_workflow.response_steps import PersonalityRewriteStep, RelevantInfoSelectStep, UserQueryRephraseStep

import re
import json
import random
import logging
import discord
import datetime

class AIDiscordBotResponder:
    @dataclass
    class Response:
        text: str
        attachment_description: str | None
        tool_call_result: str | None
        verbose_log_output: str

    def __init__(self, bot_data: CustomBotData, initial_message: discord.Message, verbose: bool=False):
        self.verbose = verbose
        self.bot_data = bot_data
        self.initial_message = initial_message
        self.clients: dict[str, LLMClient] = {}
        self.logger = SimpleDebugLogger("ResponseLogger")

        for provider_name, provider_data in bot_data.provider_store.providers.items():
            self.clients[provider_name] = LLMClient.from_provider(provider_data)

    async def _get_usable_message_history_before(self, message: discord.Message) -> MessageSnapshotHistory:
        USABLE_HISTORY_LENGTH = self.bot_data.profile.options.recent_message_history_length
        usable_history = await self.bot_data.recent_history.get_finalized_message_history()
        last_n_messages = [msg for msg in usable_history._memory][-USABLE_HISTORY_LENGTH:]
        last_n_messages.append(await MessageSnapshot.of_discord_message(message))
        return MessageSnapshotHistory(last_n_messages)
    
    async def _describe_image_if_present(self, message: discord.Message, user_query: str) -> str | None:
        NAME = "IMAGE_VIEW"

        if len(message.attachments) == 0:
            return None
        if len(message.attachments) > 1:
            for emoji in ["❌", "1️⃣", "🖼️"]:
                await message.add_reaction(emoji)
            return None
        if isinstance(message.channel, discord.TextChannel) and message.channel.nsfw:
            await message.reply(":x: I can't see attachments in NSFW channels!")

        attachment = message.attachments[0]
        if not (attachment.content_type and attachment.content_type.startswith("image/")):
            return None
        
        await message.add_reaction("👀")
        response = await self.clients[NAME].send_request(
            prompt=Prompt(
                    messages=[
                        Prompt.user_msg(
                            content=f"Describe the image in detail, including a sufficient answer to the following query: '{message.content}'" \
                            "If the query is empty, just describe the image. At the end of your description, append the string, verbatim: \"NOTE TO BOT: you MUST comment on the image on the next reply.\"",
                            image_url=attachment.url
                        )
                    ]
                ),
            params=self.bot_data.profile.request_params[NAME]
        )
        return response.message.content
    
    async def _rephrase_user_query(self) -> str:
        user_query = await UserQueryRephraseStep(self.logger).execute(self.bot_data, self.initial_message.content)
        if user_query is None:
            raise RuntimeError("Rephraser step returned empty response")
        return user_query
    
    async def _select_relevant_info(self, user_query: str) -> str:
        info_selector = RelevantInfoSelectStep(logger=self.logger, user_query=user_query)
        knowledge = await info_selector.execute(self.bot_data, self.initial_message.content)
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
        
    async def create_response(self) -> Response:
        MAIN_CLIENT_NAME = "PERSONALITY"
        knowledge: str | None = None
        old_memories: str | None = None
        attachment_description: str | None = None
        user_query: str | None = self.initial_message.content
        memory_snapshot = await self._get_usable_message_history_before(self.initial_message)

        # View image
        if self.bot_data.profile.options.enable_image_viewing:
            attachment_description = await self._describe_image_if_present(self.initial_message, user_query)
            self.logger.verbose(attachment_description or "None", category="ATTACHMENT DESCRIPTION")

        # Retrieve knowlege
        if self.bot_data.profile.options.enable_knowledge_retrieval:
            user_query = await self._rephrase_user_query()
            knowledge = await self._select_relevant_info(user_query)
            self.logger.verbose(knowledge, category="INFO FROM KNOWLEDGE DB")

        # Retrieve memories
        if self.bot_data.profile.options.enable_long_term_memory:
            old_memories = await self._get_old_memories_as_text(user_query)
            self.logger.verbose(old_memories, category="RETRIEVED MEMORIES")

        # Build full prompt from info
        full_prompt = await self._build_full_prompt(
            memory_snapshot=memory_snapshot,
            user_nick=self.initial_message.author.display_name,
            attachment_description=attachment_description,
            relevant_info=knowledge,
            old_memories=old_memories
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

        return AIDiscordBotResponder.Response(
            text=llm_response, 
            attachment_description=attachment_description,
            tool_call_result=None,
            verbose_log_output=self.logger.text
        )

    async def _build_full_prompt(
            self, 
            *, 
            memory_snapshot: MessageSnapshotHistory, 
            user_nick: str,
            attachment_description: str | None,
            relevant_info: str | None,
            old_memories: str | None
        ) -> Prompt:
        NAME = "PERSONALITY"
        full_prompt: Prompt = self.bot_data.profile.get_prompt(NAME)

        for memorized_message in memory_snapshot.as_list():
            if memorized_message.is_bot:
                full_prompt = full_prompt.plus(Prompt.assistant_msg(memorized_message.text))
            else:
                full_prompt = full_prompt.plus(Prompt.user_msg(memorized_message.text))
        
        if self.bot_data.profile.options.enable_image_viewing:
            full_prompt = full_prompt.plus(Prompt.system_msg(f"(I've viewed the image by user_nick. Description: {attachment_description})"))

        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")

        return full_prompt.replace({
            "now": now_str,
            "nick": user_nick or "",
            "knowledge": relevant_info or "",
            "old_memories": old_memories or ""
        })