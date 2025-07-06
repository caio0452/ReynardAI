import io
import discord
from typing import Any
import message_processing_util

from .message_snapshot import MessageSnapshot
from .base_chat_handler import BaseChatHandler
from ..events.message_events import MessageSnapshotEvent

class DiscordChatHandler(BaseChatHandler):
    @staticmethod
    def _get_discord_msg(event: MessageSnapshotEvent) -> discord.Message:
        if isinstance(event.raw_msg_object, discord.Message):
            return event.raw_msg_object
        raise RuntimeError(f"Event has no valid Discord message object: {event.raw_msg_object}")

    def _is_bot_message(self, event: MessageSnapshotEvent) -> bool:
        message = self._get_discord_msg(event)
        return message.author.bot

    def _is_message_for_bot(self, event: MessageSnapshotEvent) -> bool:
        message = self._get_discord_msg(event)
        mention_ids = [user.id for user in message.mentions]
        return self.ai_bot.discord_bot_id in mention_ids

    async def _send_rate_limit_warning(self, event: MessageSnapshotEvent) -> None:
        message = self._get_discord_msg(event)
        await message.reply("⚠️ You are rate limited, please wait")

    async def _send_typing_indicator(self, event: MessageSnapshotEvent) -> discord.Message:
        message = self._get_discord_msg(event)
        return await message.reply(
            self.ai_bot.profile.lang["bot_typing"],
            mention_author=False,
        )

    async def _send_reply(self, text: str, original_event: MessageSnapshotEvent) -> None:
        message = self._get_discord_msg(original_event)
        await message.reply(content=text)

    async def _send_file_reply(self, content: str, file_data: bytes, filename: str, original_event: MessageSnapshotEvent) -> None:
        message = self._get_discord_msg(original_event)
        log_file = io.BytesIO(file_data)
        await message.reply(
            content=content,
            files=[discord.File(log_file, filename=filename)]
        )

    async def _send_response(self, text: str, original_event: MessageSnapshotEvent, typing_placeholder: Any | None) -> MessageSnapshot:
        typing_msg = typing_placeholder
        disclaimer = self.ai_bot.profile.lang.get("disclaimer", "")
        max_chunk_length = 1800 - len(disclaimer)

        def strip_newline(chunk):
            return chunk.strip('\r\n') if self.ai_bot.profile.options.remove_trailing_newline else chunk

        raw_chunks = message_processing_util.split_by_length(text, max_chunk_length)
        code_balanced_chunks = [
            f"{strip_newline(chunk)}{disclaimer}"
            for chunk in message_processing_util.balance_code_fences(raw_chunks)
        ]

        ping = not self.ai_bot.profile.options.only_ping_on_response_finish
        if typing_placeholder:
            if isinstance(typing_placeholder, discord.Message):
                typing_msg = typing_placeholder
                last_msg = await typing_msg.edit(content=code_balanced_chunks[0])
            else:
                raise RuntimeError(f"typing_placeholder is not a Discord message, it's {type(typing_placeholder)}")
        else:
            last_msg = await self._get_discord_msg(original_event).reply(code_balanced_chunks[0])

        for chunk in code_balanced_chunks[1:]:
            last_msg = await last_msg.reply(content=chunk, silent=not ping)

        return await MessageSnapshot.of_discord_message(last_msg)