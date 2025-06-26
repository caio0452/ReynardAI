import io
import discord
import traceback
from chat.chatroom import Chatroom
from chat.message_history import SynchronizedMessageHistory
import message_processing_util

from ..events.event_bus import AsyncEventBus
from .message_snapshot import MessageSnapshot
from ..util.rate_limits import RateLimiter, RateLimit
from ..events.message_events import MessageSnapshotEvent
from ..bot_workflow.ai_responder import CustomBotData, AIDiscordBotResponder
from ..bot_workflow.response_logs import ResponseLogsManager, SimpleDebugLogger

MSG_LOG_FILE_REPLY = "Verbose logs for message ID {} attached (only last 10 are stored)"

class DiscordChatHandler:
    def __init__(self, bus: AsyncEventBus[MessageSnapshotEvent], ai_bot_data: CustomBotData):
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.chatroom = Chatroom(SynchronizedMessageHistory(),0) # TODO: support per channel/server/DM chat, etc
        self.ai_bot = ai_bot_data
        self.logger = SimpleDebugLogger("ChatHandlerLogger")
        bus.subscribe(self.on_message)

    @staticmethod
    def _get_discord_msg(event: MessageSnapshotEvent) -> discord.Message:
        if isinstance(event.raw_msg_object, discord.Message):
            return event.raw_msg_object
        else:
            raise RuntimeError(f"MessageSnapshotEvent has no valid underlying discord Message object: {event.raw_msg_object} ({type(event.raw_msg_object)}). Message snapshot: {event.snapshot}")

    async def on_message(self, event: MessageSnapshotEvent) -> None:
        message = self._get_discord_msg(event)
        if message.author.bot: 
            return
        
        mention_ids = [user.id for user in message.mentions]
        if self.ai_bot.discord_bot_id not in mention_ids:
            return
        
        self.rate_limiter.register_request(event.snapshot.sender_id)
        if self.rate_limiter.is_rate_limited(event.snapshot.sender_id):
            await message.reply("⚠️ You are rate limited, please wait")
            return
        
        verbose = message.content.endswith("--v")
        logs = message.content.endswith("--l")
        if logs:
            await self.handle_log_request(message)
            return 
        
        await self.respond_with_llm(message, verbose=verbose)

    async def handle_log_request(self, message: discord.Message):
        try:
            num = None
            for num_str in message.content.split(" "):
                if num_str.isdigit():
                    num = int(num_str)
                    break

            if num is None:
                await message.reply("❌ No numerical message ID found")
                return
            
            log_data = ResponseLogsManager.instance().get_log_by_id(num)
            if log_data is None:
                await message.reply(f"❌ No log with ID `{num}` found")
                return
            log_file = io.BytesIO(log_data.encode('utf-8'))
            await message.reply(
                content=MSG_LOG_FILE_REPLY.format(num),
                files=[discord.File(log_file, filename="verbose_log.txt")]
            )
        except ValueError:
            invalid_log_msg = self.ai_bot.profile.lang["invalid_log_request"]
            await message.reply(invalid_log_msg.format(message.content))

    async def respond_with_llm(self, user_message: discord.Message, *, verbose: bool=False):
        await self.memorize_discord_message(user_message, pending=True, add_after_id=None)
        typing_msg = await user_message.reply(
            self.ai_bot.profile.lang["bot_typing"], 
            mention_author=False,
        )
        
        try:
            resp = await self.generate_response(user_message, verbose)
            if self.ai_bot.profile.options.only_ping_on_response_finish:
                base_resp_msg: discord.Message = await self.send_chunked_with_disclaimers(
                    resp.text,
                    reply_to=user_message,
                    edit_msg=None,
                    ping=self.ai_bot.profile.options.only_ping_on_response_finish
                )
                await typing_msg.delete()
            else:
                base_resp_msg: discord.Message = await self.send_chunked_with_disclaimers(
                    resp.text,
                    reply_to=None,
                    edit_msg=typing_msg,
                    ping=self.ai_bot.profile.options.only_ping_on_response_finish
                )

            if verbose:
                # TODO: this edit is potentially superfluous
                log_file = io.BytesIO(resp.verbose_log_output.encode('utf-8'))
                await base_resp_msg.edit(attachments=[discord.File(log_file, filename="log.txt")])
  
            await self.memorize_message(
                MessageSnapshot(
                    text=resp.text,  
                    nick=base_resp_msg.author.name,
                    sent=base_resp_msg.created_at,
                    is_bot=True,
                    sender_id=base_resp_msg.author.id,
                    message_id=base_resp_msg.id 
                ),
                pending=False,
                add_after_id=user_message.id
            )
            await self.ai_bot.recent_history.mark_finalized(user_message.id)
            ResponseLogsManager.instance().store_log(base_resp_msg.id, resp.verbose_log_output)
        except Exception as e:
            await self.handle_error(user_message, e)

    async def generate_response(self, to_respond: discord.Message, verbose: bool) -> AIBotResponder.Response:
        resp = AIDiscordBotResponder(self.ai_bot, to_respond, verbose)
        return await resp.create_response()

    async def send_chunked_with_disclaimers(self, resp_str: str, *, reply_to: discord.Message | None, edit_msg: discord.Message | None, ping: bool) -> discord.Message:
        disclaimer = self.ai_bot.profile.lang.get("disclaimer", "")
        max_chunk_length = 1800 - len(disclaimer)

        if reply_to is not None and edit_msg is not None:
            raise ValueError("Must specify one of reply_to or edit_msg, not both")
        def strip_newline(chunk):
            return chunk.strip('\r\n') if self.ai_bot.profile.options.remove_trailing_newline else chunk

        raw_chunks = message_processing_util.split_by_length(resp_str, max_chunk_length)
        code_balanced_chunks = [
            f"{strip_newline(chunk)}{disclaimer}" 
            for chunk in message_processing_util.balance_code_fences(raw_chunks)
        ]

        last_msg = None
        if edit_msg is not None:
            last_msg = await edit_msg.edit(content=code_balanced_chunks[0])
            remaining_chunks = raw_chunks[1:]
        elif reply_to is not None:
            if self.ai_bot.profile.options.only_ping_on_response_finish:
                last_msg = await reply_to.reply(content=code_balanced_chunks[0], silent=True)
            else:
                last_msg = await reply_to.reply(content=code_balanced_chunks[0], silent=not ping)
            remaining_chunks = raw_chunks[1:]
        else:
            raise ValueError("Must specify at least one of: reply_to or edit_msg")
        
        for chunk in remaining_chunks:
            last_msg = await last_msg.reply(content=chunk, silent=not ping)

        return last_msg
    
    async def memorize_message(self, message: MessageSnapshot, *, pending: bool, add_after_id: None | int) -> None:
        if add_after_id is None:
            await self.ai_bot.recent_history.add(
                message,
                pending=pending
            )
        else:
             await self.ai_bot.recent_history.add_after(
                add_after_id,
                message,
                pending=pending
            )
        if self.ai_bot.long_term_memory is not None:
            await self.ai_bot.long_term_memory.memorize(message)

    async def forget_message(self, message: MessageSnapshot) -> None:
        await self.ai_bot.recent_history.remove(message.message_id)
        
    async def memorize_discord_message(self, message: discord.Message, *, pending: bool, add_after_id: None | int) -> None:
        to_memorize = await MessageSnapshot.of_discord_message(message)
        await self.memorize_message(
            to_memorize,
            pending=pending,
            add_after_id=add_after_id
        )
        if self.ai_bot.long_term_memory is not None:
            await self.ai_bot.long_term_memory.memorize(to_memorize)

    async def handle_error(self, reply_to: discord.Message, error: Exception):
        await self.forget_message(await MessageSnapshot.of_discord_message(reply_to))
        await reply_to.reply(content=f"There was an error: ```{str(error)[:1000]}```") # TODO: send lang message if possible
        traceback.print_exc()
