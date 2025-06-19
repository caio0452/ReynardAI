import io
import re
import discord
import traceback

from io import StringIO
from discord.ext import commands
from util.rate_limits import RateLimiter, RateLimit
from bot_workflow.message_snapshot import MessageSnapshot
from bot_workflow.ai_bot import CustomBotData, AIDiscordBotResponder
from bot_workflow.response_logs import ResponseLogsManager, SimpleDebugLogger
from bot_workflow.discord_message_parser import DiscordMessageParser, DenialReason, SpecialFunctionFlags

MSG_LOG_FILE_REPLY = "Verbose logs for message ID {} attached (only last 10 are stored)"

class DiscordChatHandler(commands.Cog):
    def __init__(self, discord_bot: commands.Bot, ai_bot_data: CustomBotData):
        self.bot: commands.Bot = discord_bot
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.message_parser = DiscordMessageParser(self.bot)
        self.ai_bot = ai_bot_data
        self.logger = SimpleDebugLogger("ChatHandlerLogger")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot: 
            return
        
        ctx = self.message_parser.parse_message(message)
        if ctx.denial_reason == DenialReason.DID_NOT_PING:
            return
        if ctx.denial_reason == DenialReason.RATE_LIMITED:
            await message.reply("You are rate limited, please wait")
            return
        if ctx.denial_reason == DenialReason.TOO_LONG:
            for emoji in ['🇹', '🇱', '🇩', '🇷']:
                await message.add_reaction(emoji)
            return
        if SpecialFunctionFlags.VIEW_MESSAGE_LOGS in ctx.called_functions:
            await self.handle_log_request(message)
            return
        
        verbose = SpecialFunctionFlags.REQUEST_VERBOSE_REPLY in ctx.called_functions
        await self.respond_with_llm(message, verbose=verbose)

    async def handle_log_request(self, message: discord.Message):
        ctx = self.message_parser.parse_message(message)
        try:
            num = None
            for num_str in ctx.sanitized_content.split(" "):
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
            await message.reply(invalid_log_msg.format(ctx.sanitized_content))

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
                log_file = StringIO(resp.verbose_log_output)
                await base_resp_msg.edit(attachments=[discord.File(log_file, filename="log.txt")])
  
            await self.memorize_message(
                MessageSnapshot(
                    text=resp.text,  
                    nick=base_resp_msg.author.name,
                    sent=base_resp_msg.created_at,
                    is_bot=True,
                    message_id=base_resp_msg.id 
                ),
                pending=False,
                add_after_id=user_message.id
            )
            await self.ai_bot.recent_history.mark_finalized(user_message.id)
            ResponseLogsManager.instance().store_log(base_resp_msg.id, resp.verbose_log_output)
        except Exception as e:
            await self.handle_error(user_message, e)

    async def generate_response(self, to_respond: discord.Message, verbose: bool) -> AIDiscordBotResponder.Response:
        resp = AIDiscordBotResponder(self.ai_bot, to_respond, verbose)
        return await resp.create_response()

    def _chunk_by_length_and_spaces(self, full_text: str, max_chunk_length: int) -> list[str]:
        chunks: list[str] = []
        current_chunk = ""
        partial_leftover_word = ""
        words = full_text.split(" ") 

        for word in words:
            if len(partial_leftover_word) > 0:
                current_chunk += partial_leftover_word + " "
                partial_leftover_word = ""

            if len(current_chunk) + len(word) <= max_chunk_length:
                current_chunk += " " + word
            else:
                # If the word is very large, maybe this is not regular text that's meant to be split in spaces
                if len(word) < max_chunk_length // 2:
                    len_until_max = max_chunk_length - len(word)
                    partial_word = word[0:len_until_max]
                    current_chunk += partial_word
                    partial_leftover_word = word.replace(partial_word, "")
                else:
                    chunks.append(current_chunk)
                    current_chunk = ""

        if current_chunk != "":
            chunks.append(current_chunk)
            
        return chunks
    
    def _balance_code_block_fences(self, *, original_text: str, chunked_text: list[str]) -> list[str]:
        fence_pattern = re.compile(r"```(\w+)?")
        balanced: list[str] = []
        inside_code = False

        def last_fence_language(text: str, *, before_pos: int) -> str:
            matches = list(fence_pattern.finditer(text, 0, before_pos))
            if not matches:
                return ""
            lang = matches[-1].group(1)
            return lang or ""

        cursor = 0
        for chunk in chunked_text:
            fence_count = len(fence_pattern.findall(chunk))
            
            if fence_count % 2 == 1:
                inside_code = not inside_code
                chunk += "\n```"

            # If inside a code block after closing, next chunk must reopen
            if inside_code:
                lang = last_fence_language(original_text, before_pos=cursor)
                chunk = f"```{lang}\n" + chunk

            balanced.append(chunk)
            cursor += len(chunk)

        return balanced

    async def send_chunked_with_disclaimers(self, resp_str: str, *, reply_to: discord.Message | None, edit_msg: discord.Message | None, ping: bool) -> discord.Message:
        disclaimer = self.ai_bot.profile.lang.get("disclaimer", "")
        max_chunk_length = 1800 - len(disclaimer)

        if reply_to is not None and edit_msg is not None:
            raise ValueError("Must specify one of reply_to or edit_msg, not both")
        def strip_newline(chunk):
            return chunk.strip('\r\n') if self.ai_bot.profile.options.remove_trailing_newline else chunk

        raw_chunks = self._chunk_by_length_and_spaces(resp_str, max_chunk_length)
        '''
        code_balanced_chunks = [
            f"{strip_newline(chunk)}{disclaimer}" 
            for chunk in self._balance_code_block_fences(original_text=resp_str, chunked_text=raw_chunks)
        ]
        ''' # TODO: readd

        last_msg = None
        if edit_msg is not None:
            last_msg = await edit_msg.edit(content= raw_chunks[0])
            remaining_chunks = raw_chunks[1:]
        elif reply_to is not None:
            if self.ai_bot.profile.options.only_ping_on_response_finish:
                last_msg = await reply_to.reply(content= raw_chunks[0], silent=True)
            else:
                last_msg = await reply_to.reply(content= raw_chunks[0], silent=not ping)
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
        # TODO: implement message forgetting
        # await self.forget_message(message)
        # await self.forget_message(reply)
        traceback.print_exc()
        await reply_to.reply(content=f"There was an error: ```{str(error)[:1000]}```") # TODO: send lang message if possible