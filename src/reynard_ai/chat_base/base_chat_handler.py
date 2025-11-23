import abc

from typing import Any
from ..bot_data.ai_bot import ReynardAIBotData
from ..events.event_bus import AsyncEventBus
from .message_snapshot import MessageSnapshot
from ..util.rate_limits import RateLimiter, RateLimit
from ..events.message_events import MessageSnapshotEvent
from ..ai_workflow.ai_responder import AIResponder
from ..ai_workflow.response_logs import ResponseLogsManager, SimpleDebugLogger

class BaseChatHandler(abc.ABC):
    def __init__(self, bus: AsyncEventBus[MessageSnapshotEvent], ai_bot: ReynardAIBotData):
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.ai_bot = ai_bot
        self.logger = SimpleDebugLogger(f"{self.__class__.__name__}Logger")
        bus.subscribe(self.on_message_received)

    async def on_message_received(self, event: MessageSnapshotEvent) -> None:
        snapshot = event.snapshot
        if self._is_bot_message(event):
            return
        if not self._is_message_for_bot(event):
            return

        self.rate_limiter.register_request(snapshot.sender_id)
        if self.rate_limiter.is_rate_limited(snapshot.sender_id):
            await self._send_rate_limit_warning(event)
            return

        verbose = snapshot.text.endswith("--v")
        logs = snapshot.text.endswith("--l")
        if logs:
            await self.handle_log_request(snapshot.text, event)
        else:
            await self.respond_with_llm(event, verbose=verbose)

    async def respond_with_llm(self, event: MessageSnapshotEvent, *, verbose: bool = False):
        user_msg_snapshot = event.snapshot
        await self.memorize_short_term(user_msg_snapshot, pending=True, add_after_id=None)

        typing_context = await self._send_typing_indicator(event)
        responder = AIResponder(self.ai_bot, event.chatroom, event.snapshot, verbose=verbose)
        resp = await responder.create_response()
        
        if resp.fail_exception is None:
            assert resp.ai_text is not None
            sent_message_snapshot = await self._send_response(resp.ai_text, event, typing_context)
            await self.memorize_short_term(sent_message_snapshot, pending=False, add_after_id=user_msg_snapshot.message_id)
            
            for to_memorize in [sent_message_snapshot, user_msg_snapshot]:
                await self.memorize_medium_term(to_memorize, pending=False)
                await self.memorize_long_term(to_memorize)
                await self.ai_bot.short_term_memory.mark_finalized(to_memorize.message_id)

            if self.ai_bot.medium_term_memory is not None:
                await self.ai_bot.medium_term_memory.mark_finalized(user_msg_snapshot.message_id)
                ResponseLogsManager.instance().store_log(sent_message_snapshot.message_id, resp.verbose_log_output)
        else:
            log_id = event.snapshot.message_id
            sent_message_snapshot = await self._send_reply(
                f"Error while generating response. The ID for this log is {log_id}: ```{str(resp.fail_exception)[:1000]}```",
                event
            )
            ResponseLogsManager.instance().store_log(log_id, resp.verbose_log_output)
            await self.ai_bot.short_term_memory.mark_finalized(user_msg_snapshot.message_id) # TODO: forget originating message on error?

    async def handle_log_request(self, content: str, original_event: MessageSnapshotEvent):
        try:
            num = None
            for num_str in content.split(" "):
                if num_str.isdigit():
                    num = int(num_str)
                    break

            if num is None:
                return await self._send_reply("❌ No numerical message ID found", original_event)

            log_data = ResponseLogsManager.instance().get_log_by_id(num)
            if log_data is None:
                return await self._send_reply(f"❌ No log with ID `{num}` found", original_event)

            await self._send_file_reply(
                content=f"Verbose logs for message ID {num} attached (only last 10 are stored)",
                file_data=log_data.encode('utf-8'),
                filename="verbose_log.txt",
                original_event=original_event
            )
        except ValueError:
            invalid_log_msg = self.ai_bot.profile.lang["invalid_log_request"].format(content)
            await self._send_reply(invalid_log_msg, original_event)

    async def memorize_short_term(self, message: MessageSnapshot, *, pending: bool, add_after_id: None | int) -> None:
        if add_after_id is None:
            await self.ai_bot.short_term_memory.add(
                message,
                pending=pending
            )
        else:
            await self.ai_bot.short_term_memory.add_after(
                add_after_id,
                message,
                pending=pending
            )

    async def memorize_medium_term(self, message: MessageSnapshot, *, pending: bool):
        if self.ai_bot.medium_term_memory is not None:
            await self.ai_bot.medium_term_memory.add(
                message,
                pending=pending
            )

    async def memorize_long_term(self, message: MessageSnapshot) -> None:
        if self.ai_bot.long_term_memory is not None:
            await self.ai_bot.long_term_memory.memorize(message)

    async def forget_message(self, message: MessageSnapshot) -> None:
        await self.ai_bot.short_term_memory.remove(message.message_id)
        if self.ai_bot.medium_term_memory is not None:
            await self.ai_bot.medium_term_memory.remove(message.message_id)

    @abc.abstractmethod
    def _is_bot_message(self, event: MessageSnapshotEvent) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def _is_message_for_bot(self, event: MessageSnapshotEvent) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _send_rate_limit_warning(self, event: MessageSnapshotEvent) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _send_typing_indicator(self, event: MessageSnapshotEvent) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _send_response(self, text: str, original_event: MessageSnapshotEvent, typing_placeholder: Any | None) -> MessageSnapshot:
        raise NotImplementedError()
    
    @abc.abstractmethod
    async def _send_reply(self, text: str, original_event: MessageSnapshotEvent) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _send_file_reply(self, content: str, file_data: bytes, filename: str, original_event: MessageSnapshotEvent) -> None:
        raise NotImplementedError()
