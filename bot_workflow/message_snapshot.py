from dataclasses import dataclass, field
import datetime
import discord

@dataclass
class MessageSnapshot:
    text: str
    nick: str
    is_bot: bool
    message_id: int
    sent: datetime.datetime
    attachment_urls: list[str] = field(default_factory=list)

    def __str__(self):
        formatted_time = datetime.datetime.strftime(self.sent, "%Y-%m-%d %H:%M:%S")
        return f"[{formatted_time}] {self.nick}: {self.text}"

    @staticmethod
    async def of_discord_message(message: discord.Message, message_sanitizer = None) -> "MessageSnapshot":
        if message_sanitizer is not None:
            text = await message_sanitizer(message)
        else:
            text = message.content

        return MessageSnapshot(
            text=f"[{message.created_at.strftime('%d/%m %H:%M:%S')} by {message.author.display_name}] {text}",
            nick=message.author.display_name,
            sent=message.created_at,
            is_bot=message.author.bot,
            message_id=message.id
        )