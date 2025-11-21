from discord.ext import commands

from ..bot_data.ai_bot import ReynardAIBotData
from ..chat_base.base_chat_handler import BaseChatHandler
from ..events.event_bus import AsyncEventBus
from .discord_events_bridge import DiscordBridge
from .discord_chat_handler import DiscordChatHandler

class ReynardChatBot():
    def __init__(self, bot_data: ReynardAIBotData, chat_handler: BaseChatHandler, event_bus: AsyncEventBus | None = None):
        if event_bus is None:
            event_bus = AsyncEventBus()
            event_bus.start()
        self.bot_data = bot_data
        self.event_bus = event_bus
        self.chat_handler = chat_handler 

    @classmethod
    async def create_discord_bot(cls, discord_bot: commands.Bot, ai_bot_data: ReynardAIBotData):
        event_bus = AsyncEventBus()
        chat_handler = DiscordChatHandler(
            bus=event_bus,
            ai_bot=ai_bot_data
        )
        discord_bridge = DiscordBridge(
            discord_bot, 
            bus=event_bus, 
            known_chatrooms=[]
        )
        event_bus.start() 
        await discord_bot.add_cog(discord_bridge)
        return cls(ai_bot_data, chat_handler, event_bus)
