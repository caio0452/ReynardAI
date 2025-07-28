from discord.ext import commands

from ..events.event_bus import AsyncEventBus
from ..events.message_events import MessageSnapshotEvent
from ..chat.message_snapshot import MessageSnapshot

class DiscordBridge(commands.Cog):
    def __init__(self, bot: commands.Bot, bus: AsyncEventBus[MessageSnapshotEvent]):
        self.bot = bot
        self.bus = bus

    @commands.Cog.listener()
    async def on_message(self, message):
        # Don't fire events for bots
        if message.author.bot:
            return

        snapshot = await MessageSnapshot.of_discord_message(message)
        await self.bus.publish(
            MessageSnapshotEvent(
                backend="discord", snapshot=snapshot, raw_msg_object=message
            )
        )