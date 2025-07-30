import discord
from typing import Any
from discord.ext import commands
from abc import ABC, abstractmethod

from ..chat.chatroom import Chatroom
from ..events.event_bus import AsyncEventBus
from ..chat.message_snapshot import MessageSnapshot
from ..events.message_events import MessageSnapshotEvent
from ..chat.message_history import SynchronizedMessageHistory

class MessageToChatroomMapper(ABC):
    def __init__(self, known_chatrooms: list[Chatroom]):
        self.known_chatrooms = known_chatrooms

    @abstractmethod
    def find_chatroom(self, raw_message_object: Any) -> Chatroom | None:
        raise NotImplementedError("find_chatroom")
    
class DiscordMessageToChatroomMapper(MessageToChatroomMapper):
    def find_chatroom(self, raw_message_object: Any, create_if_absent: bool = True) -> Chatroom | None:
        if not isinstance(raw_message_object, discord.Message):
            raise ValueError(f"Error mapping message to chatroom: Object {raw_message_object} of type {type(raw_message_object)} is not a valid Discord Message")
        
        def get_known_chatroom_from_id(id: int, *, create_if_absent: bool = True) -> Chatroom | None:
            for room in self.known_chatrooms:
                if room.room_id == id:
                    return room
            if create_if_absent:
                MAXIMUM_CHATROOM_MEMORY = 64 # TODO: make configurable
                created_room = Chatroom(
                    SynchronizedMessageHistory(max_length=MAXIMUM_CHATROOM_MEMORY), 
                    id
                )
                self.known_chatrooms.append(created_room)
                return created_room
            else:
                return None

        discord_message: discord.Message = raw_message_object
        # TODO: this is a simplistic implementation with two types of chatrooms: 1. A server 2. DMs
        if discord_message.guild is None:
            return get_known_chatroom_from_id(discord_message.channel.id, create_if_absent=create_if_absent)
        else:
            return get_known_chatroom_from_id(discord_message.guild.id, create_if_absent=create_if_absent)

class DiscordBridge(commands.Cog):
    def __init__(self, bot: commands.Bot, bus: AsyncEventBus[MessageSnapshotEvent], known_chatrooms: list[Chatroom]):
        self.bot = bot
        self.bus = bus
        self._chatroom_mapper = DiscordMessageToChatroomMapper(known_chatrooms)

    @commands.Cog.listener()
    async def on_message(self, message):
        # Don't fire events for bots
        if message.author.bot:
            return

        snapshot = await MessageSnapshot.of_discord_message(message)
        chatroom = self._chatroom_mapper.find_chatroom(message)

        if chatroom is None:
            raise RuntimeError("Cannot find chatroom to assign to message: ", message)
        
        await self.bus.publish(
            MessageSnapshotEvent(
                backend="discord", snapshot=snapshot, raw_msg_object=message, chatroom=chatroom
            )
        )