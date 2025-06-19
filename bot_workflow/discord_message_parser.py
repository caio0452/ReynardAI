from util.rate_limits import RateLimiter, RateLimit
from dataclasses import dataclass
from discord.ext import commands
from enum import Enum, auto
import discord

class DenialReason(Enum):
    RATE_LIMITED = auto()
    TOO_LONG = auto()
    DID_NOT_PING = auto()

class SpecialFunctionFlags(Enum):
    VIEW_MESSAGE_LOGS = auto()
    REQUEST_VERBOSE_REPLY = auto()

@dataclass
class UserMessageContext:
    raw_content: str
    sanitized_content: str
    denied: bool
    denial_reason: DenialReason | None
    called_functions: list[SpecialFunctionFlags]

class DiscordMessageParser:
    def __init__(self, bot: commands.Bot):
        self.MAX_CHARACTERS = 1024
        self.rate_limiter = RateLimiter()
        self.bot = bot
        self.message = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5*60),
            RateLimit(n_messages=100, seconds=2*3600),
            RateLimit(n_messages=250, seconds=8*3600)
        )

    def parse_message(self, message: discord.Message) -> UserMessageContext:
        raw_content = message.content
        sanitized_content = raw_content
        denied = False
        denial_reason = None
        called_functions: list[SpecialFunctionFlags] = []

        self.rate_limiter.register_request(message.author.id)

        if self.bot.user not in message.mentions:
            denied = True
            denial_reason = DenialReason.DID_NOT_PING

        if self.rate_limiter.is_rate_limited(message.author.id):
            denied = True
            denial_reason = DenialReason.RATE_LIMITED

        if not denied and len(raw_content) > self.MAX_CHARACTERS:
             denied = True
             denial_reason = DenialReason.TOO_LONG

        if not denied:
            if "--l" in raw_content: # TODO: inconsistent with --v
                called_functions.append(SpecialFunctionFlags.VIEW_MESSAGE_LOGS)
                sanitized_content = sanitized_content.replace("--l", "")

            if raw_content.endswith("--v"):
                called_functions.append(SpecialFunctionFlags.REQUEST_VERBOSE_REPLY)
                sanitized_content = sanitized_content.removesuffix("--v")

        return UserMessageContext(
            sanitized_content=sanitized_content,
            raw_content=raw_content,
            denied=denied,
            denial_reason=denial_reason,
            called_functions=called_functions
        )