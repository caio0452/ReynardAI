import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

from reynard_ai.ai_apis.api_types import Prompt
from reynard_ai.bot_data.ai_bot import Profile
from reynard_ai.ai_apis.client import LLMClient

class Moderator(ABC):
    @dataclass
    class Result:
        flagged: bool
        category_scores: dict[str, float]
       
    @abstractmethod
    async def moderate(self, input: Prompt) -> Result:
        raise NotImplementedError("moderate")
    
class LLMModerator(Moderator):
    def __init__(self, profile: Profile):
        NAME = "MODERATOR"
        if NAME not in profile.providers:
            raise RuntimeError(f"Could not initialize moderation, missing provider '{NAME}'")
        if NAME not in profile.prompts:
            raise RuntimeError(f"Could not initialize moderation, missing prompt '{NAME}'")
        if NAME not in profile.request_params:
            raise RuntimeError(f"Could not initialize moderation, missing parameters for '{NAME}'")
        self.moderator_prompt = profile.prompts[NAME]
        self.moderator_provider = profile.providers[NAME]
        self.moderator_parameters = profile.request_params[NAME]
        self.client: LLMClient = LLMClient.from_provider(self.moderator_provider)

    async def moderate(self, input: Prompt) -> Moderator.Result:
        result = await self.client.send_request(prompt=input, params=self.moderator_parameters)
        msg = result.message.content
        if msg is None:
            raise RuntimeError("Failed to moderate: moderator returned no response")
        
        # LLama-Guard format
        if msg.startswith("safe"):
            return LLMModerator.Result(False, {})
        elif msg.startswith("unsafe"):
            all_lines = msg.split("\n")
            flags = {}
            if len(all_lines) > 1:
                second_line_categories = all_lines[1].split(",")
                for flagged_category in second_line_categories:
                    flags[flagged_category] = 1
            return LLMModerator.Result(True, flags)

        # Custom format
        try:
            json_result = json.loads(msg)
            flagged = json_result["flagged"]
            category_scores = {
                key: float(value) 
                for key, value 
                in json_result.items() 
                if key != "flagged"
            }
            return LLMModerator.Result(flagged, category_scores)
        except Exception as ex:
            raise RuntimeError(
                'Cannot parse moderator result. Must either be a Llama-Guard output or a JSON with format {"flagged": boolean, "harmcategory1": float, "harmcategory2": float...}, instead got: ' + msg
            ) from ex
