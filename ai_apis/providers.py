from pydantic import BaseModel, Field, model_validator
from util.environment_vars import parse_api_key_in_config
from typing import List

class ProviderData(BaseModel):
    provider_name: str
    api_base: str = Field(default='https://api.openai.com/v1')
    api_key: str

    @model_validator(mode='before')
    @classmethod
    def check_and_load_api_key(cls, values):
        if 'api_key' in values:
            values['api_key'] = parse_api_key_in_config(values['api_key'])
        return values

class ProviderDataStore:
    def __init__(self, *, providers: List[ProviderData], optional_provider_names: List[str] = []):
        self.providers: dict[str, ProviderData] = {p.provider_name: p for p in providers}
        self.optional_provider_names = optional_provider_names

    def get_provider_by_name(self, name: str) -> ProviderData | None:
        provider = self.providers.get(name)
        if not provider and name not in self.optional_provider_names:
            raise ValueError(f"Missing required provider named {name}. Please ensure it is set in the profile JSON.")
        return provider