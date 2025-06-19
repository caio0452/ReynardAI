import os
import logging
from dotenv import load_dotenv

load_dotenv()

def get_environment_var(var_name: str, *, required: bool) -> str:
    value = os.getenv(var_name)
    if required and (value is None or value == ""):
        logging.info(f"Missing '{var_name}' environment variable")
        exit(1)
    return "" if value is None else value

def parse_api_key_in_config(api_key: str) -> str:
    if api_key.startswith('[') and api_key.endswith(']'):
        env_var_name = api_key[1:-1]
        loaded_api_key = get_environment_var(env_var_name, required=True)
        if not loaded_api_key:
            raise ValueError(f"Environment variable {env_var_name} not set for API key.")
        return loaded_api_key
    return api_key
