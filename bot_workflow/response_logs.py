from io import StringIO

import logging

class SimpleDebugLogger:
    def __init__(self, name: str):
        self.log_stream = StringIO()
        self.name = name
        self._setup_logging()
        
    def _setup_logging(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False 
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        if root_logger.handlers:
            self.handler.setFormatter(root_logger.handlers[0].formatter)
        else:
            formatter = logging.Formatter('%(message)s')
            self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
    
    def verbose(self, message: str, *, category: str | None = None):
        if category is None:
            self.logger.debug(message)
        else:
            self.logger.debug(f"[{category}] {message}")
        
    @property
    def text(self) -> str:
        return self.log_stream.getvalue()
    
    def __del__(self):
        self.logger.removeHandler(self.handler)

class ResponseLogsManager:
    _instance: "ResponseLogsManager | None" = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls.log_capacity = 10
            cls._last_message_id_logs: dict[int, str] = {}
        return cls._instance
    
    def store_log(self, message_id: int, log: str):
        logging.info(f"Saved log for message id {message_id}")
        self._last_message_id_logs[message_id] = log
        if len(self._last_message_id_logs) > self.log_capacity:
            oldest_key = next(iter(self._last_message_id_logs))
            del self._last_message_id_logs[oldest_key]

    def get_log_by_id(self, message_id: int) -> str | None:
        return self._last_message_id_logs.get(message_id, None)
