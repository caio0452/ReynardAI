from time import time

class RateLimit:
    def __init__(self, *, n_messages: int, seconds: int):
        self.n_messages = n_messages
        self.seconds = seconds

class RateLimiter:
    def __init__(self, *limits: RateLimit):
        self.limits = limits
        self.user_logs: dict[int, list[float]] = {}

    def register_request(self, user_id: int) -> None:
        if user_id not in self.user_logs:
            self.user_logs[user_id] = []
        self.user_logs[user_id].append(time())
        self._cleanup(user_id)

    def is_rate_limited(self, user_id: int) -> bool:
        if user_id not in self.user_logs:
            return False

        for limit in self.limits:
            if self._is_limited(user_id, limit):
                return True

        return False

    def _is_limited(self, user_id: int, limit: RateLimit) -> bool:
        relevant_logs = [log for log in self.user_logs[user_id] if log > time() - limit.seconds]
        return len(relevant_logs) > limit.n_messages

    def _cleanup(self, user_id: int):
        for limit in self.limits:
            self.user_logs[user_id] = [log for log in self.user_logs[user_id] if log > time() - limit.seconds]