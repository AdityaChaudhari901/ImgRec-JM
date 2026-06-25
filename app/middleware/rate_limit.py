from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config.settings import settings

# Shared limiter instance — imported by the router (to decorate endpoints) and
# by main.py (to register state + the 429 exception handler).
#
# With REDIS_URL set, the limit is enforced GLOBALLY across replicas (SlowAPI
# stores counters in Redis); otherwise it is in-memory per instance (dev / single
# replica). Backing it with Redis is required for a correct global limit at scale.
_storage_uri = settings.redis_url or "memory://"
limiter = Limiter(key_func=get_remote_address, storage_uri=_storage_uri)
