from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared limiter instance — imported by the router (to decorate endpoints) and
# by main.py (to register state + the 429 exception handler).
limiter = Limiter(key_func=get_remote_address)
