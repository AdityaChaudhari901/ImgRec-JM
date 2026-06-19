import logging

import structlog

from app.config.settings import settings

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)

    # Quiet noisy third-party loggers (google-genai AFC info, httpx request
    # lines, and ADC's "No project ID" warning — we pass project explicitly).
    for noisy in ("google_genai", "google_genai.models", "httpx", "google.auth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.ERROR)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str):
    """Return a structlog JSON logger bound to `name`."""
    _configure()
    return structlog.get_logger(name)
