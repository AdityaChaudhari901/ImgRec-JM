from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import quote, urljoin, urlparse

import httpx
from PIL import Image, ImageOps

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MIME_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),
)


@dataclass(frozen=True)
class FetchedImage:
    source_url: str
    fetched_url: str
    mime_type: str
    data: bytes
    model_mime_type: str | None = None
    model_data: bytes | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def model_size_bytes(self) -> int:
        return len(self.model_bytes)

    @property
    def model_bytes(self) -> bytes:
        return self.model_data or self.data

    @property
    def model_content_type(self) -> str:
        return self.model_mime_type or self.mime_type

    @property
    def data_uri(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


class ImageUrlError(ValueError):
    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def pixelbin_url_for(source_url: str) -> str:
    """Return the configured PixelBin URL for a safe source URL.

    PixelBin URL layouts vary by organization/source setup, so the production
    integration is intentionally template-driven instead of hardcoding one CDN
    path. If disabled, the original URL is returned unchanged.
    """
    if not settings.pixelbin_enabled:
        return source_url

    template = settings.pixelbin_url_template.strip()
    if not template:
        raise ImageUrlError("PixelBin is enabled but PIXELBIN_URL_TEMPLATE is empty", 500)

    parsed = urlparse(source_url)
    path = parsed.path.lstrip("/")
    try:
        return template.format(
            url=source_url,
            url_encoded=quote(source_url, safe=""),
            host=parsed.netloc,
            path=path,
            path_encoded=quote(path, safe=""),
        )
    except KeyError as exc:
        raise ImageUrlError(
            f"PIXELBIN_URL_TEMPLATE uses unsupported placeholder: {exc}",
            500,
        ) from exc


async def download_image_url(source_url: str, role: str = "image") -> FetchedImage:
    """Validate and fetch an image URL with SSRF, size, type, and redirect guards."""
    safe_source_url = await validate_public_http_url(source_url, role=role)
    fetch_url = pixelbin_url_for(safe_source_url)
    if fetch_url != safe_source_url:
        fetch_url = await validate_public_http_url(fetch_url, role=f"{role} PixelBin URL")

    try:
        return await _fetch_once(safe_source_url, fetch_url, role)
    except ImageUrlError as exc:
        if (
            fetch_url != safe_source_url
            and settings.pixelbin_allow_direct_fallback
            and exc.status_code in {408, 409, 415, 422, 502, 503, 504}
        ):
            logger.warning(
                "pixelbin_fetch_failed_using_direct_source",
                role=role,
                error=exc.detail,
            )
            return await _fetch_once(safe_source_url, safe_source_url, role)
        raise


async def validate_public_http_url(url: str, role: str = "image") -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ImageUrlError(f"{role} URL must use http or https")
    if not parsed.hostname:
        raise ImageUrlError(f"{role} URL must include a hostname")
    if parsed.username or parsed.password:
        raise ImageUrlError(f"{role} URL must not contain credentials")

    host = parsed.hostname.strip().lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise ImageUrlError(f"{role} URL host is not allowed")

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    await _assert_public_host(host, port, role)
    return url.strip()


async def _assert_public_host(host: str, port: int, role: str) -> None:
    direct_ip = _parse_ip(host)
    if direct_ip is not None:
        _assert_public_ip(direct_ip, role)
        return

    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ImageUrlError(f"{role} URL host could not be resolved") from exc

    ips = {_parse_ip(info[4][0]) for info in infos}
    public_ips = [ip for ip in ips if ip is not None]
    if not public_ips:
        raise ImageUrlError(f"{role} URL did not resolve to an IP address")
    for ip in public_ips:
        _assert_public_ip(ip, role)


def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return None


def _assert_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, role: str) -> None:
    if not ip.is_global:
        raise ImageUrlError(f"{role} URL resolves to a non-public network")


async def _fetch_once(source_url: str, fetch_url: str, role: str) -> FetchedImage:
    timeout = httpx.Timeout(settings.url_fetch_timeout_seconds)
    headers = {
        "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
        "User-Agent": settings.url_fetch_user_agent,
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        current_url = fetch_url
        redirects = 0
        processing_attempt = 0
        while True:
            try:
                response_ctx = client.stream("GET", current_url, headers=headers)
                response = await response_ctx.__aenter__()
            except httpx.TimeoutException as exc:
                raise ImageUrlError(f"{role} URL fetch timed out", 504) from exc
            except httpx.HTTPError as exc:
                raise ImageUrlError(f"{role} URL could not be fetched", 502) from exc

            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    redirects += 1
                    if redirects > settings.url_fetch_max_redirects:
                        raise ImageUrlError(f"{role} URL redirected too many times")
                    location = response.headers.get("location")
                    if not location:
                        raise ImageUrlError(f"{role} URL redirect is missing Location header")
                    current_url = urljoin(current_url, location)
                    current_url = await validate_public_http_url(current_url, role=role)
                    continue

                if response.status_code == 202 and processing_attempt < settings.url_fetch_processing_retries:
                    processing_attempt += 1
                    await asyncio.sleep(settings.url_fetch_processing_retry_delay_seconds)
                    continue

                if response.status_code != 200:
                    raise ImageUrlError(
                        f"{role} URL returned HTTP {response.status_code}",
                        422 if response.status_code < 500 else 502,
                    )

                data = await _read_bounded_response(response, role)
                mime_type = _detect_mime_type(data, response.headers.get("content-type", ""))
                _verify_image(data, mime_type, role)
                model_mime_type, model_data = _optimize_for_model(data, mime_type, role)
                return FetchedImage(
                    source_url=source_url,
                    fetched_url=current_url,
                    mime_type=mime_type,
                    data=data,
                    model_mime_type=model_mime_type,
                    model_data=model_data,
                )
            finally:
                await response_ctx.__aexit__(None, None, None)


async def _read_bounded_response(response: httpx.Response, role: str) -> bytes:
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > max_bytes:
            raise ImageUrlError(
                f"{role} image exceeds {settings.max_image_size_mb}MB limit",
                413,
            )

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ImageUrlError(
                f"{role} image exceeds {settings.max_image_size_mb}MB limit",
                413,
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise ImageUrlError(
            f"{role} image exceeds {settings.max_image_size_mb}MB limit",
            413,
        )
    if not data:
        raise ImageUrlError(f"{role} image response was empty")
    return data


def _detect_mime_type(data: bytes, content_type: str) -> str:
    header_mime = content_type.split(";", 1)[0].strip().lower()
    detected = _mime_from_magic(data)
    if detected:
        return detected
    if header_mime in ALLOWED_IMAGE_MIME_TYPES:
        return header_mime
    raise ImageUrlError("URL did not return a supported image type", 415)


def _mime_from_magic(data: bytes) -> str | None:
    for signature, mime_type in _MIME_SIGNATURES:
        if data.startswith(signature):
            if mime_type == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime_type
    return None


def _verify_image(data: bytes, mime_type: str, role: str) -> None:
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ImageUrlError(f"{role} image type is unsupported", 415)
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001
        raise ImageUrlError(f"{role} image could not be decoded") from exc


def _optimize_for_model(
    data: bytes,
    mime_type: str,
    role: str,
) -> tuple[str | None, bytes | None]:
    """Return a smaller model-facing JPEG while preserving original bytes.

    The original bytes remain available for metadata, AI, and web-provenance
    checks. Gemini receives the optimized representation to reduce latency and
    timeout risk for large CDN originals.
    """
    max_edge = max(0, int(settings.link_eval_model_max_edge_px or 0))
    quality = max(40, min(95, int(settings.link_eval_model_image_quality or 85)))
    if max_edge <= 0:
        return None, None

    try:
        with Image.open(BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            original_size = image.size
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            resized = image.size != original_size

            if image.mode in {"RGBA", "LA"}:
                alpha = image.getchannel("A")
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image.convert("RGBA"), mask=alpha)
                image = background
            elif image.mode == "P" and "transparency" in image.info:
                rgba = image.convert("RGBA")
                alpha = rgba.getchannel("A")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=alpha)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            optimized = output.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model_image_optimization_failed", role=role, error=str(exc))
        return None, None

    if not resized and len(optimized) >= len(data):
        return None, None

    logger.info(
        "model_image_optimized",
        role=role,
        source_mime_type=mime_type,
        source_bytes=len(data),
        model_bytes=len(optimized),
        max_edge=max_edge,
    )
    return "image/jpeg", optimized
