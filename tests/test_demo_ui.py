import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_root_serves_demo_html():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
