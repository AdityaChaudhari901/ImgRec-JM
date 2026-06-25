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


@pytest.mark.asyncio
async def test_demo_targets_the_single_dispute_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")

    assert r.status_code == 200
    html = r.text
    # The demo console drives the one /dispute endpoint and nothing else.
    assert "/api/v1/imgrecog/dispute" in html
    assert "/api/v1/imgrecog/evaluate-links" not in html
    assert "/api/v1/imgrecog/scan" not in html
    assert "/api/v1/imgrecog/verify-claim" not in html
    # Dispute-specific rendering wired up.
    assert "renderDispute" in html
    assert "categoryFromQuery" in html
    # Core console structure preserved.
    assert 'id="queryText"' in html
    assert 'id="userImageUrl"' in html
    assert 'id="scanBtn"' in html
    assert 'id="rawPanel"' in html
    assert 'id="copyJson"' in html
    assert 'id="detailsPanel"' in html
    assert "readResponsePayload" in html
    # No file-upload / drag-drop, no leftover identity fields.
    assert 'type="file"' not in html
    assert 'id="orderId"' not in html
    assert 'id="userId"' not in html
