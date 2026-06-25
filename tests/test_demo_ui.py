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
async def test_demo_focuses_on_product_status_without_visible_identity_fields():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")

    assert r.status_code == 200
    html = r.text
    assert 'id="orderId"' not in html
    assert 'id="userId"' not in html
    assert 'id="vOrderId"' not in html
    assert 'id="vUserId"' not in html
    assert "JM-29384" not in html
    assert "u_kaily_123" not in html
    assert "Product status" in html
    assert "Expired" in html
    assert "Damaged" in html
    assert "Valid" in html
    assert "Customer query" in html
    assert 'id="queryText"' in html
    assert 'id="userImageUrl"' in html
    assert 'id="productImageUrl"' in html
    assert 'id="scanBtn"' in html
    assert "Scan product" in html
    assert "Final decision" in html
    assert "Decision reasons" in html
    assert "Raw JSON response" in html
    assert "readResponsePayload" in html
    assert "summarizeHttpText" in html
    assert "Server returned malformed JSON response" in html
    assert 'id="rawPanel"' in html
    assert 'id="copyJson"' in html
    assert 'id="expandJson"' in html
    assert 'id="detailsPanel"' in html
    assert "quick-result" in html
    assert "decision" in html
    assert "product_status" in html
    assert "/api/v1/imgrecog/evaluate-links" in html
    assert "/api/v1/imgrecog/scan" not in html
    assert "Evaluate links" not in html
    assert "Query presets" not in html
    assert "data-query" not in html
    assert 'id="evaluateBtn"' not in html
    assert 'id="drop"' not in html
    assert 'type="file"' not in html
