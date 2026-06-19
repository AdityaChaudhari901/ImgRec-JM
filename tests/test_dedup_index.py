"""Phase 2 — perceptual-hash dedup index + classification.

Uses the in-memory index (the Redis path is byte-for-byte the same logic, exercised
in prod). Verifies: near-duplicate recall within the Hamming threshold, classification
of cross-claim (fraud) vs same-claim (benign resubmission), and exact Hamming.
"""

from app.services.dedup_service import (
    DedupResult,
    hamming_hex,
)
from app.services.dedup_index import InMemoryDedupIndex


def test_hamming_hex_counts_differing_bits():
    assert hamming_hex("0000000000000000", "0000000000000000") == 0
    assert hamming_hex("0000000000000000", "0000000000000001") == 1
    assert hamming_hex("0000000000000000", "000000000000000f") == 4


async def test_exact_duplicate_is_found():
    idx = InMemoryDedupIndex()
    await idx.add("abcdef0123456789", "JM-1", "u_1", "vfy_1")
    matches = await idx.query("abcdef0123456789", threshold=10)
    assert len(matches) == 1
    assert matches[0].order_id == "JM-1"
    assert matches[0].request_id == "vfy_1"
    assert matches[0].hamming == 0


async def test_near_duplicate_within_threshold_is_found():
    idx = InMemoryDedupIndex()
    await idx.add("abcdef0123456789", "JM-1", "u_1", "vfy_1")
    # differs in the last nibble (9 -> 8 = 1 bit)
    matches = await idx.query("abcdef0123456788", threshold=10)
    assert len(matches) == 1
    assert matches[0].hamming == 1


async def test_distant_image_beyond_threshold_is_not_found():
    idx = InMemoryDedupIndex()
    await idx.add("0000000000000000", "JM-1", "u_1", "vfy_1")
    # all 64 bits set -> Hamming 64, far beyond threshold
    matches = await idx.query("ffffffffffffffff", threshold=10)
    assert matches == []


async def test_query_returns_empty_when_index_empty():
    idx = InMemoryDedupIndex()
    assert await idx.query("abcdef0123456789", threshold=10) == []


async def test_classification_splits_cross_claim_from_same_claim():
    idx = InMemoryDedupIndex()
    # same image previously submitted by a DIFFERENT user/order -> cross-claim (fraud)
    await idx.add("abcdef0123456789", "JM-OTHER", "u_other", "vfy_other")
    # ...and by the SAME user+order -> benign resubmission
    await idx.add("abcdef0123456789", "JM-1", "u_1", "vfy_self")

    raw = await idx.query("abcdef0123456789", threshold=10)
    result = DedupResult.classify(raw, order_id="JM-1", user_id="u_1")

    assert result.is_cross_claim_duplicate is True
    assert {m.order_id for m in result.cross_claim_matches} == {"JM-OTHER"}
    assert {m.order_id for m in result.same_claim_matches} == {"JM-1"}
