from app.models.dispute_request import Ticket
from app.services.category_classifier import classify_category


def test_provided_wins():
    assert classify_category("mrp_abuse", Ticket(description="anything")) == ("mrp_abuse", "provided")


def test_description_keyword():
    assert classify_category(None, Ticket(description="I got the wrong item")) == ("wrong_product", "description")


def test_notes_when_description_blank():
    t = Ticket(description="", notes="bottle was leaking everywhere")
    assert classify_category(None, t) == ("damaged", "notes")


def test_disposition_map():
    t = Ticket(description="", notes="", disposition_code="PRICE_DISPUTE")
    assert classify_category(None, t) == ("mrp_abuse", "disposition")


def test_insufficient_data():
    assert classify_category(None, Ticket()) == (None, "none")
