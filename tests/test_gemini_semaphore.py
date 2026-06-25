from app.services import gemini_service


def test_semaphore_sized_from_settings():
    sem = gemini_service._get_gemini_semaphore()
    assert sem._value == gemini_service.settings.gemini_max_concurrency
