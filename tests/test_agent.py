import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

@pytest.fixture
def test_text():
    return (
        "Настоящим договором от 12.07.2023 компания ООО \"Пример\" с ИНН 7701234567 подтверждает обязательства "
        "перед заказчиком. Стороны договорились о гарантийном сроке в 12 месяцев. "
        "В случае нарушения условий договора предусмотрены штрафные санкции. Документ заверен печатью."
    )

def test_extracted_entities_from_text(test_text):
    response = client.post(
        "/upload-text",
        data={"text": test_text}
    )

    assert response.status_code == 200
    result = response.json().get("result", {})

    assert result.get("inn") == "7701234567"
    assert result.get("date") == "12.07.2023"
    assert result.get("has_stamp") is True
    assert result.get("mentions_guarantee") is True
    assert result.get("contains_penalty") is True
