from pydantic import BaseModel, Field
from typing import Optional

class ExtractedEntities(BaseModel):
    # Тип A — конкретные значения
    inn: Optional[str] = Field(None, description="ИНН организации")
    date: Optional[str] = Field(None, description="Дата документа")

    # Тип B — логические признаки
    contract_presence: bool
    has_stamp: bool
    contains_nso_terms: bool
    has_penalty_clause: bool
    is_offer: bool
    mentions_termination: bool
    mentions_guarantee: bool
    mentions_insurance: bool