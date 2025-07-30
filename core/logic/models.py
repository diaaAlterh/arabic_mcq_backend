from pydantic import BaseModel, Field
from typing import List, Optional

class BaseQuestion(BaseModel):
    question: str
    answers: List[str]
    correct_answer: str

class GeneratedQuestion(BaseQuestion):
    pass

class GeneratedMCQOutput(BaseModel):
    questions: List[GeneratedQuestion]

class ValidatedQuestion(GeneratedQuestion):
    is_valid: bool = Field(description="True if the question is grammatically correct, logically sound, and relevant; false otherwise.")
    feedback: Optional[str] = Field(default=None, description="Short feedback in Arabic if the question is invalid.")

class ValidatedMCQOutput(BaseModel):
    questions: List[ValidatedQuestion]

class FinalQuestion(ValidatedQuestion):
    difficulty: str = Field(description="Difficulty level of the question: 'easy', 'normal', or 'hard' (in Arabic).")

class FinalMCQOutput(BaseModel):
    questions: List[FinalQuestion]