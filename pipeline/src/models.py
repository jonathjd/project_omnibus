from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4


@dataclass
class WritingNode:
    author: str
    title: str
    testament: Optional[str]
    uuid: str = field(default_factory=uuid4)
    labels: List[str] = field(default_factory=lambda: ["SCRIPTURE", "WRITING"])


@dataclass
class ChapterNode:
    writing_title: str
    chapter: int
    uuid: str = field(default_factory=uuid4)
    labels: List[str] = field(default_factory=lambda: ["CHAPTER"])


@dataclass
class VerseNode:
    writing_title: str
    chapter: int
    verse: int
    drb: str
    uuid: str = field(default_factory=uuid4)
    embedding: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=lambda: ["VERSE"])


@dataclass(unsafe_hash=True)
class EntityNode:
    name: str
    label: str
    uuid: str = field(default_factory=uuid4)


@dataclass
class VerseSimilarity:
    verse_id_1: str
    verse_id_2: str
    similarity: float
