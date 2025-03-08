import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import tqdm
from uuid import uuid4

from constants import DRB


# 3 data classes, Chapter, Verse, Writing
@dataclass
class WritingNode:
    uuid: str = field(default_factory=str(uuid4()))
    author: str
    title: str
    testament: Optional[str] = field(default_factory=list)
    labels: List[str] = ["SCRIPTURE", "WRITING"]


@dataclass
class ChapterNode:
    uuid: str = field(default_factory=str(uuid4()))
    writing_title: str
    chapter: int
    writing_uuid: str


@dataclass
class VerseNode:
    uuid: str
    chapter: int
    verse: int
    drb: str
    embedding: List[float] = None
    chapter_uuid: str


class _Base:
    """
    A base class for processing raw text files and generating interim and cleaned file paths.

    Args:
        raw_text (Path): The path to the raw text file.

    Attributes:
        _raw_text (Path): The path to the raw text file.
        _title (str): The extracted title from the raw text file.
        _interim_text (str): The path to the interim text file.
        _cleaned_writing_file (str): The path to the cleaned writing nodes JSON file.
        _cleaned_chapter_file (str): The path to the cleaned chapter nodes JSON file.
        _cleaned_verse_file (str): The path to the cleaned verse nodes JSON file.
    """

    def __init__(self, raw_text: Path):
        self._raw_text_path = raw_text
        self._title = self._extract_title()
        self._cleaned_text_path = (
            Path("data") / "cleaned" / date.today().strftime("%Y-%m-%d") / f"{self._title}.txt"
        )

    def _extract_title(self) -> str:
        return os.path.basename(self._raw_text_path).split(".", 1)[0]


class LatinVulgate(_Base):

    def __init__(self, raw_text: Path):
        super().__init__(raw_text)
        self._work_nodes = []
        self._chapter_nodes = []
        self._verse_nodes = []
        self.WORKS = DRB["WORKS"]
        self.WRITING_NODE_IDS = DRB["WRITING_NODE_IDS"]
        # self._cosine_threshold = 0.65
        # self.k = 10
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def _reformat_text(self) -> None:
        """
        Reformat raw text into a list of verse lines without loading all lines into memory.
        """
        verse_pattern = r"^\d+:\d+\."
        verses = []

        with open(self._raw_text_path, "r", encoding="utf-8") as file:
            capture = False
            verse_text = ""

            for line in tqdm.tqdm(file, desc="Processing raw text"):
                if re.match(verse_pattern, line):
                    if verse_text:
                        verses.append(verse_text.strip())
                    verse_text = line.strip()
                    capture = True
                elif capture:
                    if line.strip() == "":
                        capture = False
                    else:
                        verse_text += " " + line.strip()

            if verse_text:
                verses.append(verse_text.strip())

        with open(self._cleaned_text_path, "w", encoding="utf-8") as cleaned_file:
            for verse in verses:
                cleaned_file.write(verse + "\n")

        return

    def _parse_writings(self) -> List[WritingNode]:
        """
        Generate Writing (Book) nodes.

        Iterates over the WORKS dictionary to create a list of writing nodes,
        each containing the writing ID, author, title, and testament.

        WORKS dictionary struture:
            {
                "Genesis": "Moses",
                "Exodus": "Moses",
                "Leviticus": "Moses",
                ...
            }

        Returns:
            List[Dict]: A list of dictionaries representing writing nodes.
        """
        writing_nodes = []
        for book, author in self.WORKS.items():

            testament = (
                "new"
                if author in ["Matthew", "Mark", "Luke", "John", "Paul", "Peter", "Jude", "James"]
                else "old"
            )

            writing_nodes.append(WritingNode(author=author, title=book, testament=testament))
        return writing_nodes

    def _parse_chapters_and_verses(self, verses: List[str]) -> tuple[List[Dict], List[Dict]]:
        """Parse verses into Chapter and Verse nodes, detecting book boundaries.

        Args:
            verses (List[str]): A list of verses in the format "chapter:verse. text".

        Returns:
            tuple[List[Dict], List[Dict]]: A tuple containing two lists of dictionaries:
                - The first list contains chapter nodes with keys:
                    - "chapter_id" (str): The unique identifier for the chapter.
                    - "chapter" (int): The chapter number.
                    - "writing" (str): The book to which the chapter belongs.
                - The second list contains verse nodes with keys:
                    - "verse_id" (str): The unique identifier for the verse.
                    - "verse" (int): The verse number.
                    - "drb" (str): The text of the verse.
                    - "writing" (str): The book to which the verse belongs.
        """
        chapter_nodes = []
        verse_nodes = []

        book_list = list(self.WORKS.keys())
        current_book_idx = 0
        current_book = book_list[current_book_idx]
        current_chapter = None
        last_chapter_num = 0

        for verse in tqdm.tqdm(verses, desc="Creating ch, vs nodes"):
            match = re.match(r"(\d+):(\d+)\.\s+(.+)", verse)
            if not match:
                continue

            chapter_num = int(match.group(1))
            verse_num, text = (
                int(match.group(2)),
                match.group(3),
            )

            if chapter_num == 1 and last_chapter_num > 1 and current_book_idx < len(book_list) - 1:
                current_book_idx += 1
                current_book = book_list[current_book_idx]
                current_chapter = None

            last_chapter_num = chapter_num

            if current_chapter != chapter_num:
                chapter_id = f"{self.WRITING_NODE_IDS[current_book]}-{chapter_num}"
                chapter_nodes.append(
                    {
                        "chapter_id": chapter_id,
                        "chapter": chapter_num,
                        "writing": current_book,
                    }
                )
                current_chapter = chapter_num

            # verse node
            verse_id = f"{self.WRITING_NODE_IDS[current_book]}-{chapter_num}-{verse_num}"
            verse_nodes.append(
                {
                    "verse_id": verse_id,
                    "verse": verse_num,
                    "drb": text,
                    "writing": current_book,
                }
            )

        return chapter_nodes, verse_nodes

    def _generate_embeddings(self) -> None:
        """
        Generates embeddings for verse nodes and adds them to each node.

        This method processes the verse nodes stored in `self._verse_nodes` by
        generating embeddings for each verse text using the model specified in
        `self.model`. The embeddings are then added to the corresponding nodes
        in the `self._verse_nodes` list.

        Returns:
            None
        """
        batch_size = 256

        verse_texts = [node["drb"] for node in self._verse_nodes]
        embeddings = self.model.encode(verse_texts, batch_size=batch_size, show_progress_bar=True)

        for node, embedding in zip(self._verse_nodes, embeddings):
            node["embedding"] = embedding.tolist()

    def _generate_relationship_nodes(self) -> List[Dict]:
        """
        Generates relationship nodes based on cosine similarity of embeddings.

        This method computes the cosine similarity between embeddings of verse nodes
        and identifies pairs of nodes that have a similarity above a specified threshold.
        The pairs are returned as a list of tuples containing the verse IDs and their
        similarity score.

        Returns:
            List[Dict]: A list of dictionaries where each dictionary represents a pair
            of verse nodes with their similarity score. Each dictionary contains:
            - verse_id_1 (str): The verse ID of the first node.
            - verse_id_2 (str): The verse ID of the second node.
            - similarity (float): The cosine similarity score between the two nodes.
        """

        embeddings = np.array([node["embedding"] for node in self._verse_nodes])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        nbrs = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        similarities = 1 - distances

        similar_pairs = []
        for i in range(len(embeddings)):
            for j in range(1, self.k):
                if similarities[i, j] > self._cosine_threshold:
                    similar_pairs.append(
                        (
                            self._verse_nodes[i]["verse_id"],
                            self._verse_nodes[indices[i, j]]["verse_id"],
                            np.round(float(similarities[i, j]), 5),
                        )
                    )
        return similar_pairs

    def parse_text(self) -> None:
        """Main parsing method."""
        print("Reformatting text...")
        verses = self._reformat_text()
        self._work_nodes = self._parse_writings()
        self._chapter_nodes, self._verse_nodes = self._parse_chapters_and_verses(verses)

        print("Generating embeddings...")
        self._generate_embeddings()

        print("Calculating cosine similarity between nodes...")
        self._rel_pairs = self._generate_relationship_nodes()

        # Dbg output
        print(f"Works: {len(self._work_nodes)}")
        print(f"Chapters: {len(self._chapter_nodes)}")
        print(f"Verses: {len(self._verse_nodes)}")
        print(f"Similarities: {len(self._rel_pairs)}")

        writing_out = open(self._cleaned_writing_file, "w")
        json.dump(self._work_nodes, writing_out, indent=6)
        writing_out.close()

        chapter_out = open(self._cleaned_chapter_file, "w")
        json.dump(self._chapter_nodes, chapter_out, indent=6)
        chapter_out.close()

        verse_out = open(self._cleaned_verse_file, "w")
        json.dump(self._verse_nodes, verse_out, indent=6)
        verse_out.close()

    @property
    def work_nodes(self) -> List[Dict]:
        return self._work_nodes

    @property
    def chapter_nodes(self) -> List[Dict]:
        return self._chapter_nodes

    @property
    def verse_nodes(self) -> List[Dict]:
        return self._verse_nodes

    @property
    def rel_pairs(self) -> List[Dict]:
        return self._rel_pairs


class Neo4jLoader:
    def __init__(
        self,
        rel_pairs: List[Dict],
        verse_nodes: List[Dict],
        chapter_nodes: List[Dict],
        writing_nodes: List[Dict],
    ):
        self.__neo4j_uri = os.environ.get("NEO4J_URI")
        self.__neo4j_username = os.environ.get("NEO4J_USERNAME")
        self.__neo4j_password = os.environ.get("NEO4J_PASSWORD")
        self.__aura_instanceid = os.environ.get("AURA_INSTANCEID")
        self.__aura_instancename = os.environ.get("AURA_INSTANCENAME")


if __name__ == "__main__":
    text_parser = LatinVulgate(Path("data/raw/douay-rheims-bible.txt"))
    text_parser.parse_text()
