import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from neo4j import AuthError, GraphDatabase, ServiceUnavailable
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from constants import DRB
from logging_config import setup_logging

logger = setup_logging()


# ---------------------
# Data Classes
# ---------------------
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


# ---------------------
# LatinVulgate Class
# ---------------------
class LatinVulgate:
    """
    Handles text preprocessing for the Latin Vulgate, including:
     - Reading and cleaning raw text
     - Parsing into Writing, Chapter, Verse nodes
    """

    def __init__(self, raw_text: Path):
        # save raw text path and clean
        self._raw_text_path = raw_text
        self._title = raw_text.stem
        self._cleaned_text_path = (
            Path("data") / "cleaned" / f"{date.today().strftime('%Y-%m-%d')}-{self._title}.txt"
        )

        # Works constants
        self.WORKS = DRB["WORKS"]

        # verse line regex
        self._verse_pattern = r"(\d+):(\d+)\.\s+(.+)"

        # Generate node lists
        self._reformat_text()
        self._parse_writings()
        self._parse_chapters_and_verses()

    def _reformat_text(self) -> None:
        """
        Reformat raw text into one verse per line.
        """
        verses = []
        capture = False
        verse_text = ""

        self._cleaned_text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._raw_text_path, "r", encoding="utf-8") as file:
            for line in file:
                if re.match(self._verse_pattern, line):
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
            for vs in verses:
                cleaned_file.write(vs + "\n")

    def _parse_writings(self) -> List[WritingNode]:
        """Generate WritingNode objects based on the WORKS dictionary."""
        writing_nodes = []
        for book, author in self.WORKS.items():
            testament = (
                "new"
                if author in ["Matthew", "Mark", "Luke", "John", "Paul", "Peter", "Jude", "James"]
                else "old"
            )
            writing_nodes.append(WritingNode(author=author, title=book, testament=testament))
        self._work_nodes = writing_nodes

    def _parse_chapters_and_verses(self) -> None:
        """Parse lines in the cleaned text file into ChapterNode and VerseNode objects."""

        chapter_nodes = []
        verse_nodes = []

        book_list = list(self.WORKS.keys())
        current_book_idx = 0
        current_book = book_list[current_book_idx]
        current_chapter = None
        last_chapter_num = 0

        with open(self._cleaned_text_path, "r", encoding="utf-8") as file:
            for verse in file:
                match = re.match(self._verse_pattern, verse)
                if not match:
                    continue

                chapter_num = int(match.group(1))
                verse_num = int(match.group(2))
                text = match.group(3)

                # Move to next book if verse numbering resets to 1
                if (
                    chapter_num == 1
                    and last_chapter_num > 1
                    and current_book_idx < len(book_list) - 1
                ):
                    current_book_idx += 1
                    current_book = book_list[current_book_idx]
                    current_chapter = None

                last_chapter_num = chapter_num

                if current_chapter != chapter_num:
                    chapter_nodes.append(
                        ChapterNode(writing_title=current_book, chapter=chapter_num)
                    )
                    current_chapter = chapter_num

                verse_nodes.append(
                    VerseNode(
                        writing_title=current_book, chapter=chapter_num, verse=verse_num, drb=text
                    )
                )

        self._chapter_nodes = chapter_nodes
        self._verse_nodes = verse_nodes

    @property
    def work_nodes(self) -> List[WritingNode]:
        return self._work_nodes

    @property
    def chapter_nodes(self) -> List[ChapterNode]:
        return self._chapter_nodes

    @property
    def verse_nodes(self) -> List[VerseNode]:
        return self._verse_nodes

    @property
    def cleaned_text_path(self) -> Path:
        return self._cleaned_text_path

    @property
    def title(self) -> str:
        return self._title


# ---------------------
# NLP Class
# ---------------------
class NLP:
    """
    Handles natural language processing tasks:
     - Named entity recognition (NER)
     - Embeddings / nearest-neighbor searches
    """

    def __init__(self):
        """
        Initialize NLP with a spaCy model, custom patterns, and a sentence transformer.
        """
        self.nlp = spacy.load("en_core_web_trf")

        # constants
        self.EXCLUSIONS = DRB["EXCLUSIONS"]
        self.NAME_ALIASES = DRB["NAME_ALIASES"]

        # Custom patterns
        ruler = self.nlp.add_pipe("entity_ruler", after="ner")
        patterns = [
            {"label": "PERSON", "pattern": "Esau"},
            {"label": "PERSON", "pattern": "Ramesses"},
            {"label": "PERSON", "pattern": "Pharao"},
        ]
        ruler.add_patterns(patterns)

        # Model for embeddings
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def extract_entities(
        self,
        verse_nodes: List[VerseNode],
        chunk_size: int = 750,
    ) -> List[EntityNode]:
        """
        Extract named entities from a list of verses using spaCy NLP.

        This method processes verses in chunks to identify and extract named entities,
        specifically focusing on persons and places while filtering out unwanted entity types.

        Args:
            verse_nodes (List[VerseNode]): A list of VerseNode objects containing biblical verses.
            chunk_size (int, optional): The number of verses to process in each batch. Defaults to 500.

        Returns:
            List[EntityNode]: A combined list of unique EntityNode and PlaceNode objects extracted from the verses.

        Raises:
            Exception: If there's an error during NLP processing of the text.

        Examples:
            >>> parser = BibleParser()
            >>> verses = [
            ...     VerseNode(drb="And Adam called his wife's name Eve."),
            ...     VerseNode(drb="Cain spoke to Abel his brother.")
            ... ]
            >>> entities = parser.extract_entities(verses, chunk_size=2)
            >>> [entity.name for entity in entities]
            ['Adam', 'Eve', 'Cain', 'Abel']

        Notes:
            - Filters out entity types: DATE, CARDINAL, TIME, ORDINAL, QUANTITY
            - Uses NAME_ALIASES for name normalization
            - Excludes entities listed in EXCLUSIONS
            - Processes PERSON, NORP, LOC, and GPE entity types
        """
        verses = [node.drb for node in verse_nodes]
        entities = set()
        entity_list = []

        for i in range(0, len(verses), chunk_size):
            logger.info(f"Processing verses {i} through {i + chunk_size}...")
            text = " ".join(verses[i : i + chunk_size])
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    # Add only wanted entities
                    if ent.label_ in ["PERSON", "NORP", "LOC", "GPE"]:
                        text_lower = ent.text.lower()
                        if text_lower not in self.EXCLUSIONS:
                            normalized = self.NAME_ALIASES.get(text_lower, ent.text)
                            if (normalized, ent.label_) not in entities:
                                # EntityNode will add a UUID which will allow duplicate nodes to be
                                # inserted into set. Add only text and label for O(1) lookup then make the node.
                                entities.add((normalized, ent.label_))
                                entity_list.append(EntityNode(name=normalized, label=ent.label_))
            except Exception as e:
                logger.error(f"Error processing text for NER: {str(e)}")

        del entities

        return entity_list

    def compute_embeddings(self, verse_nodes: List[VerseNode]) -> List[VerseNode]:
        """Generates embeddings for verse nodes and stores them in each VerseNode instance.

        Args:
            verse_nodes: List of VerseNode objects containing biblical verses.

        Returns:
            List[VerseNode]: The same verse nodes with embeddings added to each node.

        Examples:
            >>> nlp = NLP()
            >>> verses = [
            ...     VerseNode(writing_title="Genesis", chapter=1, verse=1,
            ...              drb="In the beginning God created heaven and earth."),
            ...     VerseNode(writing_title="Genesis", chapter=1, verse=2,
            ...              drb="And the earth was void and empty.")
            ... ]
            >>> verses_with_embeddings = nlp.compute_embeddings(verses)
            >>> len(verses_with_embeddings[0].embedding)  # Embedding dimension
            384
        """
        verse_texts = [v.drb for v in verse_nodes]
        embeddings = self.model.encode(verse_texts, batch_size=256, show_progress_bar=True)
        for node, emb in zip(verse_nodes, embeddings):
            node.embedding = emb.tolist()

        return verse_nodes

    def generate_relationship_nodes(
        self, verse_nodes: List[VerseNode], n_neighbors: int = 10
    ) -> List[VerseSimilarity]:
        """Generates similarity-based relationship data between verse nodes.

        Uses cosine similarity and k-nearest neighbors to find similar verses based on
        their embeddings.

        Args:
            verse_nodes: A list of VerseNode objects containing verse embeddings.

        Returns:
            List[VerseSimilarity]: A list of VerseSimilarity objects containing:
                - verse_id_1: UUID of the first verse
                - verse_id_2: UUID of the similar verse
                - similarity: Cosine similarity score between the verses

        Examples:
            >>> nlp = NLP(k=3)
            >>> verses = [
            ...     VerseNode(drb="In the beginning God created heaven.",
            ...              embedding=[0.1, 0.2, 0.3]),
            ...     VerseNode(drb="God made the firmament.",
            ...              embedding=[0.15, 0.25, 0.35])
            ... ]
            >>> relationships = nlp.generate_relationship_nodes(verses)
            >>> len(relationships)
            2  # k-1 relationships per verse
            >>> relationships[0]
            VerseSimilarity(verse_id_1='uuid1', verse_id_2='uuid2', similarity=0.95)
        """
        embeddings = np.array([v.embedding for v in verse_nodes])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        similarities = 1 - distances

        similar_pairs = [
            VerseSimilarity(
                verse_id_1=verse_nodes[i].uuid,
                verse_id_2=verse_nodes[indices[i, j]].uuid,
                similarity=float(similarities[i, j]),
            )
            for i in range(len(embeddings))
            for j in range(1, n_neighbors)
        ]

        return similar_pairs

    def similarity_distribution(self, similarities: List[VerseSimilarity]) -> None:
        """Plot distribution of verse similarities with statistics."""

        sims = np.array([node.similarity for node in similarities])
        mean = np.mean(sims).round(2)
        median = np.median(sims).round(2)
        stdev = np.std(sims).round(2)

        # plot distribution
        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=sims, fill=True, color="skyblue")

        plt.axvline(x=mean, color="red", linestyle="--", alpha=0.8, label=f"Mean: {mean}")
        plt.axvline(x=median, color="green", linestyle="--", alpha=0.8, label=f"Median: {median}")
        plt.title("Distribution of Verse Similarities", fontsize=14, pad=20)
        plt.xlabel("Similarity Score", fontsize=12)
        plt.ylabel("Density", fontsize=12)

        stats_text = f"Mean: {mean}\nMedian: {median}\nStd Dev: {stdev}"
        plt.text(
            0.02,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig("plots/similarity_distribution.png", dpi=300, bbox_inches="tight")

        return


class Neo4jLoader:
    # initialize connection
    def __init__(
        self,
        URI: str,
        USERNAME: str,
        PASSWORD: str,
    ):
        self.__neo4j_uri = URI
        self.__neo4j_username = USERNAME
        self.__neo4j_password = PASSWORD

        try:
            self.driver = GraphDatabase.driver(
                self.__neo4j_uri, auth=(self.__neo4j_username, self.__neo4j_password)
            )
            self.driver.verify_connectivity()

        except (ServiceUnavailable, AuthError) as e:
            logger.error("Cannot connect to Neo4j database", exc_info=True)
            raise ConnectionError("Failed to connect to Neo4j database") from e

    def upload_nodes(self, nodes: List[dataclass], dry_run: bool = False) -> None:
        assert all(
            isinstance(node, (ChapterNode, WritingNode, VerseNode, EntityNode)) for node in nodes
        )

        with self.driver.session() as session:
            tx = session.begin_transaction()
            try:
                for node in nodes:
                    query = """
                        MERGE (n:$all($labels) {uuid: $uuid})
                        SET n += $props
                    """
                    props = {
                        k: v
                        for k, v in dataclass.asdict(node)
                        if k not in ["uuid", "labels", "label"]
                    }
                    tx.run(
                        query,
                        labels=node.labels if not isinstance(node, EntityNode) else node.label,
                        uuid=node.uuid,
                        props=props,
                    )

                if dry_run:
                    tx.rollback()

                else:
                    tx.commit()

            except Exception as e:
                tx.rollback()
                logger.error("Could not upload node %s with query %s", node, query, exc_info=True)
                raise type(e)(f"Failed to upload node: {e}") from e

    def upload_relationships(self, rels, rel_type):
        # Link nodes
        pass

    def close(self):
        # Cleanup
        pass
