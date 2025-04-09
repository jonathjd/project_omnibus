import re
from src.models import WritingNode, ChapterNode, VerseNode, EntityNode, VerseSimilarity
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from constants import DRB
from logging_config import setup_logging

logger = setup_logging()


# ---------------------
# LatinVulgate Class
# ---------------------
class LatinVulgate:
    """
    Handles text preprocessing for the Latin Vulgate, including:
     - Reading and cleaning raw text
     - Parsing into Writing, Chapter, Verse nodes
    """

    WORKS = DRB["WORKS"]

    def __init__(self, raw_text: Path):
        # save raw text path and clean
        self._raw_text_path = raw_text
        self._title = raw_text.stem
        self._cleaned_text_path = (
            Path("data") / "cleaned" / f"{date.today().strftime('%Y-%m-%d')}-{self._title}.txt"
        )

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

    EXCLUSIONS = DRB["EXCLUSIONS"]
    NAME_ALIASES = DRB["NAME_ALIASES"]
    PATTERNS = [
        {"label": "PERSON", "pattern": "Esau"},
        {"label": "PERSON", "pattern": "Ramesses"},
        {"label": "PERSON", "pattern": "Pharao"},
    ]

    INCLUDED_ENTS = ["PERSON", "NORP", "LOC", "GPE"]

    def __init__(self):
        """
        Initialize NLP with a spaCy model, custom patterns, and a sentence transformer.
        """
        # NLP model for entity recognition
        self._nlp = spacy.load("en_core_web_trf")

        # Model for embeddings
        self._model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Custom patterns
        ruler = self._nlp.add_pipe("entity_ruler", after="ner")
        ruler.add_patterns(self.PATTERNS)

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
            >>> entities = parser.extract_entities(verses, chunk_size=750)
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
                doc = self._nlp(text)
                for ent in doc.ents:
                    # Add only wanted entities
                    if ent.label_ in self.INCLUDED_ENTS:
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
        embeddings = self._model.encode(verse_texts, batch_size=256, show_progress_bar=True)
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

        similar_pairs = []
        for i in range(len(embeddings)):
            for j in range(1, n_neighbors):  # start from 1 to skip self-similarity
                verse_1 = verse_nodes[i]
                verse_2 = verse_nodes[indices[i, j]]
                sim_score = float(similarities[i, j])

                similar_pairs.append(
                    VerseSimilarity(
                        verse_id_1=verse_1.uuid, verse_id_2=verse_2.uuid, similarity=sim_score
                    )
                )

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

    @property
    def model(self):
        return self._model


class Neo4jLoader:
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
            # initialize connection
            self._driver = GraphDatabase.driver(
                self.__neo4j_uri, auth=(self.__neo4j_username, self.__neo4j_password)
            )
            self._driver.verify_connectivity()

        except (ServiceUnavailable, AuthError) as e:
            logger.error("Cannot connect to Neo4j database", exc_info=True)
            raise ConnectionError("Failed to connect to Neo4j database") from e

    def upload_nodes(self, nodes: List[dataclass], dry_run: bool = True) -> None:
        """
        Uploads nodes to Neo4j using MERGE and optional dry run.

        Args:
            nodes (List[dataclass]): A list of WritingNode, ChapterNode, VerseNode, or EntityNode objects.
            dry_run (bool): If True, rolls back the transaction without committing changes.

        Example:
            >>> loader = Neo4jLoader("bolt://localhost:7687", "neo4j", "password")
            >>> nodes = [WritingNode(author="John", title="Sample Book")]
            >>> loader.upload_nodes(nodes, dry_run=True)
        """
        assert all(
            isinstance(node, (ChapterNode, WritingNode, VerseNode, EntityNode)) for node in nodes
        ), f"Invalid node types in node list! {[node for node in nodes if not isinstance(node, (ChapterNode, WritingNode, VerseNode, EntityNode))]}"

        logger.info(f"Attempting to upload {len(nodes)} nodes into neo4j DB...")

        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                for node in nodes:
                    sanitized_labels = []
                    for label in node.labels if not isinstance(node, EntityNode) else [node.label]:
                        label_escaped = label.replace("`", "``")
                        sanitized_labels.append(f"`{label_escaped}`")

                    final_labels = ":".join(sanitized_labels)

                    query = f"""
                        MERGE (n:{final_labels} {{uuid: $uuid}})
                        SET n += $props
                    """
                    props = {
                        k: v
                        for k, v in asdict(node).items()
                        if k not in ["uuid", "labels", "label"]
                    }

                    tx.run(query, uuid=str(node.uuid), props=props)

                if dry_run:
                    tx.rollback()
                else:
                    tx.commit()
            except Exception as e:
                tx.rollback()
                logger.error(f"Failed to upload node: {e}", exc_info=True)
                raise

    def upload_relationships(
        self, rel_nodes: List[VerseSimilarity], chunk_size: int = 5000, dry_run: bool = True
    ) -> None:
        """
        Uploads SIMILAR_TO relationships between verse nodes.

        Args:
            rel_nodes (List[VerseSimilarity]): Relationship data to be uploaded.
            dry_run (bool): If True, rolls back the transaction without committing changes.

        Example:
            >>> loader = Neo4jLoader("bolt://localhost:7687", "neo4j", "password")
            >>> rel_nodes = [VerseSimilarity(verse_id_1={uuid}, verse_id_2=uuid, similiarity=0.7117224)]
            >>> loader.upload_relationships(nodes, dry_run=True)

        Raises:
            Exception: If relationship creation fails.

        Note:
            Both verse nodes must already exist in the database for this to succeed.
        """

        assert all(
            isinstance(node, VerseSimilarity) for node in rel_nodes
        ), f"Invalid node types in node list! {[node for node in rel_nodes if not isinstance(node, VerseSimilarity)]}"
        logger.info(f"Attempting to upload {len(rel_nodes)} relationships into neo4j DB...")

        query = """
        UNWIND $relNodes AS rel
        MATCH (v1 {uuid: rel.verse_id1}), (v2 {uuid: rel.verse_id2})
        MERGE (v1)-[r:SIMILAR_TO]->(v2)
        SET r.cos_sim = rel.similarity
        """

        for i in range(0, len(rel_nodes), chunk_size):
            logger.info(f"Uploading relationship nodes {i} through {i + chunk_size}...")

            params = [
                {
                    "verse_id1": str(node.verse_id_1),
                    "verse_id2": str(node.verse_id_2),
                    "similarity": node.similarity,
                }
                for node in rel_nodes[i : i + chunk_size]
            ]

            with self._driver.session() as session:
                try:
                    tx = session.begin_transaction()
                    tx.run(query, relNodes=params)

                    if dry_run:
                        tx.rollback()
                    else:
                        tx.commit()
                except Exception as e:
                    tx.rollback()
                    logger.error(f"Failed to upload node: {e}", exc_info=True)
                    raise

    def close(self):
        # Cleanup
        self._driver.close()
        pass

    @property
    def neo4j_uri(self):
        return self.__neo4j_uri

    @property
    def neo4j_username(self):
        return self.__neo4j_username

    @property
    def neo4j_password(self):
        return self.__neo4j_password

    @neo4j_uri.setter
    def neo4j_uri(self, value):
        assert isinstance(value, str) and any(
            value.startswith(prefix)
            for prefix in ["bolt://", "neo4j://", "neo4j+s://", "neo4j+ssc://"]
        ), "neo4j_uri must be a string starting with one of ['bolt://','neo4j://','neo4j+s://','neo4j+ssc://']"
        self.__neo4j_uri = value

    @neo4j_username.setter
    def neo4j_username(self, value):
        assert (
            isinstance(value, str) and len(value) > 0
        ), "neo4j_username must be a non-empty string"
        self.__neo4j_username = value

    @neo4j_password.setter
    def neo4j_password(self, value):
        assert (
            isinstance(value, str) and len(value) > 0
        ), "neo4j_username must be a non-empty string"
        self.__neo4j_password = value
