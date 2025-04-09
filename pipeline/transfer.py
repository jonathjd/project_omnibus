import argparse
import os
from pathlib import Path
from time import perf_counter

import dotenv

dotenv.load_dotenv()

from src.parser import NLP, LatinVulgate, Neo4jLoader, logger


def main(dry_run: bool, sandbox: bool):
    start = perf_counter()
    if dry_run:
        logger.info("Current run is a dry run")

    try:
        if sandbox:
            URI = os.environ["LOCAL_URI"]
            USERNAME = os.environ["LOCAL_USERNAME"]
            PASSWORD = os.environ["LOCAL_PASSWORD"]
        else:
            URI = os.environ["WEB_NEO4J_URI"]
            USERNAME = os.environ["WEB_NEO4J_USERNAME"]
            PASSWORD = os.environ["WEB_NEO4J_PASSWORD"]

    except KeyError as e:
        raise Exception(f"Missing required environment variable: {e}")

    vulgate_path = Path("data/raw/douay-rheims-bible.txt")
    logger.info(f"Parsing {vulgate_path}")

    # clean text and create work, verse, chapter nodes
    text_parser = LatinVulgate(vulgate_path)
    logger.info(f"Cleaned text path: {text_parser.cleaned_text_path}")
    logger.info(f"Text title: {text_parser.title}")

    # NLP
    nlp = NLP()
    logger.info("Extracting named entities from text")
    entities = nlp.extract_entities(text_parser.verse_nodes)
    logger.info(f"Found {len(entities)} entities.")
    for node in entities[0:10]:
        logger.info(f"Example entity node: {node}")

    # embeddings and similarities
    logger.info(f"Computing text embeddings with {nlp.model.model_card_data}")
    verse_nodes = nlp.compute_embeddings(text_parser.verse_nodes)
    similarity_nodes = nlp.generate_relationship_nodes(verse_nodes)
    logger.info(f"Number of relationships: {len(similarity_nodes)}")

    # distribution and visualize
    logger.info("Generate KDE plot of similarity distribution")
    nlp.similarity_distribution(similarity_nodes)

    # upload to DB
    logger.info(f"Uploading to Neo4j DB {URI}")
    neo4j_loader = Neo4jLoader(URI, USERNAME, PASSWORD)
    neo4j_loader.upload_nodes(verse_nodes, dry_run)
    neo4j_loader.upload_nodes(text_parser.work_nodes, dry_run)
    neo4j_loader.upload_nodes(text_parser.chapter_nodes, dry_run)
    neo4j_loader.upload_nodes(entities, dry_run)
    neo4j_loader.upload_relationships(similarity_nodes, 10000, dry_run)
    neo4j_loader.close()
    end = perf_counter()

    time_elapsed = end - start
    mins = int(time_elapsed // 60)
    secs = time_elapsed % 60

    logger.info(f"Pipeline run completed in {mins} minutes {secs:2f} seconds!")


if __name__ == "__main__":

    # helper function to convert truthy string input to bool
    def str2bool(value):
        return value.lower() in ("true", "t", "yes", "1")

    parser = argparse.ArgumentParser(description="Parse text and upload to neo4j database.")
    parser.add_argument(
        "--dry_run",
        type=str2bool,
        required=False,
        default=True,
        help="Dry run will not upload any data or nodes to the database. Defaults to True",
    )
    parser.add_argument(
        "--sandbox",
        type=str2bool,
        required=False,
        default=True,
        help="Whether to upload to the sandbox database or production database. Defaults to True.",
    )
    args = parser.parse_args()
    main(args.dry_run, args.sandbox)
