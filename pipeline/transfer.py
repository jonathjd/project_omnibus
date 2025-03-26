import argparse
import os
from parser import NLP, LatinVulgate, Neo4jLoader, logger
from pathlib import Path


def main(dry_run: bool, sandbox: bool):
    if dry_run:
        logger.info("Current run is a dry run")

    if sandbox:
        URI = os.environ.get("LOCAL_URI")
        USERNAME = os.environ.get("LOCAL_USERNAME")
        PASSWORD = os.environ.get("LOCAL_PASSWORD")
    else:
        URI = os.environ.get("WEB_NEO4J_URI")
        USERNAME = os.environ.get("WEB_NEO4J_USERNAME")
        PASSWORD = os.environ.get("WEB_NEO4J_PASSWORD")

    # Log initial information
    logger.info(f"Starting transfer pipeline")

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
    logger.info("Uploading to AuraDB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse text and upload to neo4j database.")
    parser.add_argument(
        "--dry_run",
        required=False,
        default=True,
        help="Dry run will not upload any data or nodes to the database. Defaults to True",
    )
    parser.add_argument(
        "--sandbox",
        required=False,
        default=True,
        help="Whether to upload to the sandbox database or production database. Defaults to True.",
    )
    args = parser.parse_args()
    main(args.dry_run, args.sandbox)
