import argparse
import os
import sys
import traceback
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models


from app.config import settings

_logfire_configured = False


def configure_logfire() -> None:
    """Configure Logfire only when ingestion actually runs."""
    global _logfire_configured
    if not _logfire_configured:
        logfire.configure(service_name="enterprise-ingestion-service")
        _logfire_configured = True

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def ensure_collection_embedding_compatible() -> None:
    """Fail before ingestion if the collection does not use the Gemini vector size."""
    from app.services.retrieval.embedding import get_embedding_dim

    collection = qdrant_client.get_collection(settings.QDRANT_COLLECTION)
    vectors = collection.config.params.vectors
    vector_size = getattr(vectors, "size", None)
    expected_size = get_embedding_dim()
    if vector_size != expected_size:
        raise RuntimeError(
            f"Collection '{settings.QDRANT_COLLECTION}' uses {vector_size} dimensions, "
            f"but the configured Gemini model requires {expected_size}."
        )


def ensure_source_payload_indexes() -> None:
    """Create the keyword indexes required to find prior points for a source."""
    for field_name in ("source", "source_type"):
        qdrant_client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


def existing_source_point_ids(filename: str, source_type: str) -> set[str]:
    """Return current point IDs for a source so legacy random IDs can be replaced safely."""
    source_filter = models.Filter(
        must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=filename)),
            models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
        ]
    )
    point_ids: set[str] = set()
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=source_filter,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.update(str(point.id) for point in points)
        if offset is None:
            return point_ids


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest

def process_file(file_path: str, filename: str, source_type: str) -> bool:
    """Parse → chunk → save locally → embed → index in Qdrant."""
    configure_logfire()
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                from app.ingestion.loaders.pdf import parse_pdf

                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                from app.ingestion.loaders.html import parse_html

                full_text = parse_html(file_path)
            elif ext == "txt":
                from app.ingestion.loaders.text import parse_text

                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return False

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return False

            # 2. Chunk text
            from app.ingestion.chunking.splitter import chunk_text

            chunks = chunk_text(full_text)
            if not chunks:
                logfire.warning(f"No chunks generated for {filename} — skipping.")
                return False

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                from app.services.retrieval.embedding import (
                    embed_texts,
                    get_embedding_dim,
                    get_embedding_model_name,
                )

                embeddings = embed_texts(chunks)
                if len(embeddings) != len(chunks):
                    raise RuntimeError(
                        f"Embedding count mismatch for {filename}: "
                        f"expected {len(chunks)}, received {len(embeddings)}."
                    )
                existing_ids = existing_source_point_ids(filename, source_type)
                source_path = os.path.relpath(file_path).replace(os.sep, "/")
                embedding_model = get_embedding_model_name()
                point_ids = [
                    str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{source_type}:{source_path}:{chunk_index}",
                    ))
                    for chunk_index in range(len(chunks))
                ]
                points = [
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_path": source_path,
                            "source_type": source_type,
                            "embedding_model": embedding_model,
                            "embedding_dimension": get_embedding_dim(),
                        },
                    )
                    for point_id, chunk, vector in zip(point_ids, chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                    wait=True,
                )
                stale_ids = existing_ids.difference(point_ids)
                if stale_ids:
                    qdrant_client.delete(
                        collection_name=settings.QDRANT_COLLECTION,
                        points_selector=models.PointIdsList(points=list(stale_ids)),
                        wait=True,
                    )
                    logfire.info(f"Replaced {len(stale_ids)} legacy points from {filename}.")
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
            return True

        except Exception:
            # Keep batch ingestion running, but make individual failures diagnosable.
            logfire.exception(f"Failed to process {filename}")
            print(f"\nFailed to process {filename}:", file=sys.stderr)
            traceback.print_exc()
            return False


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def infer_source_type(path: str) -> str:
    """Infer the retrieval source type from a file or directory path."""
    path_lower = os.path.normpath(path).lower()
    if "true" in path_lower:
        return "true"
    if "noisy" in path_lower:
        return "noisy"
    return "general"


def process_selected_files(file_paths: list[str], source_type: str | None = None) -> list[str]:
    """Process only the requested files and return the ones that failed."""
    configure_logfire()
    failed_files: list[str] = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}", file=sys.stderr)
            failed_files.append(file_path)
            continue

        filename = os.path.basename(file_path)
        file_source_type = source_type or infer_source_type(file_path)
        if not process_file(file_path, filename, file_source_type):
            failed_files.append(file_path)
    return failed_files


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    configure_logfire()
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            from app.services.retrieval.embedding import get_embedding_dim

            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        ensure_collection_embedding_compatible()
        ensure_source_payload_indexes()

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local documents into Qdrant.")
    parser.add_argument("target_dir", nargs="?", default="DATA")
    parser.add_argument("source_type", nargs="?")
    parser.add_argument("--wipe", action="store_true", help="Delete and recreate the collection.")
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Process only these files. Do not use with --wipe.",
    )
    parser.add_argument(
        "--source-type",
        help="Override the source type for --files (for example: true or noisy).",
    )
    args = parser.parse_args()

    if args.files:
        if args.wipe:
            parser.error("--wipe cannot be used with --files.")
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            parser.error(f"Collection '{settings.QDRANT_COLLECTION}' does not exist.")
        ensure_collection_embedding_compatible()
        ensure_source_payload_indexes()

        failed_files = process_selected_files(args.files, args.source_type)
        if failed_files:
            print(f"\nIngestion failed for {len(failed_files)} file(s):", file=sys.stderr)
            for file_path in failed_files:
                print(f"- {file_path}", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.path.exists(args.target_dir):
            parser.error(f"Path '{args.target_dir}' does not exist.")
        run_universal_ingestion(
            args.target_dir,
            explicit_source_type=args.source_type,
            wipe=args.wipe,
        )

    logfire.info("Ingestion job completed.")
    logfire.info("Ingestion job completed.")