#!/usr/bin/env python3
"""Populate the Cloud SQL apparels.embedding column with Vertex AI embeddings.

The script pulls rows whose embedding is null, generates an embedding for each
row using a Vertex AI text-embedding model (text-embedding-005 by default) and writes the
result back into the table. Make sure you run the Cloud SQL Auth Proxy (or have
another secure connection method) before executing this tool.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_batch

try:
    import vertexai
    from vertexai.preview.language_models import (
        TextEmbeddingInput,
        TextEmbeddingModel,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Install with: pip install google-cloud-aiplatform"
    ) from exc


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
DEFAULT_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
DEFAULT_TASK_TYPE = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")


@dataclass
class AppConfig:
    project_id: str
    location: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    dry_run: bool = False
    embedding_task_type: str = DEFAULT_TASK_TYPE
    embedding_output_dim: Optional[int] = None


class EmbeddingClient:
    """Wrapper for Vertex text embedding models."""

    def __init__(
        self,
        project_id: str,
        location: str,
        model_name: str,
        task_type: str,
        output_dimensionality: Optional[int] = None,
    ) -> None:
        self._model_name = model_name
        self._task_type = task_type
        self._output_dimensionality = output_dimensionality

        lowered = model_name.lower()
        if "gemini" in lowered and not lowered.startswith("text-embedding"):
            raise SystemExit(
                "Gemini generative models do not expose embeddings via TextEmbeddingModel. "
                "Use a text-embedding-* model such as text-embedding-005."
            )

        logging.info(
            "Initialising Vertex AI embedding model %s in %s",
            model_name,
            location,
        )
        vertexai.init(project=project_id, location=location)
        try:
            self._model = TextEmbeddingModel.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                f"Failed to load embedding model '{model_name}'. Make sure the model"
                " exists in region {location} or choose a supported model."
            ) from exc

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        inputs = [
            TextEmbeddingInput(task_type=self._task_type, text=text)
            for text in texts
        ]

        kwargs = {}
        if self._output_dimensionality is not None:
            kwargs["output_dimensionality"] = self._output_dimensionality

        responses = self._model.get_embeddings(inputs, **kwargs)
        return [list(response.values) for response in responses]
def fetch_batch(
    conn: psycopg2.extensions.connection,
    limit: int,
    include_description: bool,
) -> Tuple[List[int], List[str]]:
    column_list = [
        "id",
        "category",
        "sub_category",
        "content",
        "uri",
        "image",
    ]
    if include_description:
        column_list.append("pdt_desc")

    sql = (
        "SELECT "
        + ",".join(column_list)
        + " FROM apparels WHERE embedding IS NULL ORDER BY id FOR UPDATE SKIP LOCKED LIMIT %s"
    )

    ids: List[int] = []
    payloads: List[str] = []

    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        for row in rows:
            data = dict(zip(column_list, row))
            ids.append(data["id"])
            payloads.append(_build_payload(data))

    return ids, payloads


def _build_payload(row: dict) -> str:
    description = row.get("pdt_desc") or ""
    description = description.strip()
    if not description:
        description = (
            "Category: {category}. Sub-category: {sub_category}. "
            "Content: {content}. Image URI: {uri}."
        ).format(**row)
    return description


def ensure_pdt_desc_column(
    conn: psycopg2.extensions.connection,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'apparels'
              AND column_name = 'pdt_desc'
            """
        )
        return cur.fetchone() is not None


def write_embeddings(
    conn: psycopg2.extensions.connection,
    ids: Sequence[int],
    embeddings: Sequence[Sequence[float]],
    expected_dim: int,
    dry_run: bool,
) -> None:
    if len(ids) != len(embeddings):
        raise ValueError("IDs and embeddings length mismatch")

    rows = []
    for apparel_id, vector in zip(ids, embeddings):
        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding length {len(vector)} for id {apparel_id} does not match expected {expected_dim}"
            )
        rows.append((_format_vector(vector), apparel_id))

    if dry_run:
        logging.info("Dry-run: skipping update for %d rows", len(rows))
        return

    with conn.cursor() as cur:
        execute_batch(
            cur,
            "UPDATE apparels SET embedding = %s::vector WHERE id = %s",
            rows,
            page_size=len(rows) or 1,
        )
    conn.commit()


def _format_vector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in vector) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Back-fill Gemini embeddings into Cloud SQL")
    parser.add_argument("--project-id", default=os.getenv("VERTEX_PROJECT"), help="Google Cloud project for Vertex AI")
    parser.add_argument("--location", default=os.getenv("VERTEX_LOCATION", "us-central1"), help="Vertex AI region")
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "apparel_db"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-task-type",
        default=DEFAULT_TASK_TYPE,
        help="Task type passed to TextEmbeddingInput",
    )
    parser.add_argument(
        "--embedding-output-dim",
        type=int,
        default=None,
        help="Optional output dimensionality override",
    )
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional upper bound on rows to update")
    parser.add_argument("--dry-run", action="store_true", help="Compute embeddings but do not persist")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    missing = [
        name
        for name, value in {
            "project id": args.project_id,
            "location": args.location,
            "db password": args.db_password,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required configuration: {', '.join(missing)}")

    return AppConfig(
        project_id=args.project_id,
        location=args.location,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        embedding_task_type=args.embedding_task_type,
        embedding_output_dim=args.embedding_output_dim,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    config = build_config(args)
    logging.info("Connecting to Cloud SQL at %s:%s/%s", config.db_host, config.db_port, config.db_name)

    with psycopg2.connect(
        host=config.db_host,
        port=config.db_port,
        dbname=config.db_name,
        user=config.db_user,
        password=config.db_password,
    ) as conn:
        conn.autocommit = False
        has_pdt_desc = ensure_pdt_desc_column(conn)
        logging.info("pdt_desc column available: %s", has_pdt_desc)

        embedder = EmbeddingClient(
            config.project_id,
            config.location,
            config.embedding_model,
            config.embedding_task_type,
            config.embedding_output_dim,
        )

        processed = 0
        while True:
            if args.max_rows is not None and processed >= args.max_rows:
                break

            to_fetch = config.batch_size
            if args.max_rows is not None:
                to_fetch = min(to_fetch, args.max_rows - processed)

            ids, payloads = fetch_batch(conn, to_fetch, has_pdt_desc)
            if not ids:
                logging.info("No more rows with NULL embedding found. Exiting.")
                break

            logging.info("Generating embeddings for ids %s", ids)
            embeddings = embedder.embed(payloads)
            write_embeddings(conn, ids, embeddings, config.embedding_dim, config.dry_run)
            processed += len(ids)

        logging.info("Finished updating embeddings for %d rows", processed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logging.exception("Embedding back-fill failed: %s", exc)
        sys.exit(1)
