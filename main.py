#!/usr/bin/env python3
"""Interactive product recommendation UI backed by Cloud SQL and Vertex AI.

This Gradio app loads apparel records with image URLs and pgvector embeddings,
shows them in a gallery, and lets the user pick one. When a product is selected
we feed the product narrative (``pdt_desc``) to Gemini 2.5 Flash, ask it to
write a fresh description for a complementary item, embed that generated copy,
and look up the nearest neighbours inside Cloud SQL to suggest the top match.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import psycopg2
from psycopg2.extras import DictCursor

try:
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    from vertexai.preview.language_models import TextEmbeddingInput, TextEmbeddingModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Install with: pip install google-cloud-aiplatform gradio"
    ) from exc

DEFAULT_DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DEFAULT_DB_PORT = int(os.getenv("DB_PORT", "5432"))
DEFAULT_DB_NAME = os.getenv("DB_NAME", "apparel_db")
DEFAULT_DB_USER = os.getenv("DB_USER", "postgres")
DEFAULT_VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
DEFAULT_VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
DEFAULT_EMBEDDING_TASK = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")
DEFAULT_LIMIT = int(os.getenv("PRODUCT_LIMIT", "30"))


@dataclass
class Product:
    product_id: int
    category: str
    sub_category: str
    uri: str
    image: str
    description: str
    embedding: List[float]

    @property
    def image_url(self) -> str:
        return self.uri or self.image

    def title(self) -> str:
        return f"#{self.product_id} · {self.category} / {self.sub_category}"

    def short_description(self, max_len: int = 180) -> str:
        text = self.description.strip() or "(no description)"
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"


class ProductRepository:
    """Loads apparel products with embeddings from Cloud SQL."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def fetch_products(self, limit: int) -> List[Product]:
        query = (
            "SELECT id, category, sub_category, uri, image, COALESCE(pdt_desc, content) AS description, "
            "embedding::text AS embedding_text "
            "FROM apparels WHERE embedding IS NOT NULL ORDER BY id LIMIT %s"
        )

        with psycopg2.connect(self._dsn, cursor_factory=DictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()

        products: List[Product] = []
        for row in rows:
            embedding = parse_embedding(row["embedding_text"])
            products.append(
                Product(
                    product_id=row["id"],
                    category=row["category"],
                    sub_category=row["sub_category"],
                    uri=row["uri"],
                    image=row["image"],
                    description=row["description"],
                    embedding=embedding,
                )
            )
        return products

    def find_similar_by_vector(
        self,
        vector: Sequence[float],
        limit: int,
        exclude_id: Optional[int] = None,
    ) -> List[Tuple[Product, float]]:
        vector_literal = format_vector(vector)

        base_query = (
            "SELECT id, category, sub_category, uri, image, COALESCE(pdt_desc, content) AS description, "
            "embedding::text AS embedding_text, embedding <=> %s::vector AS distance "
            "FROM apparels WHERE embedding IS NOT NULL"
        )
        params: List[object] = [vector_literal]

        if exclude_id is not None:
            base_query += " AND id <> %s"
            params.append(exclude_id)

        base_query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([vector_literal, limit])

        with psycopg2.connect(self._dsn, cursor_factory=DictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(base_query, params)
                rows = cur.fetchall()

        neighbours: List[Tuple[Product, float]] = []
        for row in rows:
            product = Product(
                product_id=row["id"],
                category=row["category"],
                sub_category=row["sub_category"],
                uri=row["uri"],
                image=row["image"],
                description=row["description"],
                embedding=parse_embedding(row["embedding_text"]),
            )
            distance = float(row["distance"])
            neighbours.append((product, distance))

        return neighbours


def parse_embedding(value: Optional[str]) -> List[float]:
    if value is None:
        raise ValueError("Encountered NULL embedding while parsing result set")
    cleaned = value.replace("(", "[").replace(")", "]")  # pgvector parentheses → brackets
    try:
        numbers = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Failed to parse embedding: {value!r}") from exc
    return [float(x) for x in numbers]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def format_vector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in vector) + "]"


class RecommendationEngine:
    def __init__(
        self,
        repo: ProductRepository,
        products: List[Product],
        project_id: str,
        location: str,
        model_name: str,
        embedding_model_name: str,
        embedding_task_type: str,
        embedding_output_dim: Optional[int] = None,
    ) -> None:
        self._repo = repo
        self._products = products
        self._project_id = project_id
        self._location = location
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name
        self._embedding_task_type = embedding_task_type
        self._embedding_output_dim = embedding_output_dim

        vertexai.init(project=project_id, location=location)
        self._gen_model = GenerativeModel(model_name)
        self._embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)

    def recommend(self, source: Product, k: int = 5) -> Dict[str, object]:
        logging.debug("Requesting Gemini recommendation for product %s", source.product_id)
        prompt = self._build_prompt(source)
        response = self._gen_model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.9, max_output_tokens=256),
        )
        text = response.text if hasattr(response, "text") else str(response)
        logging.debug("Gemini raw response: %s", text)

        gen_payload = self._parse_generation_response(text)
        generated_desc = gen_payload.get("generated_description")
        if not generated_desc:
            logging.warning("Gemini response missing 'generated_description'. Falling back to original pdt_desc.")
            generated_desc = source.description

        logging.debug("Embedding generated description: %s", generated_desc)
        embedding = self._embed_text(generated_desc)
        neighbours = self._repo.find_similar_by_vector(
            embedding,
            limit=k,
            exclude_id=source.product_id,
        )

        if not neighbours:
            logging.warning("No neighbours returned for generated description; reverting to preloaded catalogue neighbours.")
            neighbours = self._fallback_neighbours(source, k)

        results = []
        for product, distance in neighbours:
            similarity = max(0.0, 1.0 - distance)
            results.append(
                {
                    "id": product.product_id,
                    "title": product.title(),
                    "description": product.short_description(240),
                    "image": product.image_url,
                    "distance": round(distance, 4),
                    "similarity": round(similarity, 4),
                }
            )

        recommendation = {
            "generated_description": generated_desc,
            "reason": gen_payload.get("reason", "Gemini did not explain the recommendation."),
            "confidence": gen_payload.get("confidence", "N/A"),
            "results": results,
        }
        return recommendation

    def _fallback_neighbours(self, source: Product, k: int) -> List[Tuple[Product, float]]:
        scores: List[Tuple[Product, float]] = []
        for candidate in self._products:
            if candidate.product_id == source.product_id:
                continue
            distance = 1.0 - cosine_similarity(source.embedding, candidate.embedding)
            scores.append((candidate, distance))
        scores.sort(key=lambda item: item[1])
        return scores[:k]

    def _embed_text(self, text: str) -> List[float]:
        inputs = [TextEmbeddingInput(task_type=self._embedding_task_type, text=text)]
        kwargs = {}
        if self._embedding_output_dim is not None:
            kwargs["output_dimensionality"] = self._embedding_output_dim
        responses = self._embedding_model.get_embeddings(inputs, **kwargs)
        if not responses:
            raise ValueError("Embedding model returned no results")
        return list(responses[0].values)

    def _build_prompt(self, source: Product) -> str:
        return (
            "You are a fashion stylist. Study the given product description and propose a complementary"
            " apparel item. Respond with ONE line only, no quotes, no JSON, and no commentary.\n"
            "Your line must look like a concise product description, e.g. 'Girls Tops Red Casual Gini and Jony Girls Red Top'.\n"
            "Do not copy the wording verbatim from the input—choose a complementary category, colour, or style."
            "\nInput product description:\n"
            f"{source.description}\n"
        )

    def _parse_generation_response(self, text: str) -> Dict[str, object]:
        cleaned = text.strip().replace("\n", " ").strip()
        if not cleaned:
            cleaned = "No recommendation generated."

        return {
            "generated_description": cleaned,
            "reason": "Gemini generated a complementary apparel description for embedding lookup.",
            "confidence": "N/A",
        }


def build_gallery_payload(products: List[Product]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for product in products:
        items.append((product.image_url, product.title()))
    return items


def create_interface(engine: RecommendationEngine, products: List[Product]) -> gr.Blocks:
    gallery_items = build_gallery_payload(products)

    def on_select(evt: gr.SelectData) -> Tuple[str, str, List[Tuple[str, str]], str]:
        index = evt.index
        if index is None or index < 0 or index >= len(products):
            return (
                "No product selected.",
                "",
                [],
                "",
            )
        product = products[index]
        rec = engine.recommend(product)
        top_result = rec["results"][0] if rec.get("results") else None
        selected_markdown = (
            f"### Selected Product\n"
            f"**{product.title()}**\n\n"
            f"**Input product description**\n\n{product.description}\n\n"
            f"Embedding dims: {len(product.embedding)}"
        )
        recommendation_md = "### Gemini Suggests\n"
        recommendation_md += f"**Gemini recommended description**\n\n{rec['generated_description']}\n\n"
        recommendation_md += f"**Reason:** {rec['reason']}\n\n"
        recommendation_md += f"**Confidence:** {rec['confidence']}\n"
        if top_result:
            recommendation_md += (
                "\n**Top match from catalogue**\n\n"
                f"{top_result['title']} (similarity {top_result['similarity']})\n"
                f"{top_result['description']}"
            )
        gallery_payload = [
            (item["image"], f"{item['title']}\n(similarity {item['similarity']})")
            for item in rec.get("results", [])
        ]

        return (
            selected_markdown,
            recommendation_md,
            gallery_payload,
            json.dumps(rec, indent=2),
        )

    with gr.Blocks(title="Delami Apparel Recommender") as demo:
        gr.Markdown("# Delami Apparel Explorer\nSelect an item to see what Gemini recommends next.")
        gallery = gr.Gallery(
            value=gallery_items,
            label="Catalog",
            columns=4,
            show_label=True,
            height=400,
        )
        with gr.Row():
            selected_info = gr.Markdown("Select a product to view details.")
            recommendation_info = gr.Markdown("")
        with gr.Row():
            recommended_gallery = gr.Gallery(
                label="Top 5 Matches",
                columns=5,
                show_label=True,
                height=240,
                interactive=False,
            )
            raw_json = gr.Code(label="Recommendation payload", language="json")

        gallery.select(
            on_select,
            inputs=None,
            outputs=[selected_info, recommendation_info, recommended_gallery, raw_json],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the apparel recommendation UI")
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    parser.add_argument("--vertex-project", default=DEFAULT_VERTEX_PROJECT)
    parser.add_argument("--vertex-location", default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-task-type", default=DEFAULT_EMBEDDING_TASK)
    parser.add_argument(
        "--embedding-output-dim",
        type=int,
        default=None,
        help="Optional reduced embedding dimensionality",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_dsn(args: argparse.Namespace) -> str:
    required = {
        "db password": args.db_password,
        "vertex project": args.vertex_project,
        "embedding model": args.embedding_model,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required configuration: {', '.join(missing)}")
    return (
        f"host={args.db_host} port={args.db_port} dbname={args.db_name} "
        f"user={args.db_user} password={args.db_password}"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    dsn = build_dsn(args)
    repo = ProductRepository(dsn)
    logging.info("Loading up to %s products from Cloud SQL", args.limit)
    products = repo.fetch_products(args.limit)
    if not products:
        raise SystemExit("No products with embeddings found. Run generate_embeddings.py first.")

    engine = RecommendationEngine(
        repo,
        products,
        args.vertex_project,
        args.vertex_location,
        args.model,
        args.embedding_model,
        args.embedding_task_type,
        args.embedding_output_dim,
    )
    logging.info("Loaded %s products. Launching Gradio interface…", len(products))
    app = create_interface(engine, products)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
