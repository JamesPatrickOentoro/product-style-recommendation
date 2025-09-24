#!/usr/bin/env python3
"""Interactive product recommendation UI backed by Cloud SQL and Vertex AI.

This Gradio app loads apparel records with image URLs and pgvector embeddings,
shows them in a gallery, and lets the user pick one. When a product is selected
we compute similar items via cosine similarity and ask Gemini 2.5 Flash to
recommend the best alternative, returning the suggestion and image preview.
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


class RecommendationEngine:
    def __init__(
        self,
        products: List[Product],
        project_id: str,
        location: str,
        model_name: str,
    ) -> None:
        self._products = products
        self._project_id = project_id
        self._location = location
        self._model_name = model_name
        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(model_name)

    def top_candidates(self, source: Product, k: int = 5) -> List[Tuple[Product, float]]:
        scores: List[Tuple[Product, float]] = []
        for candidate in self._products:
            if candidate.product_id == source.product_id:
                continue
            score = cosine_similarity(source.embedding, candidate.embedding)
            scores.append((candidate, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def recommend(self, source: Product, k: int = 5) -> Dict[str, str]:
        candidates = self.top_candidates(source, k=k)
        if not candidates:
            return {"message": "No candidate products available."}

        candidate_payload = []
        for candidate, score in candidates:
            snippet = ", ".join(f"{value:.3f}" for value in candidate.embedding[:15])
            candidate_payload.append(
                {
                    "id": candidate.product_id,
                    "category": candidate.category,
                    "sub_category": candidate.sub_category,
                    "description": candidate.short_description(200),
                    "similarity": round(score, 4),
                    "embedding_snippet": snippet,
                }
            )

        prompt = self._build_prompt(source, candidate_payload)
        logging.debug("Gemini prompt: %s", prompt)

        response = self._model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=512),
        )
        text = response.text if hasattr(response, "text") else str(response)
        logging.debug("Gemini raw response: %s", text)

        recommendation = self._parse_response(text, candidate_payload)
        return recommendation

    def _build_prompt(self, source: Product, candidates: List[Dict[str, object]]) -> str:
        source_snippet = ", ".join(f"{value:.3f}" for value in source.embedding[:15])
        prompt = (
            "You are a retail stylist working with embedding vectors produced by a text-embedding model.\n"
            "Analyze the user's selected product and choose the single best alternative from the candidate list.\n"
            "Always justify your choice using category, sub-category, and description.\n"
            "Respond with JSON containing keys 'recommended_id', 'reason', and 'confidence' (0-1).\n"
            "Selected product:\n"
            f"  id: {source.product_id}\n"
            f"  category: {source.category}\n"
            f"  sub_category: {source.sub_category}\n"
            f"  description: {source.short_description(240)}\n"
            f"  embedding_snippet: [{source_snippet}] (first 15 of {len(source.embedding)} dims)\n"
            "Candidates:\n"
        )
        for candidate in candidates:
            prompt += (
                f"- id: {candidate['id']}, category: {candidate['category']}, "
                f"sub_category: {candidate['sub_category']}, similarity: {candidate['similarity']}, "
                f"description: {candidate['description']}, embedding_snippet: [{candidate['embedding_snippet']}]\n"
            )
        prompt += "If the candidates are unsuitable, choose the closest match but note any limitations in the reason."
        return prompt

    def _parse_response(
        self, text: str, candidates: List[Dict[str, object]]
    ) -> Dict[str, str]:
        try:
            payload = json.loads(text)
            recommended_id = int(payload.get("recommended_id"))
            reason = payload.get("reason", "Gemini did not provide a reason.")
            confidence = payload.get("confidence")
        except (json.JSONDecodeError, TypeError, ValueError):
            logging.warning("Failed to parse Gemini response as JSON. Falling back to top candidate.")
            top_candidate = candidates[0]
            recommended_id = top_candidate["id"]  # type: ignore[index]
            reason = (
                "Falling back to highest cosine similarity match. "
                f"Candidate {recommended_id} with similarity {top_candidate['similarity']}"
            )
            confidence = None

        match = next(
            (product for product in self._products if product.product_id == recommended_id),
            None,
        )
        if match is None:
            logging.warning(
                "Gemini picked id %s which is not in the candidate set. Falling back to first candidate.",
                recommended_id,
            )
            top_candidate = candidates[0]
            match = next(
                product
                for product in self._products
                if product.product_id == top_candidate["id"]
            )
            recommended_id = match.product_id
            if not reason:
                reason = "Gemini selected an unknown product; defaulted to top cosine match."

        result = {
            "recommended_id": str(recommended_id),
            "reason": reason,
            "confidence": f"{confidence:.2f}" if isinstance(confidence, (float, int)) else "N/A",
            "recommended_title": match.title(),
            "recommended_description": match.short_description(240),
            "recommended_image": match.image_url,
        }
        return result


def build_gallery_payload(products: List[Product]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for product in products:
        items.append((product.image_url, product.title()))
    return items


def create_interface(engine: RecommendationEngine, products: List[Product]) -> gr.Blocks:
    gallery_items = build_gallery_payload(products)

    def on_select(evt: gr.SelectData) -> Tuple[str, str, str, str]:
        index = evt.index
        if index is None or index < 0 or index >= len(products):
            return (
                "No product selected.",
                "",
                "",
                "",
            )
        product = products[index]
        rec = engine.recommend(product)
        selected_markdown = (
            f"### Selected Product\n"
            f"**{product.title()}**\n\n"
            f"{product.short_description(400)}\n\n"
            f"Embedding dims: {len(product.embedding)}"
        )
        recommendation_md = (
            f"### Gemini Suggests\n"
            f"**{rec['recommended_title']}**\n\n"
            f"{rec['reason']}\n\n"
            f"Confidence: {rec['confidence']}"
        )
        return (
            selected_markdown,
            recommendation_md,
            rec["recommended_image"],
            json.dumps({k: v for k, v in rec.items() if k != "recommended_image"}, indent=2),
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
            recommended_image = gr.Image(label="Recommended Product", interactive=False)
            raw_json = gr.Code(label="Recommendation payload", language="json")

        gallery.select(
            on_select,
            inputs=None,
            outputs=[selected_info, recommendation_info, recommended_image, raw_json],
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
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_dsn(args: argparse.Namespace) -> str:
    required = {"db password": args.db_password, "vertex project": args.vertex_project}
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

    engine = RecommendationEngine(products, args.vertex_project, args.vertex_location, args.model)
    logging.info("Loaded %s products. Launching Gradio interface…", len(products))
    app = create_interface(engine, products)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
