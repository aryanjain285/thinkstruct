"""Index mapping and bulk loading."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from patsearch.config import ROOT
from patsearch.models import SearchRecord

SYNONYMS_PATH = ROOT / "config" / "synonyms.txt"

def load_synonyms(path: Path | None = None) -> list[str]:
    """Read Solr-format synonym groups from config/synonyms.txt.

    Externalised so the vocabulary can be tuned for a different corpus (chemical,
    software, non-English) without touching code. Missing file is not an error —
    the index simply has no synonym expansion.
    """
    path = path or SYNONYMS_PATH
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out

_BASE_FILTERS = {
    "english_possessive_stemmer": {"type": "stemmer", "language": "possessive_english"},
    "english_stop": {"type": "stop", "stopwords": "_english_"},
    "english_stemmer": {"type": "stemmer", "language": "english"},
}


def build_analysis(synonyms: list[str] | None = None) -> dict[str, Any]:
    """Analyzer definition. Synonyms are applied before stemming so that expanded
    forms are stemmed consistently with everything else."""
    syns = load_synonyms() if synonyms is None else synonyms
    filters = dict(_BASE_FILTERS)
    chain = ["lowercase"]
    if syns:
        filters["patent_spelling"] = {"type": "synonym_graph", "synonyms": syns}
        chain.append("patent_spelling")
    chain += ["english_possessive_stemmer", "english_stop", "english_stemmer"]

    return {
        "filter": filters,
        "analyzer": {
            "patent_english": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": chain,
            }
        },
    }


def build_mapping(dimension: int | None) -> dict[str, Any]:
    props: dict[str, Any] = {
        "record_id": {"type": "keyword"},
        "patent_id": {"type": "keyword"},
        "record_type": {"type": "keyword"},
        "text": {"type": "text", "analyzer": "patent_english"},
        "title": {
            "type": "text",
            "analyzer": "patent_english",
            "fields": {"exact": {"type": "keyword"}},
        },
        "abstract": {"type": "text", "analyzer": "patent_english"},
        "classification_raw": {"type": "keyword"},
        "classification_section": {"type": "keyword"},
        "classification_class": {"type": "keyword"},
        "classification_subclass": {"type": "keyword"},
        "claim_number": {"type": "integer"},
        "is_independent": {"type": "boolean"},
        "paragraph_start": {"type": "integer"},
        "paragraph_end": {"type": "integer"},
    }
    settings: dict[str, Any] = {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": build_analysis(),
    }

    if dimension:
        settings["index"]["knn"] = True
        props["embedding"] = {
            "type": "knn_vector",
            "dimension": dimension,
            # lucene engine supports efficient pre-filtering during graph traversal,
            # which matters because every query here carries metadata filters.
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene",
                "parameters": {"ef_construction": 128, "m": 16},
            },
        }

    return {"settings": settings, "mappings": {"properties": props}}


def create_index(
    client: OpenSearch, name: str, *, dimension: int | None = None, recreate: bool = False
) -> None:
    if client.indices.exists(index=name):
        if not recreate:
            return
        client.indices.delete(index=name)
    client.indices.create(index=name, body=build_mapping(dimension))


def _actions(
    index: str, records: Iterable[SearchRecord], vectors: dict[str, list[float]] | None
) -> Iterator[dict[str, Any]]:
    for r in records:
        src = r.to_dict()
        if vectors is not None:
            v = vectors.get(r.record_id)
            if v is not None:
                src["embedding"] = v
        yield {"_index": index, "_id": r.record_id, "_source": src}


def index_records(
    client: OpenSearch,
    name: str,
    records: list[SearchRecord],
    *,
    vectors: dict[str, list[float]] | None = None,
    batch_size: int = 500,
    refresh: bool = True,
) -> tuple[int, list]:
    """Bulk-index records. Returns (succeeded, errors)."""
    ok, errors = bulk(
        client,
        _actions(name, records, vectors),
        chunk_size=batch_size,
        max_retries=3,
        initial_backoff=2,
        raise_on_error=False,
        request_timeout=120,
    )
    if refresh:
        client.indices.refresh(index=name)
    return ok, list(errors)


def index_stats(client: OpenSearch, name: str) -> dict[str, Any]:
    client.indices.refresh(index=name)
    count = client.count(index=name)["count"]
    by_type = client.search(
        index=name,
        body={"size": 0, "aggs": {"t": {"terms": {"field": "record_type", "size": 20}}}},
    )
    return {
        "documents": count,
        "by_record_type": {
            b["key"]: b["doc_count"] for b in by_type["aggregations"]["t"]["buckets"]
        },
    }
