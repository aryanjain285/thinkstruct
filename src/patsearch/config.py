"""Central paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — avoids a dependency and never overwrites a real env var."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "patent_data" / "data" / "patent_data_small"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

for _d in (PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# OpenSearch — local zip install, security plugin disabled, plain HTTP on loopback.
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "http://localhost:9200")
INDEX_NAME = os.environ.get("PATSEARCH_INDEX", "patents")

# Embedder spec — see patsearch.embeddings.service.PRESETS
EMBEDDER = os.environ.get("PATSEARCH_EMBEDDER", "minilm")

# Description paragraphs shorter than this are formulae/figure-reference noise.
# Measured: 4.4% of non-blank paragraphs fall under 100 chars.
MIN_PARAGRAPH_CHARS = 100
