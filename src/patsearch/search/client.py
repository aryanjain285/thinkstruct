"""OpenSearch connection handling."""
from __future__ import annotations

import time
from urllib.parse import urlparse

from opensearchpy import OpenSearch

from patsearch.config import OPENSEARCH_HOST


class OpenSearchUnavailable(RuntimeError):
    pass


def get_client(host: str = OPENSEARCH_HOST, *, timeout: int = 30) -> OpenSearch:
    u = urlparse(host)
    return OpenSearch(
        hosts=[{"host": u.hostname or "localhost", "port": u.port or 9200}],
        use_ssl=(u.scheme == "https"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=timeout,
        max_retries=3,
        retry_on_timeout=True,
    )


def wait_for_health(client: OpenSearch, *, timeout_s: int = 60, status: str = "yellow") -> dict:
    """Block until the cluster reports at least `status`, else raise."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            return client.cluster.health(wait_for_status=status, request_timeout=5)
        except Exception as exc:  # connection refused while the node boots
            last = exc
            time.sleep(1)
    raise OpenSearchUnavailable(
        f"OpenSearch did not reach '{status}' within {timeout_s}s. Last error: {last}"
    )
