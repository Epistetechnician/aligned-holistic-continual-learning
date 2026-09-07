"""Neuronpedia HTTP/OpenAPI boundary with no client dependency.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

import json
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from typing import Any, Mapping


class NeuronpediaClient:
    """Call a caller-selected local Neuronpedia endpoint and build feature URLs."""

    def __init__(self, base_url: str = "http://127.0.0.1:5002", timeout: float = 30.0, *, enabled: bool = False):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.enabled = enabled

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        if not self.enabled:
            raise RuntimeError("Neuronpedia network access is closed until independent review")
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def feature_url(model: str, source: str, feature_index: int) -> str:
        if feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        return f"https://neuronpedia.org/{model}/{source}/{feature_index}"
