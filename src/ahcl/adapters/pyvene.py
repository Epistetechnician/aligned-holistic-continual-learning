"""pyvene composable intervention adapter.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import InterventionBackend
from .common import optional_import, require_enabled


class PyveneInterventionAdapter(InterventionBackend):
    """Wrap a pyvene IntervenableModel without owning model construction."""

    def __init__(self, intervenable_model: Any):
        self.model = intervenable_model

    @classmethod
    def from_model(
        cls,
        model: Any,
        representations: list[dict[str, Any]],
        *,
        enabled: bool = False,
    ) -> "PyveneInterventionAdapter":
        require_enabled(enabled, "pyvene model intervention")
        module = optional_import("pyvene", "pyvene")
        config = module.IntervenableConfig(representations=representations)
        return cls(module.IntervenableModel(config, model))

    def replace(self, activation: Any, donor: Any, position: int) -> Any:
        """Apply one caller-defined source/base intervention.

        The exact representation path and batch indexing are part of the
        serialized pyvene configuration supplied to :meth:`from_model`.
        """

        return self.model(activation, donor)
