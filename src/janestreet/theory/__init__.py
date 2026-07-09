"""Theoretical building blocks — the heavy mathematics, kept out of the models.

This layer holds the reusable machinery so the model layer
(:mod:`janestreet.models`) stays lean: a model *composes* these pieces rather
than reimplementing them. Nothing here depends on ``janestreet.models`` — the
dependency arrow points one way, models -> theory.

Contents
--------
- :mod:`~janestreet.theory.signatures` — path-signature transforms (Chen
  iteration, log-signatures over the free Lie algebra, Volterra / Hurst
  reweighting, minimal Lyndon basis via iisignature) and the trainable
  :class:`~janestreet.theory.signatures.SignatureBlock`.
- :mod:`~janestreet.theory.torch_utils` — shared torch primitives
  (device selection, flat <-> (time, feature) sequence reshapes).
"""

from __future__ import annotations

from janestreet.theory.signatures import (
    SignatureBlock,
    compute_log_signature,
    compute_log_signature_minimal,
    compute_signature,
    compute_volterra_log_signature,
    compute_volterra_signature,
)
from janestreet.theory.torch_utils import (
    auto_device,
    reshape_flat_to_sequence,
    reshape_sequence_to_flat,
)

__all__ = [
    "SignatureBlock",
    "auto_device",
    "compute_log_signature",
    "compute_log_signature_minimal",
    "compute_signature",
    "compute_volterra_log_signature",
    "compute_volterra_signature",
    "reshape_flat_to_sequence",
    "reshape_sequence_to_flat",
]
