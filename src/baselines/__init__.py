"""Baseline vulnerability prioritization methods for CyberGuard AI."""

from .cvss_baseline import cvss_only_baseline
from .epss_baseline import epss_only_baseline
from .kev_baseline import kev_first_baseline
from .weighted_baseline import weighted_baseline

__all__ = [
    "cvss_only_baseline",
    "epss_only_baseline",
    "kev_first_baseline",
    "weighted_baseline",
]
