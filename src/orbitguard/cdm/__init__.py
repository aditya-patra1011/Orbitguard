"""Collision risk modelling on Conjunction Data Messages (ESA Collision Avoidance Challenge)."""

from pathlib import Path

# Defined here (not in data.py) so the CLI can use it without importing pandas.
DEFAULT_DATA_DIR = Path(
    "data/kelvins/extracted/Collision Avoidance Challenge - Dataset/kelvins_competition_data"
)
