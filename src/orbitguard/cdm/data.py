"""Load ESA Collision Avoidance Challenge CDMs and build the prediction task.

Each row is one Conjunction Data Message (CDM). An *event* (``event_id``) is the
time series of CDMs issued for one close approach, refined as TCA nears. The task,
as defined by the competition (Uriot et al., 2020):

- inputs: only CDMs issued at least 2 days before TCA (``time_to_tca >= 2``), because
  avoidance manoeuvres are planned at least 2 days ahead;
- target: the risk (log10 Pc) in the event's last CDM, issued within 1 day of TCA.

Events are eligible only if they have at least 2 CDMs, their first CDM is at least
2 days before TCA and their last CDM is within 1 day of TCA. The released test set
already follows these rules (inputs in ``test_data.csv``, targets in
``test_data_private.csv``); the training set is filtered here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(
    "data/kelvins/extracted/Collision Avoidance Challenge - Dataset/kelvins_competition_data"
)
INPUT_CUTOFF_DAYS = 2.0
TARGET_WINDOW_DAYS = 1.0
HIGH_RISK = -6.0  # log10 Pc; the competition's notification threshold


@dataclass
class Task:
    inputs: pd.DataFrame   # CDMs usable for prediction, sorted by event then time
    targets: pd.Series     # final risk per event_id

    @property
    def event_ids(self) -> pd.Index:
        return self.targets.index


def sort_cdms(cdms: pd.DataFrame) -> pd.DataFrame:
    """Chronological order within each event (time_to_tca counts down)."""
    return cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).reset_index(drop=True)


def build_training_task(cdms: pd.DataFrame) -> Task:
    """Apply the competition's eligibility rules to raw training CDMs."""
    cdms = sort_cdms(cdms)
    grouped = cdms.groupby("event_id")["time_to_tca"]
    first, last = grouped.first(), grouped.last()
    eligible = first.index[(first >= INPUT_CUTOFF_DAYS) & (last < TARGET_WINDOW_DAYS)]

    cdms = cdms[cdms.event_id.isin(eligible)]
    targets = cdms.groupby("event_id")["risk"].last().rename("final_risk")
    inputs = cdms[cdms.time_to_tca >= INPUT_CUTOFF_DAYS].reset_index(drop=True)
    # Rule i: at least one input CDM and one target CDM. Guaranteed by first >= 2 and last < 1.
    return Task(inputs, targets.loc[inputs.event_id.unique()])


def load_training_task(data_dir: Path = DEFAULT_DATA_DIR) -> Task:
    return build_training_task(pd.read_csv(data_dir / "train_data.csv"))


def load_test_task(data_dir: Path = DEFAULT_DATA_DIR) -> Task:
    inputs = sort_cdms(pd.read_csv(data_dir / "test_data.csv"))
    labels = pd.read_csv(data_dir / "test_data_private.csv", usecols=["event_id", "true_risk"])
    targets = labels.set_index("event_id")["true_risk"].rename("final_risk")
    return Task(inputs, targets.loc[inputs.event_id.unique()])
