from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import pandas as pd

class CriterionDirection(Enum):
    MAXIMIZE = "Maximize"
    MINIMIZE = "Minimize"

@dataclass
class Criterion:
    name: str
    weight: float
    direction: CriterionDirection = CriterionDirection.MAXIMIZE

@dataclass
class Alternative:
    name: str
    scores: Dict[str, float] = field(default_factory=dict)

class DecisionMatrix:
    def __init__(self, criteria: List[Criterion], alternatives: List[Alternative]):
        self.criteria = criteria
        self.alternatives = alternatives
        self._validate()

    def _validate(self):
        crit_names = {c.name for c in self.criteria}
        for alt in self.alternatives:
            if not all(c in alt.scores for c in crit_names):
                missing = crit_names - set(alt.scores.keys())
                raise ValueError(f"Alternative '{alt.name}' is missing scores for criteria: {missing}")

    def to_dataframe(self) -> pd.DataFrame:
        data = {}
        for alt in self.alternatives:
            data[alt.name] = alt.scores
        return pd.DataFrame.from_dict(data, orient='index')

    @property
    def criteria_weights(self) -> Dict[str, float]:
        return {c.name: c.weight for c in self.criteria}

    @property
    def criteria_directions(self) -> Dict[str, CriterionDirection]:
        return {c.name: c.direction for c in self.criteria}
