from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class SimConfig:
    sim_name: str                  # eg "sim5_24_10"
    n_agents: int                  # 3/5
    maze_size: int                 # 12/24
    n_steps: int                   # steps per trial
    move_types: List[str]          # ["rulemap","random","logic",...]
    n_trials_per_type: Dict[str,int]  # {"rulemap": 200, ...}

    project_root: Path             # /Users/.../scripts

    @property
    def data_root(self) -> Path:
        return self.project_root / "data" / self.sim_name

    @property
    def plot_root(self) -> Path:
        # If plot_root needs to be split, you might need a way to specify it,
        # or hardcode it to a default value like "training_data"
        # or dynamically generate the path where plot_root is needed
        return self.project_root / "training_data" / self.sim_name 

    @property
    def csv_root(self) -> Path:
        return self.plot_root / "csv"