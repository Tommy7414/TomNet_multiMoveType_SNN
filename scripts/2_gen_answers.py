# scripts/gen_answers.py
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from mazesim.config import SimConfig
from mazesim.dataset import build_all_csv
"""
Pipeline 2
Generate option for both human and model to train and test
Make sure the hyperparameters in SimConfig are consistent with those in gen_trails.py
"""

def main():
    config = SimConfig(
        sim_name = "5_24",
        n_agents = 5,
        maze_size = 24,
        n_steps = 10,
        move_types = [
            "rulemap",
            "random",
            "intermediate_case1",
            "intermediate_case2",
            "logic",
        ],
        n_trials_per_type = {
            "rulemap": 10,
            "random": 10,
            "intermediate_case1": 10,
            "intermediate_case2": 10,
            "logic": 10,
        },
        project_root = Path("/Users/Jer_ry/Desktop/script_tom"),
    )

    build_all_csv(config)


if __name__ == "__main__":
    main()
