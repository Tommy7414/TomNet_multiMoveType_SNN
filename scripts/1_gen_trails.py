# scripts/gen_trails.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from mazesim.config import SimConfig
from mazesim.simulate import gen_trails
"""
Pipeline 1:
Generate trails data under data/{sim_name}
You can set number of agents, maze size, number of steps, move types, and number of trials per type.
Make sure to adjust project_root to your own path
"""
def main():
    config = SimConfig(
        sim_name = "5_24",
        n_agents = 5,
        maze_size = 24,
        n_steps = 10,
        move_types = ["rulemap","random","intermediate_case1","intermediate_case2","logic"],
        n_trials_per_type = {
            "rulemap": 10,
            "random": 10,
            "intermediate_case1": 10,
            "intermediate_case2": 10,
            "logic": 10,
        },
    project_root = Path("/Users/Jer_ry/Desktop/script_tom"), # CHANGE THIS TO YOUR OWN PATH
    )
    gen_trails(config)

if __name__ == "__main__":
    main()
