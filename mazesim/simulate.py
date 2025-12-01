# simulate.py
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# simulate.py
from .env import set_maze_env, get_env, world_to_disk
from .dynamics import (
    generate_social_score,
    inputs,
    move_rulemap,
    move_random,
    move_intermediate,
    make_role_labels,
    move_logic,   
    overlay_agents_on_maze  
)
from .config import SimConfig

SIM_ROOT: Optional[Path] = None


def simulation_data_dynamic(
    fname_csv: str = "dS103_test",
    RANDOM_NUM_GOALS: bool = False,
    VERSION_NAME: str = "sim1",
    STEP_TOTAL: int = 20,
    AGENT_NAME: str = "dS_test",
    N_AGENTS: int = 3,
    SET_UP_MAZE_TOTAL: int = 1000,
    PRINT_FINAL_MAZE: bool = False,
    move_type: str = "rulemap",
    INNER_SIZE: Optional[int] = None,
):
    """
    respond to generate non-logic type trail txt:
    - support move_type:
        'rulemap', 'random',
        'intermediate', 'intermediate_case1', 'intermediate_case2'
    - do not handle 'logic'; logic is handled by run_logic_simulation.
    """
    global SIM_ROOT
    if SIM_ROOT is None:
        raise RuntimeError("SIM_ROOT has not been set. Please set it before calling simulation_data_dynamic.")

    raw_move_type = move_type.lower().strip()

    if raw_move_type == "logic":
        raise ValueError("simulation_data_dynamic does not handle 'logic'; please use run_logic_simulation(...) instead.")

    base_move_type = raw_move_type
    fixed_case_type: Optional[int] = None

    if raw_move_type in ("intermediate_case1", "intermediate_case2"):
        base_move_type = "intermediate"
        fixed_case_type = 1 if raw_move_type.endswith("case1") else 2

    if base_move_type not in ("rulemap", "random", "intermediate"):
        raise ValueError(f"Unknown move_type: {move_type} (base={base_move_type})")

    # ---- ENV ----
    if INNER_SIZE is not None:
        set_maze_env(inner_size=int(INNER_SIZE), world_min=7)
    env = get_env()  # get the current environment from env.py

    # --- I/O root: each "external mode name" has its own folder ---
    # For example:
    #   SIM_ROOT / 'rulemap'
    #   SIM_ROOT / 'random'
    #   SIM_ROOT / 'intermediate'
    #   SIM_ROOT / 'intermediate_case1'
    #   SIM_ROOT / 'intermediate_case2'
    dir_txt_output: Path = SIM_ROOT / raw_move_type
    dir_txt_output.mkdir(parents=True, exist_ok=True)
    file_csv_summary: Path = dir_txt_output / "summary.csv"

    # --- social score matrix (social_score) ---
    dir_csv = SIM_ROOT / "log" / "social_score"
    csv_file = dir_csv / f"{fname_csv}.csv"

    if not csv_file.exists():
        dir_csv.mkdir(parents=True, exist_ok=True)
        generate_social_score(out_dir=dir_csv, fname=fname_csv, n_agents=N_AGENTS)

    agents_arr = pd.read_csv(csv_file, index_col=0).to_numpy(float)

    # --- basic parameters and role labels ---
    n_chosen_agents = int(N_AGENTS)
    n_move_agents = n_chosen_agents - 1
    chosen_agents_label = make_role_labels(base_move_type, n_chosen_agents)

    social_reward = agents_arr.copy()
    fix_central_index = [
        i for i in range(n_chosen_agents)
        if sum(abs(social_reward[i])) == 0
    ]

    # --- generate mazes one by one ---
    maze_total = 0
    df_collect_summary = pd.DataFrame()

    while maze_total < SET_UP_MAZE_TOTAL:
        # 1) decide the center agent (the one with social force 0, otherwise round-robin)   
        if fix_central_index:
            center_agent = fix_central_index[maze_total % len(fix_central_index)]
        else:
            center_agent = maze_total % n_chosen_agents

        # 2) decide the case_type for this maze (only affects intermediate type)
        if base_move_type == "intermediate":
            if fixed_case_type is not None:
                case_type_maze = fixed_case_type  #
        else:
            case_type_maze = None  # not needed for non-intermediate

        # 3) randomly choose movable positions, excluding the center cell
        available = [
            (x, y)
            for x in range(env.move_min, env.move_max + 1)
            for y in range(env.move_min, env.move_max + 1)
            if (x, y) != env.center
        ]
        chosen = random.sample(available, n_move_agents)
        x_random = [p[0] for p in chosen]
        y_random = [p[1] for p in chosen]

        # Insert the center agent at the center_agent position
        cx, cy = env.center
        x_random.insert(center_agent, cx)
        y_random.insert(center_agent, cy)

        # Initial positions (world coordinates)
        init_world = np.c_[x_random, y_random].astype(float)
        env.assert_world(init_world, "init start")

        # 4) maze ASCII initial image
        maze_lines = overlay_agents_on_maze(env, init_world, chosen_agents_label)

        # ===== simulation =====
        agent_place = np.zeros((n_chosen_agents, 2), dtype=float)
        agent_place, _, _ = inputs(agent_place, x_random, y_random, social_reward, n_chosen_agents)

        all_agent_place = [agent_place.copy()]
        same_counter = wobble_counter = 0
        prev_prev = None
        step = 0

        # intermediate specific state
        if base_move_type == "intermediate":
            if n_chosen_agents == 3:
                primary_idx_maze = tuple(sorted(random.sample(range(3), 2)))
            elif n_chosen_agents == 5:
                primary_idx_maze = (0, 1) # 5人固定前兩隻為主
            else:
                raise ValueError("Intermediate 只支援 N=3 或 N=5")


        # ---- main loop: update positions each step ----
        while step < STEP_TOTAL:
            if base_move_type == "random":
                agent_place_new = move_random(
                    agent_place,
                    n_chosen_agents,
                    mu=n_chosen_agents,
                    env=env,
                )
            elif base_move_type == "rulemap":
                agent_place, _, force = inputs(
                    agent_place, x_random, y_random, social_reward, n_chosen_agents
                )
                agent_place_new = move_rulemap(env, agent_place, force, n_chosen_agents)
            elif base_move_type == "intermediate":
                agent_place_new = move_intermediate(
                    env, 
                    agent_place, 
                    case_type_maze,
                    n_chosen_agents
                )
            else:
                raise ValueError(f"unknown base_move_type: {base_move_type}")

            agent_place_new = env.clamp_world(agent_place_new)
            all_agent_place.append(agent_place_new.copy())
            step += 1

            # ---- stopping conditions ----
            same_counter = same_counter + 1 if np.array_equal(agent_place_new, agent_place) else 0
            wobble_counter = (
                wobble_counter + 1
                if (prev_prev is not None and np.array_equal(agent_place_new, prev_prev))
                else 0
            )

            stop_same = 4 if base_move_type == "random" else 2
            stop_wobble = 4 if base_move_type == "random" else 2

            if same_counter >= stop_same or wobble_counter >= stop_wobble:
                break

            prev_prev = agent_place.copy()
            agent_place = agent_place_new.copy()

        true_step = step
        all_agent_place = np.vstack(all_agent_place)  # (T+1, N, 2) 世界
        env.assert_world(all_agent_place, "before write")
        all_agent_place_disk = world_to_disk(all_agent_place)  # convert to 1..inner_size

        # Skip too short trajectories
        if all_agent_place_disk.shape[0] <= 1:
            maze_total += 1
            continue

        # ===== statistics =====
        maze_final_stat = np.zeros((n_chosen_agents, n_chosen_agents))
        for i in range(-n_chosen_agents, 0):
            for j in range(-n_chosen_agents, 0):
                if (
                    (int(all_agent_place[i][1]) - int(all_agent_place[j][1])) ** 2
                    + (int(all_agent_place[i][0]) - int(all_agent_place[j][0])) ** 2
                ) <= 2:
                    maze_final_stat[i][j] = 1

        pair_cnt = n_chosen_agents * (n_chosen_agents - 1) // 2
        maze_final_stat_bin = np.zeros(pair_cnt)
        l = 0
        for i in range(n_chosen_agents):
            for j in range(i):
                maze_final_stat_bin[l] = maze_final_stat[j][i]
                l += 1

        maze_final_stat_bin = maze_final_stat_bin.astype(int).astype(str)
        maze_final_stat_bin = "".join(maze_final_stat_bin)
        maze_final_stat_dec = int(maze_final_stat_bin, 2)

        df_summary = pd.DataFrame(
            {
                "maze": [maze_total + 1],
                "steps": [true_step],
                "maze_final_binary": [maze_final_stat_bin],
                "maze_final_decimal": [maze_final_stat_dec],
                "center_agent": [chosen_agents_label[center_agent]],
                "flag_1": [0],
                "flag_2": [0],
                "case_type": [case_type_maze if base_move_type == "intermediate" else None],
                "move_dominant_agents": [
                    primary_idx_maze if base_move_type == "intermediate" else None
                ],
                "move_type": [raw_move_type],
            }
        )
        df_collect_summary = pd.concat([df_collect_summary, df_summary], ignore_index=True)

        save_dir = dir_txt_output
        save_dir.mkdir(parents=True, exist_ok=True)

        output_file = save_dir / f"{AGENT_NAME}_{maze_total + 1}.txt"
        with open(output_file, "w", encoding="utf-8") as text_file:
            text_file.write("Maze:\n")
            for row in maze_lines:
                text_file.write(row + "\n")
            np.set_printoptions(threshold=np.inf, linewidth=200)
            agent_place_str = np.array2string(all_agent_place_disk, separator=" ")
            text_file.write(agent_place_str + "\n")

        if PRINT_FINAL_MAZE:
            final_xy = all_agent_place[-n_chosen_agents:, :]  # 世界
            final_lines = overlay_agents_on_maze(env, final_xy, chosen_agents_label)
            output_file_final = save_dir / f"{AGENT_NAME}_{maze_total + 1}FINAL.txt"
            with open(output_file_final, "w", encoding="utf-8") as text_file:
                text_file.write("Final_Maze:\n")
                for row in final_lines:
                    text_file.write(row + "\n")
                text_file.write("\n")

        maze_total += 1

        if maze_total == SET_UP_MAZE_TOTAL:
            df_collect_summary.to_csv(file_csv_summary, index=False)

def _run_single_logic_traj(case_type: int,
                           trial_id: int,
                           *,
                           agent_name: str,
                           n_agents: int,
                           total_steps: int,
                           INNER_SIZE: int):
    """Generate a single logic trial, output simName_case{case_type}_{trial_id}.txt"""
    global SIM_ROOT
    if SIM_ROOT is None:
        raise RuntimeError("SIM_ROOT is not set, please specify config.sim_root in gen_trails")

    # Set environment
    set_maze_env(inner_size=int(INNER_SIZE), world_min=7)
    env = get_env()

    out_dir = SIM_ROOT / 'logic'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{agent_name}_case{case_type}_{trial_id}.txt'

    N = int(n_agents)

    init = np.column_stack([
        np.random.randint(env.move_min, env.move_max + 1, size=N),
        np.random.randint(env.move_min, env.move_max + 1, size=N),
    ]).astype(float)

    init[0] = np.array(list(env.center), float)
    env.assert_world(init, 'logic init')

    # Determine direc_case (the actual index passed to move_logic)
    if N == 3:
        # 3-agent version: supports 0..11 (12 types), cycle through case_type
        direc_case = case_type % 12
    elif N == 5:
        # 5-agent version: supports 1..80, mod case_type if exceeded
        direc_case = ((case_type - 1) % 80) + 1
    else:
        raise ValueError("logic only supports n_agents ∈ {3,5}")

    # main loop
    trajectory = [init.copy()]
    cur = init.copy()
    prev_prev = None

    for _ in range(total_steps):
        nxt = move_logic(env, cur, direc_case)  # ← 從 dynamics 匯入
        if np.array_equal(nxt, cur) or (prev_prev is not None and np.array_equal(nxt, prev_prev)):
            break
        trajectory.append(nxt.copy())
        prev_prev = cur.copy()
        cur = nxt.copy()

    traj_world = np.stack(trajectory, axis=0)
    env.assert_world(traj_world, 'logic traj')

    # Draw ASCII maze (only use initial positions)
    role_letters = ['S'] + [chr(ord('A') + i) for i in range(max(0, N - 1))]
    maze_lines = overlay_agents_on_maze(env, init, role_letters[:N])

    # world → disk(1..inner)
    traj_disk = world_to_disk(traj_world)
    if traj_disk.shape[0] <= 1:
        return  # 太短就不存

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('Maze:\n')
        for row in maze_lines:
            f.write(row + '\n')
        np.set_printoptions(threshold=np.inf, linewidth=200)
        f.write(np.array2string(traj_disk.reshape(-1, 2), separator=' ') + '\n')

def run_logic_simulation(config: SimConfig, n_trials: int):
    """
    Generate logic data for config.sim_name:
    - Filename format: <sim_name>_case{case_type}_{trial_id}.txt
    - case_type cycles through trial_id:
        N=3 → 0..11
        N=5 → 1..80
    """
    agent_name = config.sim_name
    n_agents   = config.n_agents
    INNER_SIZE = config.maze_size
    total_steps = config.n_steps

    if n_agents == 3:
        max_case = 12
    elif n_agents == 5:
        max_case = 80
    else:
        raise ValueError("logic only supports n_agents ∈ {3,5}")
    for trial_id in range(1, n_trials + 1):
        case_type = ((trial_id - 1) % max_case) + 1
        _run_single_logic_traj(
            case_type=case_type,
            trial_id=trial_id,
            agent_name=agent_name,
            n_agents=n_agents,
            total_steps=total_steps,
            INNER_SIZE=INNER_SIZE,
        )

def gen_trails(config: SimConfig):
    """
    Entry function: run corresponding simulations based on config.move_types.
    - rulemap / random / intermediate / intermediate_case1 / intermediate_case2 -> simulation_data_dynamic
    - logic -> run_logic_simulation
    """
    global SIM_ROOT
    SIM_ROOT = config.data_root  

    for move_type in config.move_types:
        n_trials = config.n_trials_per_type[move_type]

        if move_type == "logic":
            run_logic_simulation(config, n_trials=n_trials)
            continue
        

        simulation_data_dynamic(
            fname_csv=f"{config.sim_name}_{move_type}",
            RANDOM_NUM_GOALS=False,
            VERSION_NAME=config.sim_name,
            STEP_TOTAL=config.n_steps,
            AGENT_NAME=config.sim_name,
            N_AGENTS=config.n_agents,
            SET_UP_MAZE_TOTAL=n_trials,
            PRINT_FINAL_MAZE=False,
            move_type=move_type,
            INNER_SIZE=config.maze_size,
        )
