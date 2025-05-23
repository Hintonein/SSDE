import os
import sys
import time
import multiprocessing
from copy import deepcopy
from datetime import datetime
import click

import numpy as np
from ssde import PDESymbolicSolver
from ssde.gradients import hessian, jacobian
from ssde.config import load_config
from ssde.utils import rect
from ssde.utils import safe_update_summary
from ssde.logeval import LogEval

def train_sympde(config):
    print("\n== TRAINING SEED {} START ============".format(config["experiment"]["seed"]))
    
    benchmark = config["task"]["dataset"]
    data = np.load(benchmark)
    test_dataset = [data['test_X'], data['test_y']]
    config["task"]["dataset"] = benchmark
    
    model = PDESymbolicSolver(deepcopy(config))
    start = time.time()
    result = model.train()

    test_X, test_y = test_dataset
    pred_y = result["program"].execute(test_X).reshape(-1,1)
    # R2 = 1 - np.sum((pred_y - test_y) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2)
    mrmse = (1/result["program"].r)-1
    result["t"] = time.time() - start
    result["MRMSE"] = mrmse
    result["MSE"] = np.mean((pred_y - test_y) ** 2)
    result.pop("program")

    save_path = model.config_experiment["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    print("== TRAINING SEED {} END ==============".format(config["experiment"]["seed"]))

    return result, summary_path

def print_summary(config, runs, messages):
    text = '\n== EXPERIMENT SETUP START ===========\n'
    text += 'Task type            : {}\n'.format(config["task"]["task_type"])
    if config["task"]["task_type"] == "regression":
        text += 'Dataset              : {}\n'.format(config["task"]["dataset"])
    elif config["task"]["task_type"] == "control":
        text += 'Environment          : {}\n'.format(config["task"]["env"])
    text += 'Starting seed        : {}\n'.format(config["experiment"]["seed"])
    text += 'Runs                 : {}\n'.format(runs)
    if len(messages) > 0:
        text += 'Additional context   :\n'
        for message in messages:
            text += "      {}\n".format(message)
    text += '== EXPERIMENT SETUP END ============='
    print(text)

@click.command()
@click.argument('config_template', default="")
@click.option('--runs', '--r', default=1, type=int, help="Number of independent runs with different seeds")
@click.option('--n_cores_task', '--n', default=1, help="Number of cores to spread out across tasks")
@click.option('--seed', '--s', default=None, type=int, help="Starting seed (overwrites seed in config), incremented for each independent run")
@click.option('--benchmark', '--b', default=None, type=str, help="Name of benchmark")
@click.option('--exp_name', default=None, type=str, help="Name of experiment to manually generate log path")
def main(config_template, runs, n_cores_task, seed, benchmark, exp_name):
    """Runs Sympde in parallel across multiple seeds using multiprocessing."""

    messages = []

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)

    # Overwrite named benchmark (for tasks that support them)
    # task_type = config["task"]["task_type"]
    # Read the X_input, y_input from data
    if benchmark is not None:
        # For regression, --b overwrites config["task"]["dataset"]
        config["task"]["dataset"] = benchmark
       
    # Update save dir if provided
    if exp_name is not None:
        config["experiment"]["exp_name"] = exp_name

    # Overwrite config seed, if specified
    if seed is not None:
        if config["experiment"]["seed"] is not None:
            messages.append(
                "INFO: Replacing config seed {} with command-line seed {}.".format(
                    config["experiment"]["seed"], seed))
        config["experiment"]["seed"] = seed

    # Save starting seed and run command
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    config["experiment"]["cmd"] = " ".join(sys.argv)

    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp

    # Fix incompatible configurations
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > runs:
        messages.append(
                "INFO: Setting 'n_cores_task' to {} because there are only {} runs.".format(
                    runs, runs))
        n_cores_task = runs
    if config["training"]["verbose"] and n_cores_task > 1:
        messages.append(
                "INFO: Setting 'verbose' to False for parallelized run.")
        config["training"]["verbose"] = False
    if config["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
        messages.append(
                "INFO: Setting 'n_cores_batch' to 1 to avoid nested child processes.")
        config["training"]["n_cores_batch"] = 1
    if config["gp_meld"]["run_gp_meld"] and n_cores_task > 1 and runs > 1:
        messages.append(
                "INFO: Setting 'parallel_eval' to 'False' as we are already parallelizing.")
        config["gp_meld"]["parallel_eval"] = False


    # Start training
    print_summary(config, runs, messages)

    # Generate configs (with incremented seeds) for each run
    configs = [deepcopy(config) for _ in range(runs)]
    for i, config in enumerate(configs):
        config["experiment"]["seed"] += i

    # Farm out the work
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for i, (result, summary_path) in enumerate(pool.imap_unordered(train_sympde, configs)):
            if not safe_update_summary(summary_path, result):
                print("Warning: Could not update summary stats at {}".format(summary_path))
            print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))
    else:
        for i, config in enumerate(configs):
            result, summary_path = train_sympde(config)
            if not safe_update_summary(summary_path, result):
                print("Warning: Could not update summary stats at {}".format(summary_path))
            print("INFO: Completed run {} of {} in {:.0f} s".format(i + 1, runs, result["t"]))

    # Evaluate the log files
    print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["logging"]["hof"] is not None and config["logging"]["hof"] > 0,
        show_pf=config["logging"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"])
    print("== POST-PROCESS END ===================")


if __name__ == "__main__":
    main()