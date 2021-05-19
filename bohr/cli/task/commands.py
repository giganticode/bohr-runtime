import logging

import click

from bohr import api
from bohr.collection.artifacts import artifact_map
from bohr.config.pathconfig import PathConfig
from bohr.datamodel.bohrrepo import load_bohr_repo
from bohr.datamodel.dataset import get_all_linked_datasets, train_and_test
from bohr.datamodel.heuristic import get_heuristic_module_list
from bohr.datamodel.task import Task
from bohr.fs import find_project_root

logger = logging.getLogger(__name__)


@click.group()
def task():
    pass


@task.command()
@click.argument("task", type=str)
@click.argument("dataset", type=str)
@click.option("--repro", is_flag=True)
def add_dataset(task: str, dataset: str, repro: bool) -> None:
    bohr_repo = load_bohr_repo()
    if task not in bohr_repo.tasks:
        logger.error(f"Task {task} is not defined")
        exit(404)
    if dataset not in bohr_repo.datasets:
        logger.error(f"Dataset {dataset} is not defined")
        exit(404)
    dataset = api.add_dataset(
        bohr_repo.tasks[task], bohr_repo.datasets[dataset], bohr_repo
    )
    print(f"Dataset {dataset} is added to the task {task}.")
    if repro:
        logger.info("Re-running the pipeline ...")
        api.repro(task, bohr_repo=bohr_repo)


@task.command()
@click.argument("name", type=str)
@click.option("-t", "--artifact", required=True)
@click.option("-l", "--labels", type=str, required=True)
@click.option("-c", "--label-column", type=str, required=True)
@click.option("-a", "--authors", type=str, required=False)
@click.option("-d", "--description", type=str, required=False)
@click.option("-A", "--use-all-datasets", is_flag=True)
@click.option("--repro", is_flag=True)
@click.option("--force", is_flag=True)
def add(
    name: str,
    artifact: str,
    labels: str,
    label_column: str,
    authors: str,
    description: str,
    use_all_datasets: bool,
    repro: bool,
    force: bool,
) -> None:
    project_root = find_project_root()
    bohr_repo = load_bohr_repo(project_root)
    path_config = PathConfig.load(project_root)
    if name in bohr_repo.tasks and not force:
        logger.error(f"Task {name} is already defined")
        exit(400)
    try:
        artifact_type = artifact_map[artifact]
    except KeyError:
        logger.error(f"Artifact not found: {artifact}")
        exit(404)
    label_list = list(map(lambda s: s.strip(), labels.split(",")))
    if not use_all_datasets:
        train_datasets, test_datasets = {}, {}
    else:
        all_datasets = {
            n: d
            for n, d in bohr_repo.datasets.items()
            if d.artifact_type == artifact_type
        }
        train_datasets, test_datasets = train_and_test(all_datasets, label_column)
    heuristic_groups = get_heuristic_module_list(artifact_type, path_config.heuristics)
    task = Task(
        name,
        authors,
        description,
        artifact_type,
        label_list,
        train_datasets,
        test_datasets,
        label_column,
        heuristic_groups,
    )
    bohr_repo.tasks[name] = task
    bohr_repo.dump(project_root)
    if repro:
        logger.info("Re-running the pipeline ...")
        api.repro(name, bohr_repo=bohr_repo)
