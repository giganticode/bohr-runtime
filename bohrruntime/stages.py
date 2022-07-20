from typing import List, Optional

import numpy as np
import pandas as pd
from bohrapi.core import Heuristic
from bohrlabels.core import Label, OneOrManyLabels
from tqdm import tqdm

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment, SynteticExperiment
from bohrruntime.datamodel.model import HeuristicOutputs
from bohrruntime.datamodel.task import Task
from bohrruntime.datasource import (
    load_local_dataset,
    query_dataset_from_explorer,
    save_dataset,
)
from bohrruntime.heuristicuri import HeuristicURI
from bohrruntime.storageengine import StorageEngine
from bohrruntime.tasktypes.labeling.lfs import HeuristicApplier


def load_dataset(dataset: Dataset, storage_engine: StorageEngine) -> None:
    print(f"Loading dataset: {dataset.id}")
    save_to_fs = storage_engine.cached_datasets_subfs()
    if dataset.path is not None:
        artifacts, metadata = load_local_dataset(dataset)
    else:
        artifacts, metadata = query_dataset_from_explorer(dataset)
    save_dataset(artifacts, metadata, save_to_fs)
    print(f"Dataset loaded: {dataset.id}, and save to {save_to_fs}")


def apply_heuristics_to_dataset(
    applier: HeuristicApplier,
    heuristic_uri: HeuristicURI,
    dataset: Dataset,
    storage_engine: StorageEngine,
) -> None:
    heuristics = storage_engine.get_heuristic_loader().load_heuristics(heuristic_uri)
    if not heuristics:
        raise AssertionError(f"No heuristics at {heuristic_uri}")
    artifacts = dataset.load_artifacts(storage_engine.cached_datasets_subfs())
    heuristics_output = applier.apply(heuristics, artifacts)
    storage_engine.save_single_heuristic_outputs(
        heuristics_output, dataset, heuristic_uri
    )


def combine_applied_heuristics(
    exp: Experiment, dataset: Dataset, storage_engine: StorageEngine
) -> None:
    matrix_list = []
    for heuristic_module_path in storage_engine.get_heuristic_module_paths(exp):
        matrix = storage_engine.load_single_heuristic_output(
            dataset, heuristic_module_path
        )
        matrix_list.append(matrix)
    combined_output = exp.task.combine_heuristics(matrix_list)
    storage_engine.save_heuristic_outputs(combined_output, exp, dataset)


def calculate_experiment_metrics(
    exp: Experiment, dataset: Dataset, storage_engine: StorageEngine
) -> None:
    cached_datasets_subfs = storage_engine.cached_datasets_subfs()
    ground_truth_labels = exp.task.load_ground_truth_labels(
        dataset, cached_datasets_subfs
    )
    if isinstance(exp, SynteticExperiment):
        heuristic_outputs = HeuristicOutputs(
            pd.DataFrame(
                np.zeros(
                    (dataset.get_n_datapoints(cached_datasets_subfs), 0), dtype=np.int32
                )
            )
        )
    else:
        heuristic_outputs = storage_engine.load_heuristic_outputs(exp, dataset)
    heuristic_output_metrics = exp.task.calculate_heuristic_output_metrics(
        heuristic_outputs, ground_truth_labels
    )
    model = storage_engine.load_model(exp)

    if ground_truth_labels is not None:
        model_metrics = model.calculate_model_metrics(
            heuristic_outputs, ground_truth_labels
        )
    else:
        model_metrics = {}

    metrics = {**heuristic_output_metrics, **model_metrics}

    storage_engine.save_experiment_metrics(metrics, exp, dataset)

    exp.do_analysis(
        storage_engine.exp_dataset_fs(exp, dataset),
        heuristic_outputs,
        ground_truth_labels,
    )


def train_model(exp: Experiment, storage_engine: StorageEngine):
    heuristics_output = storage_engine.load_heuristic_outputs(exp, exp.train_dataset)
    label_model_trainer = exp.task.get_model_trainer(storage_engine.fs)
    model = label_model_trainer.train(heuristics_output)
    storage_engine.save_model(model, exp)


def prepare_dataset(exp: Experiment, dataset: Dataset, storage_engine: StorageEngine):
    heuristic_outputs = storage_engine.load_heuristic_outputs(exp, dataset)
    model = storage_engine.load_model(exp)
    fuzzy_labels = model.predict(heuristic_outputs)
    prepared_dataset = exp.task.get_preparator().prepare(
        dataset, exp.task, fuzzy_labels, storage_engine.cached_datasets_subfs()
    )
    storage_engine.save_prepared_dataset(prepared_dataset, exp, dataset)


def calculate_single_heuristic_metrics(
    task: Task, storage_engine: StorageEngine
) -> None:
    heuristic_loader = storage_engine.get_heuristic_loader()
    for dataset in tqdm(task.test_datasets, desc="Calculating metrics for dataset: "):
        label_series = task.load_ground_truth_labels(
            dataset, storage_engine.cached_datasets_subfs()
        )
        heuristic_groups: List[HeuristicURI] = heuristic_loader.get_heuristic_uris(
            input_artifact_type=task.heuristic_input_artifact_type
        )
        for heuristic_uri in heuristic_groups:
            heuristics = heuristic_loader.load_heuristics(heuristic_uri)
            if not heuristics:
                raise AssertionError(
                    f"No heuristics at {heuristic_uri.absolute_path()}"
                )

            label_matrix = storage_engine.load_single_heuristic_output(
                dataset, heuristic_uri
            )

            metrics = task.calculate_heuristic_output_metrics(
                label_matrix, label_series
            )
            storage_engine.save_single_heuristic_metrics(
                metrics, task, dataset, heuristic_uri
            )
