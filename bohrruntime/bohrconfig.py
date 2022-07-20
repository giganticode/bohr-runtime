import inspect
import types
from typing import Dict, Optional

from bohrapi import core as proxies
from bohrlabels.core import LabelSet
from frozendict import frozendict
from fs.base import FS
from fs.errors import NoSysPath

from bohrruntime import version
from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment
from bohrruntime.datamodel.task import Task
from bohrruntime.datamodel.workspace import Workspace
from bohrruntime.tasktypes.grouping.core import GroupingTask
from bohrruntime.tasktypes.grouping.dataset import GroupingDataset
from bohrruntime.tasktypes.labeling.core import LabelingTask
from bohrruntime.util.paths import create_fs


def convert_proxy_to_dataset(dataset_proxy: proxies.Dataset) -> Dataset:
    dataset_type = type(dataset_proxy).__name__
    m = {"Dataset": Dataset, "GroupingDataset": GroupingDataset}
    return m[dataset_type](
        dataset_proxy.id,
        dataset_proxy.heuristic_input_artifact_type,
        dataset_proxy.query,
        dataset_proxy.projection,
        dataset_proxy.n_datapoints,
        dataset_proxy.path,
    )


def convert_proxy_to_task(
    task_proxy: proxies.Task, dataset_proxies: Dict[str, Dataset]
) -> Task:
    task_type = type(task_proxy).__name__
    m = {"LabelingTask": LabelingTask, "GroupingTask": GroupingTask}
    if task_type not in m:
        raise ValueError(f"Invalid task type: {task_type}")
    args = [
        task_proxy.name,
        task_proxy.author,
        task_proxy.description,
        task_proxy.heuristic_input_artifact_type,
        frozendict(
            {dataset_proxies[k.id]: v for k, v in task_proxy.test_datasets.items()}
        ),
    ]
    if hasattr(task_proxy, "labels"):
        args.append(tuple(list(map(lambda l: LabelSet.of(l), task_proxy.labels))))
    if hasattr(task_proxy, "class_balance"):
        args.append(task_proxy.class_balance)
    return m[task_type](*args)


def convert_proxies_to_real_objects(proxy_workspace: proxies.Workspace) -> Workspace:
    """
    >>> from bohrruntime.testtools import get_stub_dataset, get_stub_task, get_stub_experiment
    >>> dataset = get_stub_dataset(name='dataset1')
    >>> task = get_stub_task(test_dataset=dataset)
    >>> experiment = get_stub_experiment(task)
    >>> workspace = Workspace("0.6.0", [experiment])
    >>> convert_proxies_to_real_objects(workspace)
    Workspace(bohr_runtime_version='0.6.0', experiments=[Experiment(name='stub-exp', task=LabelingTask(name='stub-task', author='stub-author', description='stub-description', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, test_datasets=frozendict.frozendict({Dataset(id='dataset1', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, query=None, projection=None, n_datapoints=None, path=None): None}), labels=(['NonBugFix'], ['BugFix']), class_balance=None), train_dataset=Dataset(id='stub-test-dataset', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, query=None, projection=None, n_datapoints=None, path=None), heuristics_classifier=None, extra_test_datasets=frozendict.frozendict({}))])
    """
    task_proxies: Dict[str, Task] = {}
    dataset_proxies: Dict[str, Dataset] = {}
    for proxy_experiment in proxy_workspace.experiments:
        proxy_train_dataset = proxy_experiment.train_dataset
        if proxy_train_dataset.id not in dataset_proxies:
            dataset_proxies[proxy_train_dataset.id] = convert_proxy_to_dataset(
                proxy_train_dataset
            )
        for proxy_extra_test_dataset in proxy_experiment.extra_test_datasets.keys():
            if proxy_extra_test_dataset.id not in dataset_proxies:
                dataset_proxies[proxy_extra_test_dataset.id] = convert_proxy_to_dataset(
                    proxy_extra_test_dataset
                )
        for proxy_test_dataset in proxy_experiment.task.test_datasets.keys():
            if proxy_test_dataset.id not in dataset_proxies:
                dataset_proxies[proxy_test_dataset.id] = convert_proxy_to_dataset(
                    proxy_test_dataset
                )
    experiments = []
    for proxy_experiment in proxy_workspace.experiments:
        proxy_task = proxy_experiment.task
        if proxy_task.name not in task_proxies:
            task_proxies[proxy_task.name] = convert_proxy_to_task(
                proxy_task, dataset_proxies
            )
        experiments.append(
            Experiment(
                proxy_experiment.name,
                task_proxies[proxy_experiment.task.name],
                dataset_proxies[proxy_experiment.train_dataset.id],
                proxy_experiment.heuristics_classifier,
                frozendict(
                    {
                        dataset_proxies[k.id]: v
                        for k, v in proxy_experiment.extra_test_datasets.items()
                    }
                ),
            )
        )

    return Workspace(proxy_workspace.bohr_runtime_version, experiments)


def load_workspace(fs: Optional[FS] = None) -> Workspace:
    """
    >>> from fs.memoryfs import MemoryFS
    >>> fs = MemoryFS()
    >>> fs.writetext('bohr.py', 'a=3')
    >>> load_workspace(fs)
    Traceback (most recent call last):
    ...
    ValueError: Object of type Workspace not found in bohr.py
    >>> fs.writetext('bohr.py', 'from bohrapi.core import Workspace; w=Workspace("0.7.0", [])')
    >>> load_workspace(fs)
    Workspace(bohr_runtime_version='0.7.0', experiments=[])
    """
    fs = fs or create_fs()
    try:
        file = fs.getsyspath("bohr.py")
        import importlib.util

        spec = importlib.util.spec_from_file_location("bohr.userconfig", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except NoSysPath:
        module = types.ModuleType("bohr.userconfig")
        text = fs.readtext("bohr.py")
        exec(text, module.__dict__)

    for name, obj in inspect.getmembers(module):
        if isinstance(obj, proxies.Workspace):
            workspace = obj

            version_installed = version()
            if str(workspace.bohr_runtime_version) != version_installed:
                raise EnvironmentError(
                    f"Version of bohr framework from config: {workspace.bohr_runtime_version}. "
                    f"Version of bohr installed: {version_installed}"
                )
            try:
                return convert_proxies_to_real_objects(workspace)
            except Exception as e:
                raise ValueError("Error parsing bohr.py. See the error above.") from e
    raise ValueError(
        f"Object of type {proxies.Workspace.__name__} not found in bohr.py"
    )
