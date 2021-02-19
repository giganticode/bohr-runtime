import numpy as np
import pandas as pd
from snorkel.labeling.model import LabelModel

from bohr.config import Config, get_dataset_loader
from bohr.core import load_heuristics, to_labeling_functions


def label_dataset(
    task_name: str, dataset_name: str, config: Config, debug: bool = False
):
    task = config.tasks[task_name]
    dataset_loader = get_dataset_loader(dataset_name)
    df = dataset_loader.load(config.project_root)

    lines_train = np.load(
        config.paths.generated / task.name / "heuristic_matrix_train.pkl",
        allow_pickle=True,
    )

    print(lines_train.shape)
    print(df.shape)

    heuristics = load_heuristics(task.top_artifact, config)
    labeling_functions = to_labeling_functions(
        heuristics, dataset_loader.get_mapper(), task.labels
    )

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lines_train, n_epochs=100, log_freq=100, seed=123)

    labels, probs = label_model.predict(L=lines_train, return_probs=True)
    df_labeled = df.assign(bug=labels)

    df_probs = pd.DataFrame(probs, columns=["prob_bugless", "prob_bug"])
    df_labeled = pd.concat([df_labeled, df_probs], axis=1)

    if debug:
        df_lfs = pd.DataFrame(
            lines_train, columns=[lf.name for lf in labeling_functions]
        )
        for name, col in df_lfs.iteritems():
            df_lfs[name] = col.map({0: name, 1: name, -1: ""})
        col_lfs = df_lfs.apply(lambda c: ";".join([v for v in c if v]), axis=1)
        df_labeled["lfs"] = col_lfs

    labeled_data_path = config.paths.labeled_data
    if not labeled_data_path.exists():
        labeled_data_path.mkdir(parents=True)
    target_file = labeled_data_path / f"{task_name}.csv"
    df_labeled.to_csv(target_file, index=False)
    print(f"Labeled dataset has been written to {target_file}.")
