import numpy as np
import pandas as pd

from bohr.config import Config, get_dataset_loader
from bohr.core import load_heuristics, to_labeling_functions
from bohr.pipeline.core import label, train_lmodel


def label_dataset(
    task_name: str, dataset_name: str, config: Config, debug: bool = False
):
    task = config.tasks[task_name]
    dataset_loader = get_dataset_loader(dataset_name)

    lines_train = np.load(
        str(config.paths.generated / task.name / "heuristic_matrix_train.pkl"),
        allow_pickle=True,
    )

    heuristics = load_heuristics(task.top_artifact, config)
    labeling_functions = to_labeling_functions(
        heuristics, dataset_loader.get_mapper(), task.labels
    )

    label_model = train_lmodel(lines_train)
    df = dataset_loader.load(config.project_root)
    df_labeled = label(label_model, lines_train, df)

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
