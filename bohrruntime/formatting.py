import pandas as pd
from tabulate import tabulate


def tabulate_artifacts(df: pd.DataFrame) -> None:
    pd.options.mode.chained_assignment = None
    df.loc[:, "message"] = df["message"].str.wrap(70)
    print(
        tabulate(
            df,
            headers=df.columns,
            tablefmt="fancy_grid",
        )
    )
