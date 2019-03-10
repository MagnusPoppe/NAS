import pandas as pd

def score_changes_per_training_session(report_df: pd.DataFrame) -> pd.DataFrame:
    """ Requires two levels within your dataframe"""
    result = None
    if len(report_df.index.levels[0]) > 1:
        previous = None
        for i in reversed(range(len(report_df.index.levels[0]))):
            if not previous is None:
                result = previous - report_df.loc[report_df.index.levels[0][i]]
            else:
                previous = report_df.loc[report_df.index.levels[0][i]]
    return result

def find_multi_session_reports(all_reports: dict) -> [pd.DataFrame]:
    for name, versions in all_reports.items():
        for version, df in versions.items():
            if len(df.index.levels[0]) > 1:
                yield df

def split_data_into_sessions(df: pd.DataFrame) -> [pd.DataFrame]:
    frames = []
    for i in range(len(df.index.levels[0])):
        frames += [df.loc[df.index.levels[0][i]]]
    return frames

