import math
import os
import re
import warnings
from typing import List

import numpy as np
import pandas as pd

from src.experiments.print_results.TextParser import TextParser

from src.experiments.utils import get_paths, Keys, default_suffix, load_results

base_ = os.path.dirname(os.path.realpath(__file__))


def find_most_recent_file(experiment_name, dataset_name):
    suffices = ['']
    while len(suffices) > 0:
        path, path_txt = get_paths(experiment_name, dataset_name, suffix=suffices[0])
        if os.path.exists(path_txt):
            return path, path_txt
        else:
            suffices.pop(0)
    return None, None


def format_metric(value, std, round_value=3, round_std=2):
    """
    Format a metric value and its standard deviation for LaTeX. If one of the values is None, return "{missing}". The
    standard deviation is displayed in gray and in a smaller font. The value and std are rounded to 3 decimal places,
    except for time related metrics, which are rounded to 0 decimal places.
    Parameters
    ----------
    value:
        The value of the metric
    std:
        The standard deviation of the metric
    round_value:
        The number of decimal places to round the value to
    round_std:
        The number of decimal places to round the standard deviation to

    Returns
    -------
    str
        The formatted metric value and standard deviation
    """
    value = round(value, round_value)
    std = round(std, round_std)
    if round_std == 0 and not math.isnan(std):
        std = int(std)

    if value is None or std is None:
        return "{missing}"
    return f'\\num{{{value}}} {{\\tiny \\textcolor{{gray}}{{$\\pm$ \\num{{{std}}}}}}}'


def format_table(table, round_value=3, round_std=2):
    """
    Format a table for LaTeX. The standard deviation is displayed in gray and in a smaller font.

    Parameters
    ----------
    table: np.ndarray
        The table to format
    round_value: int, default=3
        The number of decimal places to round the value to
    round_std: int, default=2
        The number of decimal places to round the standard deviation to

    Returns
    -------
    np.ndarray
        The formatted table

    """
    f = np.vectorize(lambda x:
                     format_metric(x[0], x[1], round_value, round_std) if x[0] is not None and not math.isnan(x[0])
                     else "{missing}")
    return f(table).astype("object")


def rel_diff_metrics(scores: np.ndarray, base_config, metric):
    """
    Calculate the relative difference between the base configuration and the other configurations for all
    datasets. The relative difference is calculated as (value - base_value), except if the metric is time related, then
    it is calculated as (value - base_value) / base_value.
    Parameters
    ----------
    scores: np.ndarray
        2D array containing the scores for each dataset (rows) and configuration (columns)
    base_config: int
        Index of the base configuration
    metric: str
        The name of the metric to calculate the relative difference for

    Returns
    -------
    List[str]
        List of formatted relative differences and standard deviations for each configuration and dataset.
    """
    if "time" in metric.lower():
        time_related = True
    else:
        time_related = False
    rel_diff = np.empty(scores.shape[0:2])
    for i in range(scores.shape[0]):
        base_score = scores[i, base_config][0]
        if base_score is np.nan or base_score is None:
            rel_diff[i, :] = np.nan
            continue
        rel_diff[i, 0] = base_score
        for j in range(0, scores.shape[1]):
            if j == base_config:
                continue

            index = j if base_config == 0 else j + 1
            score = scores[i, j][0]
            if score is np.nan or score is None:
                rel_diff[i, index] = np.nan
                continue

            if time_related:
                rel_diff[i, index] = (score - base_score) / base_score * 100
            else:
                rel_diff[i, index] = score - base_score

    return rel_diff


def average_rel_diff_metrics(scores: np.ndarray, base_config, metric):
    """
    Calculate the average relative difference between the base configuration and the other configurations over all
    datasets. The relative difference is calculated as (value - base_value), except if the metric is time related, then
    it is calculated as (value - base_value) / base_value.
    Parameters
    ----------
    scores: np.ndarray
        2D array containing the scores for each dataset (rows) and configuration (columns)
    base_config: int
        Index of the base configuration
    metric: str
        The name of the metric to calculate the relative difference for

    Returns
    -------
    List[str]
        List of formatted average relative differences and standard deviations for each configuration. The average is
        over the datasets.
    """
    result = []
    rel_diff = rel_diff_metrics(scores, base_config, metric)

    # Per column average and std of rel_diff
    for j in range(0, scores.shape[1]):
        avg = np.nanmean(rel_diff[:, j])
        std = np.nanstd(rel_diff[:, j])
        result.append((avg, std))
    return np.array(result, dtype="f,f")


def best_in_bold(table, columns, nb_configs, nb_datasets, time_related=False):
    for i in range(nb_datasets):
        try:
            for j in range(len(columns)):
                str_table_values = table[i, j * nb_configs:(j + 1) * nb_configs]
                table_values = [float(re.findall(r"[-+]?\d*\.\d+|\d+", value)[0]) if value != '{missing}' else -1 for
                                value in str_table_values]
                # table_values = [float(value.split(" $\pm$ ")[0]) if value != '{missing}' else -1 for value in
                #                 str_table_values]
                if time_related or 'time' in columns[j]:
                    best_value = np.min(table_values)
                else:
                    best_value = np.max(table_values)
                for k, value in enumerate(table_values):
                    if value == best_value:
                        table[i, j * nb_configs + k] = f"\\bfseries {str_table_values[k]}"
        except (AttributeError, AttributeError):
            pass
    return table


def average_rmse(results, method, name):
    return results[method][name][Keys.average_rmse]


def std_rmse(results, method, name):
    return results[method][name][Keys.std_rmse]


def average_nrmse(results, method, name):
    return results[method][name][Keys.average_nrmse]


def std_nrmse(results, method, name):
    return results[method][name][Keys.std_nrmse]


all_metrics = {
    'average_rmse': average_rmse,
    'std_rmse': std_rmse,
    'average_nrmse': average_nrmse,
    'std_nrmse': std_nrmse,
}


def print_results_excel(results: dict, metric: str):
    """
    Print results in Excel format

    Parameters
    ----------
    results: dict
        Results dictionary
    metric: str
        Metric to print
    """
    # Start header with 1 empty cell
    header = "\t"
    values = f"{results['dataset']}\t"
    for method in results.keys():
        header += f"{method}\t"
        values += f"{results[method][metric]}\t"
    print(header)
    print(values)


def print_all_results_excel(datasets: List[str], metric: str,  experiment_name, present_substring: str = None,
                            absent_substring: str = None, suffix: str = default_suffix, from_text=True,
                            column_order=None):
    """
    Print results in Excel format

    Parameters
    ----------
    datasets
        List of datasets to print the results of
    metric: str
        Metric to print
    experiment_name: str
        The name of the experiments whose results are being printed
    present_substring: str
        Substring that the method name should contain
    absent_substring: str
        Substring that the method name should not contain
    suffix: str
        Suffix to add to the file name
    from_text: bool
        Whether to load the results from a txt file. If False, the results are loaded from a pickle file
    """
    with warnings.catch_warnings():
        values = get_results(datasets, metric, experiment_name, present_substring, absent_substring, suffix, from_text,
                             column_order)
        print(values.to_csv(sep='\t'))


def get_results(datasets, metric, experiment_name, present_substring: str = None,
                            absent_substring: str = None, suffix: str = default_suffix, from_text=True,
                            column_order=None) -> pd.DataFrame:
    warnings.simplefilter("ignore")
    values = pd.DataFrame(index=datasets, columns=[])
    for dataset in datasets:
        try:
            if from_text:
                results = load_results_txt(experiment_name, dataset, suffix=suffix)
            else:
                results = load_results(experiment_name, dataset, suffix=suffix)
        except FileNotFoundError:
            print(f"Results for {dataset} not found. Skipping.")
            continue
        for method in results.keys():
            if method == 'dataset' or (present_substring is not None and present_substring not in method) or (
                    absent_substring is not None and absent_substring in method):
                continue
            values.at[dataset, method] = results[method][metric]
    if column_order is not None:
        missing_columns = set(column_order) - set(values.columns)
        values[list(missing_columns)] = np.nan
        values = values[column_order]
    return values


def load_results_txt(experiment_name, dataset_name, suffix=None):
    _, path_txt = get_paths(experiment_name, dataset_name, suffix=suffix)
    return parse_txt_file(path_txt)


def parse_txt_file(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()
        parser = TextParser(lines, length_indentation=5, current_indentation=0)
        parser.parse()
        result = parser.parsed_text
    return result
