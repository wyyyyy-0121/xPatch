"""Nested expanding-date partitions with both purge boundaries kept out of fit."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Partition:
    name: str
    fit_dates: tuple
    calibration_dates: tuple
    evaluation_dates: tuple
    purge_before_calibration: tuple
    purge_before_evaluation: tuple

    def dictionary(self):
        return asdict(self)


def partition(name, prior_dates, evaluation_dates, calibration_sessions=20, purge_sessions=1):
    prior = tuple(prior_dates)
    evaluation = tuple(evaluation_dates)
    if prior != tuple(sorted(set(prior))) or evaluation != tuple(sorted(set(evaluation))):
        raise ValueError("Partition dates must be sorted and unique")
    if calibration_sessions < 1 or purge_sessions < 1:
        raise ValueError("Calibration and purge require positive whole-session counts")
    needed = calibration_sessions + 2 * purge_sessions
    if not evaluation or len(prior) <= needed or prior[-1] >= evaluation[0]:
        raise ValueError("Insufficient or overlapping pre-evaluation history")
    purge_evaluation = prior[-purge_sessions:]
    calibration = prior[-purge_sessions - calibration_sessions:-purge_sessions]
    purge_calibration = prior[-needed:-calibration_sessions - purge_sessions]
    fit = prior[:-needed]
    parts = (fit, purge_calibration, calibration, purge_evaluation, evaluation)
    flattened = [day for group in parts for day in group]
    if flattened != sorted(set(flattened)):
        raise ValueError("Partition overlap or ordering violation")
    return Partition(name, fit, calibration, evaluation, purge_calibration, purge_evaluation)


def inner_partitions(outer, folds=3, validation_sessions=20, calibration_sessions=20, purge_sessions=1):
    dates = outer.fit_dates
    if folds < 3 or validation_sessions < 1:
        raise ValueError("At least three expanding inner folds required")
    result = []
    for i in range(folds):
        stop = len(dates) - (folds - 1 - i) * validation_sessions
        start = stop - validation_sessions
        if start <= 0:
            raise ValueError("Not enough training history for all inner folds")
        result.append(partition(f"{outer.name}_inner{i + 1}", dates[:start], dates[start:stop],
            calibration_sessions, purge_sessions))
    return result


def outer_partitions(dates, start="2023-01-01", end="2025-06-30", calibration_sessions=20, purge_sessions=1):
    dates = tuple(dates)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("Exchange dates must be sorted and unique")
    quarters = sorted({f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}" for day in dates if start <= day <= end})
    result = []
    for name in quarters:
        evaluation = tuple(day for day in dates if start <= day <= end and
            f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}" == name)
        prior = tuple(day for day in dates if day < evaluation[0])
        result.append(partition(name, prior, evaluation, calibration_sessions, purge_sessions))
    if len(result) != 10 or result[0].name != "2023Q1" or result[-1].name != "2025Q2":
        raise ValueError("Expected exactly the ten registered development quarters")
    return result
