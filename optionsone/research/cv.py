"""Chronological splits that only train on outcomes known before the next test."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train: tuple[int, ...]
    test: tuple[int, ...]


def purged_walk_forward_splits(
    event_times: Sequence[datetime],
    label_end_times: Sequence[datetime],
    n_splits: int,
    *,
    embargo: timedelta = timedelta(0),
) -> tuple[WalkForwardSplit, ...]:
    """Group equal timestamps, reserve the first block, and expand training chronologically.

    Intervals are closed [event, label_end]. In past-only training a post-test
    gap would have no effect, so apply the embargo before each test: a training
    outcome must end strictly before the first test event minus the embargo.
    Indices reference the original inputs, even when supplied out of order.
    """
    if n_splits < 1 or embargo < timedelta(0) or len(event_times) != len(label_end_times):
        raise ValueError("invalid split count, embargo, or interval alignment")
    groups: dict[datetime, list[int]] = {}
    for idx, (start, end) in enumerate(zip(event_times, label_end_times, strict=True)):
        if end < start:
            raise ValueError("label end precedes its event")
        groups.setdefault(start, []).append(idx)
    times = sorted(groups)
    size, remainder = divmod(len(times), n_splits + 1)
    if size == 0:
        raise ValueError("need at least n_splits + 1 distinct event times")
    offset = size + int(remainder > 0)
    splits: list[WalkForwardSplit] = []
    for fold in range(1, n_splits + 1):
        stop = offset + size + int(fold < remainder)
        cutoff = times[offset] - embargo
        train = tuple(
            idx for time in times[:offset] for idx in groups[time] if label_end_times[idx] < cutoff
        )
        if not train:
            raise ValueError("purging and embargo leave a fold without training samples")
        test = tuple(idx for time in times[offset:stop] for idx in groups[time])
        splits.append(WalkForwardSplit(train, test))
        offset = stop
    return tuple(splits)
