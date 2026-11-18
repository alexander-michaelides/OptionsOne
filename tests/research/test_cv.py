from datetime import UTC, datetime, timedelta

from optionsone.research.cv import purged_walk_forward_splits


def test_purge_and_embargo_drop_overlaps_and_keep_tied_events_together() -> None:
    t0 = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    minute = timedelta(minutes=1)
    events = [t0 + x * minute for x in (5, 0, 1, 2, 3, 4, 4, 6, 7, 8)]
    ends = [t0 + x * minute for x in (6, 1, 3, 4, 11, 5, 6, 7, 8, 9)]
    folds = purged_walk_forward_splits(events, ends, 2, embargo=minute)
    assert folds[0].train == (1,)
    assert folds[0].test == (4, 5, 6, 0)
    assert folds[1].train == (1, 2, 3)
    for fold in folds:
        for i in fold.train:
            assert all(ends[i] + minute < events[j] for j in fold.test)
