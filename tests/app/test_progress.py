from pathlib import Path

from optionsone.app.fixtures import synthetic_fixture
from optionsone.app.progress import ConsoleProgress
from optionsone.app.research_cmd import backfill, backtest_store
from optionsone.classify.stub import DeterministicStubClassifier
from tests.app.test_research_cmd import FakeBars, FakeNews


class FakeTimer:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_ticks_are_throttled_and_show_rate_and_eta() -> None:
    lines: list[str] = []
    timer = FakeTimer()
    progress = ConsoleProgress(lines.append, interval=2.0, timer=timer)
    progress.stage("classify", "100 pairs")
    timer.now = 1.0
    progress.tick(10, 100)  # within the interval: suppressed
    timer.now = 5.0
    progress.tick(50, 100)  # 10/s, 50 left -> 5s
    timer.now = 6.0
    progress.tick(100, 100)  # completion always prints
    progress.close()
    assert lines == [
        "[classify] 100 pairs",
        "[classify] 50/100 (50%)  10.0/s  ETA 5s",
        "[classify] 100/100 (100%)",
        "[classify] done in 6s",
        "[total] 6s",
    ]


def test_long_durations_use_hours_and_minutes() -> None:
    lines: list[str] = []
    timer = FakeTimer()
    progress = ConsoleProgress(lines.append, interval=0, timer=timer)
    progress.stage("classify")
    timer.now = 100.0
    progress.tick(1, 100)  # 0.01/s, 99 left -> 9900s
    assert lines[-1].endswith("ETA 2h45m")


class Recorder:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.final: dict[str, tuple[int, int]] = {}

    def stage(self, name: str, detail: str = "") -> None:
        self.stages.append(name)

    def tick(self, done: int, total: int) -> None:
        self.final[self.stages[-1]] = (done, total)

    def note(self, message: str) -> None:
        pass


async def test_store_backtest_reports_every_stage_to_completion(tmp_path: Path) -> None:
    data = synthetic_fixture(days=6, seed=3)
    start, end = data.bars[0].start, data.bars[-1].end
    await backfill(
        tmp_path / "s", (), start, end, news=FakeNews(data.articles), bars=FakeBars(data.bars)
    )
    recorder = Recorder()
    await backtest_store(
        tmp_path / "s",
        start,
        end,
        DeterministicStubClassifier(),
        registry_path=tmp_path / "t.jsonl",
        progress=recorder,
    )
    assert recorder.stages == ["load", "classify", "split", "features", "evaluate", "fit", "replay"]
    for stage in ("classify", "features", "replay"):
        done, total = recorder.final[stage]
        assert done == total > 0
