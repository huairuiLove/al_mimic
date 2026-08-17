"""First-party 76-channel hourly discretisation of benchmark events.

Reimplements the MIMIC-III Benchmark Discretizer configuration actually used
downstream: one-hour bins anchored at ICU admission (``start_time='zero'``),
``impute_strategy='previous'`` with the channel's normal value before the first
observation, categorical channels one-hot expanded, and one observed/imputed
mask channel appended per variable. 17 variables expand to 59 data columns plus
17 mask columns = 76 features, matching ``discretizer_config.json``.

Within one bin the value of the latest event wins, and events sharing a
timestamp keep the largest value — the same reduction the benchmark's
timeseries pivot performs upstream of the discretizer.

``end`` is the stay's length of stay in hours, exactly as the benchmark's
phenotyping reader passes the listfile's ``period_length``; the bin count is
``int(end / timestep + 1 - eps)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

RESOURCES = Path(__file__).with_name("resources")
DISCRETIZER_CONFIG = RESOURCES / "discretizer_config.json"
EPS = 1e-6
NORMALIZER_STD_FLOOR = 1e-7


@dataclass(frozen=True, slots=True)
class ChannelLayout:
    """Column layout of the 76-channel frame, fixed by the config file."""

    channels: tuple[str, ...]
    is_categorical: Mapping[str, bool]
    possible_values: Mapping[str, tuple[str, ...]]
    normal_values: Mapping[str, str]

    @property
    def data_columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for channel in self.channels:
            if self.is_categorical[channel]:
                names.extend(f"{channel}->{value}" for value in self.possible_values[channel])
            else:
                names.append(channel)
        return tuple(names)

    @property
    def columns(self) -> tuple[str, ...]:
        """Data columns followed by one mask column per variable."""
        return (*self.data_columns, *(f"mask->{channel}" for channel in self.channels))

    @property
    def continuous_columns(self) -> tuple[int, ...]:
        """Indices of data columns without a categorical expansion (no '->')."""
        return tuple(index for index, name in enumerate(self.data_columns) if "->" not in name)

    def offsets(self) -> dict[str, int]:
        """First data-column index of each channel."""
        positions: dict[str, int] = {}
        cursor = 0
        for channel in self.channels:
            positions[channel] = cursor
            cursor += len(self.possible_values[channel]) if self.is_categorical[channel] else 1
        return positions


def load_layout(config_path: str | Path = DISCRETIZER_CONFIG) -> ChannelLayout:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    layout = ChannelLayout(
        channels=tuple(config["id_to_channel"]),
        is_categorical=config["is_categorical_channel"],
        possible_values={
            channel: tuple(config["possible_values"][channel]) for channel in config["id_to_channel"]
        },
        normal_values=config["normal_values"],
    )
    total = len(layout.columns)
    if total != 76:
        raise ValueError(f"benchmark discretizer config must define 76 columns, got {total}")
    return layout


class BenchmarkDiscretizer:
    """Events of one ICU stay to a padded [steps, 76] frame plus a step mask."""

    def __init__(self, timestep: float = 1.0, max_steps: int = 256) -> None:
        self.layout = load_layout()
        self.timestep = float(timestep)
        self.max_steps = int(max_steps)
        self._offsets = self.layout.offsets()
        self._channel_index = {channel: position for position, channel in enumerate(self.layout.channels)}

    def bin_count(self, end_hours: float) -> int:
        return int(end_hours / self.timestep + 1.0 - EPS)

    def transform(
        self,
        events: Mapping[str, Mapping[float, float | str]],
        end_hours: float,
        *,
        max_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Discretise one stay.

        ``events`` maps channel name to {hours since admission: value}; within a
        channel the caller has already collapsed duplicate timestamps.
        Returns ``(frame[min(bins, max_steps), 76] float32, observed[...] bool)``
        where ``observed`` marks bins that exist before the step cap. Passing a
        ``max_steps`` larger than the stay's bin count lifts the cap, which the
        normalizer needs to mirror the benchmark's full-length statistics.
        """
        cap = self.max_steps if max_steps is None else max_steps
        bins = min(self.bin_count(end_hours), cap)
        data = np.zeros((bins, len(self.layout.data_columns)), dtype=np.float64)
        observed = np.zeros((bins, len(self.layout.channels)), dtype=bool)

        for channel, series in events.items():
            channel_id = self._channel_index[channel]
            for hours, value in sorted(series.items()):
                bin_id = int(hours / self.timestep - EPS)
                if not 0 <= bin_id < bins:
                    continue
                self._write(data, bin_id, channel, value)
                observed[bin_id, channel_id] = True

        self._impute_previous(data, observed)
        frame = np.zeros((bins, len(self.layout.columns)), dtype=np.float32)
        frame[:bins, : data.shape[1]] = data.astype(np.float32)
        frame[:bins, data.shape[1] :] = observed.astype(np.float32)
        step_mask = np.zeros(bins, dtype=bool)
        step_mask[:bins] = True
        return frame, step_mask

    def _write(self, data: np.ndarray, bin_id: int, channel: str, value: float | str) -> None:
        base = self._offsets[channel]
        if self.layout.is_categorical[channel]:
            values = self.layout.possible_values[channel]
            text = str(value)
            try:
                category = values.index(text)
            except ValueError as exc:
                raise ValueError(
                    f"value {text!r} of channel {channel!r} is not one of the configured categories"
                ) from exc
            data[bin_id, base + category] = 1.0
        else:
            data[bin_id, base] = float(value)

    def _impute_previous(self, data: np.ndarray, observed: np.ndarray) -> None:
        for channel in self.layout.channels:
            channel_id = self._channel_index[channel]
            base = self._offsets[channel]
            width = len(self.layout.possible_values[channel]) if self.layout.is_categorical[channel] else 1
            last: np.ndarray | None = None
            for bin_id in range(data.shape[0]):
                if observed[bin_id, channel_id]:
                    if width == 1:
                        last = data[bin_id, base : base + 1].copy()
                    else:
                        last = data[bin_id, base : base + width].copy()
                    continue
                if last is None:
                    self._write(data, bin_id, channel, self.layout.normal_values[channel])
                elif width == 1:
                    data[bin_id, base : base + 1] = last
                else:
                    data[bin_id, base : base + width] = last


class BenchmarkNormalizer:
    """Z-score the continuous columns with train-split statistics."""

    def __init__(self, columns: Sequence[int]) -> None:
        self.columns = list(columns)
        self._sum: np.ndarray | None = None
        self._sum_sq: np.ndarray | None = None
        self._count = 0
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def feed(self, frame: np.ndarray) -> None:
        """Accumulate statistics over one stay's full-length frame."""
        selected = frame[:, self.columns].astype(np.float64)
        self._count += selected.shape[0]
        sums = selected.sum(axis=0)
        squares = (selected**2).sum(axis=0)
        self._sum = sums if self._sum is None else self._sum + sums
        self._sum_sq = squares if self._sum_sq is None else self._sum_sq + squares

    def finalize(self) -> None:
        if self._count < 2 or self._sum is None or self._sum_sq is None:
            raise ValueError("normalizer needs at least two rows of statistics")
        count = self._count
        self.means = self._sum / count
        variance = (self._sum_sq - count * self.means**2) / (count - 1)
        self.stds = np.sqrt(np.maximum(variance, 0.0))
        self.stds[self.stds < NORMALIZER_STD_FLOOR] = NORMALIZER_STD_FLOOR

    def transform(self, frame: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("normalizer statistics are not finalized")
        normalized = frame
        for position, column in enumerate(self.columns):
            normalized[:, column] = (frame[:, column] - self.means[position]) / self.stds[position]
        return normalized

    def state(self) -> dict[str, list[float]]:
        if self.means is None or self.stds is None:
            raise ValueError("normalizer statistics are not finalized")
        return {
            "columns": list(self.columns),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "rows": self._count,
        }


def iter_stay_events(
    rows: Iterable[tuple[float, str, float | str]],
) -> dict[str, dict[float, float | str]]:
    """Collapse (hours, channel, value) rows, keeping the max value per timestamp.

    Continuous channels compare numerically; categorical text compares
    lexicographically, the order a string-valued sort would give upstream.
    """
    series: dict[str, dict[float, float | str]] = {}
    for hours, channel, value in rows:
        slot = series.setdefault(channel, {})
        previous = slot.get(hours)
        if previous is None or _max_value(previous, value) != previous:
            slot[hours] = value
    return series


def _max_value(left: float | str, right: float | str) -> float | str:
    try:
        return left if float(left) >= float(right) else right
    except (TypeError, ValueError):
        return left if str(left) >= str(right) else right
