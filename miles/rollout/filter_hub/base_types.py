from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if any(sample.reward is None or sample.get_reward_value(args) is None for sample in iter_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_missing_reward")

    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
