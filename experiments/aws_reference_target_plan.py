"""Exact target coverage for the audited Jan26 modded-nanoGPT reference.

Pure protocol planning: no model imports, fitting or public scoring. Execution
chunks preserve sequence state; only the first chunk resets it. EOS resets are
handled by the native causal evaluator before consuming each observed EOS.
"""
from dataclasses import dataclass

REFERENCE_TARGETS = 10485760
REFERENCE_SEQUENCE_TARGETS = 262144
REFERENCE_LANES = 8
REFERENCE_BATCHES = 5


@dataclass(frozen=True)
class TargetWindow:
    batch: int
    offset: int
    count: int
    input_starts: tuple
    reset_sequence: bool

    @property
    def target_starts(self):
        return tuple(start + 1 for start in self.input_starts)

    @property
    def input_ranges(self):
        return tuple((start, start + self.count) for start in self.input_starts)

    @property
    def target_ranges(self):
        return tuple((start, start + self.count) for start in self.target_starts)

    @property
    def lane_load_ranges(self):
        """Half-open token slices, including a lookahead for every lane."""
        return tuple((start, start + self.count + 1) for start in self.input_starts)

    @property
    def scored_targets(self):
        return len(self.input_starts) * self.count


def target_windows(chunk_targets, available_tokens):
    if isinstance(chunk_targets, bool) or not isinstance(chunk_targets, int) or chunk_targets <= 0:
        raise ValueError('Positive integer execution chunk required')
    if available_tokens < REFERENCE_TARGETS + 1:
        raise ValueError('Reference scoring requires its final lookahead token')
    for batch in range(REFERENCE_BATCHES):
        base = batch * REFERENCE_LANES * REFERENCE_SEQUENCE_TARGETS
        for offset in range(0, REFERENCE_SEQUENCE_TARGETS, chunk_targets):
            count = min(chunk_targets, REFERENCE_SEQUENCE_TARGETS - offset)
            starts = tuple(base + lane * REFERENCE_SEQUENCE_TARGETS + offset
                           for lane in range(REFERENCE_LANES))
            yield TargetWindow(batch, offset, count, starts, offset == 0)


def validate_plan(windows):
    """Prove coverage by contiguous per-sequence ranges without enumerating tokens."""
    ends = {}
    total = 0
    resets = 0
    windows_count = 0
    for window in windows:
        if not 0 <= window.batch < REFERENCE_BATCHES or window.count <= 0:
            raise ValueError('Invalid batch or empty target window')
        if len(window.input_starts) != REFERENCE_LANES:
            raise ValueError('Reference requires eight rank sequences per batch')
        if window.reset_sequence != (window.offset == 0):
            raise ValueError('Only sequence starts reset persistent state')
        if window.offset + window.count > REFERENCE_SEQUENCE_TARGETS:
            raise ValueError('Execution window crosses a sequence boundary')
        for lane, (begin, end) in enumerate(window.target_ranges):
            sequence = window.batch * REFERENCE_LANES + lane
            expected = ends.get(sequence, sequence * REFERENCE_SEQUENCE_TARGETS + 1)
            if begin != expected or begin != sequence * REFERENCE_SEQUENCE_TARGETS + window.offset + 1:
                raise ValueError('Missing, duplicate or reordered targets within a sequence')
            ends[sequence] = end
        total += window.scored_targets
        resets += REFERENCE_LANES if window.reset_sequence else 0
        windows_count += 1
    if len(ends) != 40 or resets != 40 or total != REFERENCE_TARGETS:
        raise ValueError('Reference coverage/reset count is incomplete')
    if any(end != (sequence + 1) * REFERENCE_SEQUENCE_TARGETS + 1 for sequence, end in ends.items()):
        raise ValueError('Final sequence targets omitted')
    return dict(targets=total, reset_sequences=resets, execution_windows=windows_count,
                first_target=1, last_target=REFERENCE_TARGETS,
                required_tokens=REFERENCE_TARGETS+1,
                history='Carry numerical state between execution chunks; reset at each rank sequence start and observed EOS')
