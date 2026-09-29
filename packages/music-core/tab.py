"""Guitar tablature: string/fret positions, fingering optimization and ASCII rendering.

The ML model only detects notes (pitch, onset, offset). Turning pitches into
string/fret positions is an algorithmic problem: every pitch can be played in several
places, so we pick the sequence of positions with minimum total cost (Viterbi):

    cost = |hand - preferred_fret| + open_strings + chord_span + awkward_stretch  (per event)
         + position_change + string_change                          (consecutive events)
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from music_core.notes import NOTE_NAMES, Note, sort_notes

# Open-string MIDI pitches from the lowest string (index 0) to the highest.
STANDARD_TUNING: tuple[int, ...] = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4
DROP_D_TUNING: tuple[int, ...] = (38, 45, 50, 55, 59, 64)
DEFAULT_NUM_FRETS = 20
MAX_CAPO = 12

# Six-string tunings the app offers. Keep in sync with TUNINGS in apps/web/lib/music.ts
# (checked by tests/test_web_fixtures.py). The lowest note, C2, is the note model's.
TUNINGS: dict[str, tuple[int, ...]] = {
    "standard": STANDARD_TUNING,
    "half_step_down": (39, 44, 49, 54, 58, 63),  # Eb Ab Db Gb Bb eb
    "full_step_down": (38, 43, 48, 53, 57, 62),  # D G C F A d
    "drop_d": DROP_D_TUNING,
    "drop_c": (36, 43, 48, 53, 57, 62),  # C G C F A d
    "c_standard": (36, 41, 46, 51, 55, 60),  # C F Bb Eb G c
    "dadgad": (38, 45, 50, 55, 57, 62),
    "open_g": (38, 43, 50, 55, 59, 62),  # D G D G B d
    "open_d": (38, 45, 50, 54, 57, 62),  # D A D F# A d
}


def open_strings(name: str = "standard", capo: int = 0) -> tuple[int, ...]:
    """Open-string pitches for a named tuning with a capo; frets then count from the capo."""
    if name not in TUNINGS:
        raise ValueError(f"Unknown tuning {name!r}; expected one of {sorted(TUNINGS)}")
    if not 0 <= capo <= MAX_CAPO:
        raise ValueError(f"Capo must be between 0 and {MAX_CAPO}, got {capo}")
    return tuple(pitch + capo for pitch in TUNINGS[name])


Position = tuple[int, int]  # (string, fret)
State = tuple[Position | None, ...]  # one position per playable note of an event


@dataclass(frozen=True)
class TabCostWeights:
    """Defaults were grid-searched on the GuitarSet validation split (player 04):
    string/fret accuracy on ground-truth notes rose from 0.62 (lowest-position,
    free-open-string heuristic) to 0.77; players mostly sit around frets 5-9.
    """

    fret_height: float = 0.3  # per fret of distance between the hand and preferred_fret
    preferred_fret: float = 5.0  # hand position players gravitate to
    open_string: float = 1.0  # cost per open string
    span: float = 1.0  # distance between lowest and highest fret in a chord
    awkward: float = 3.0  # extra cost per fret of stretch beyond 4 frets
    position_change: float = 2.0  # hand movement between consecutive events (frets)
    string_change: float = 0.0  # jumps across strings between consecutive events
    unplaced: float = 10.0  # a note that could not be placed (more notes than strings)
    # Per note: -log p(position) from a learned tab model; only used when assign_tab
    # receives position_probs. Validation accuracy plateaus for weights of 8-32.
    model_evidence: float = 16.0


EVIDENCE_FLOOR = 1e-4  # caps -log p for positions the model considers impossible


def fret_to_pitch(string: int, fret: int, tuning: Sequence[int] = STANDARD_TUNING) -> int:
    return tuning[string] + fret


def pitch_to_positions(
    pitch: int,
    tuning: Sequence[int] = STANDARD_TUNING,
    num_frets: int = DEFAULT_NUM_FRETS,
) -> list[Position]:
    """All (string, fret) pairs that produce ``pitch``."""
    return [
        (string, pitch - open_pitch)
        for string, open_pitch in enumerate(tuning)
        if 0 <= pitch - open_pitch <= num_frets
    ]


def is_valid_position(
    note: Note,
    tuning: Sequence[int] = STANDARD_TUNING,
    num_frets: int = DEFAULT_NUM_FRETS,
) -> bool:
    if note.string is None or note.fret is None:
        return False
    if not (0 <= note.string < len(tuning) and 0 <= note.fret <= num_frets):
        return False
    return tuning[note.string] + note.fret == note.pitch


def assign_tab(
    notes: Sequence[Note],
    tuning: Sequence[int] = STANDARD_TUNING,
    num_frets: int = DEFAULT_NUM_FRETS,
    chord_tolerance: float = 0.05,
    weights: TabCostWeights | None = None,
    max_candidates: int = 32,
    position_probs: Sequence[Mapping[Position, float] | None] | None = None,
) -> list[Note]:
    """Give every note a string/fret along the minimum-cost fingering path.

    Notes starting within ``chord_tolerance`` seconds form one event (a chord) and must
    use distinct strings. Notes that already carry a valid position keep it. Notes that
    cannot be placed get ``string=None, fret=None``. The input order is preserved.

    ``position_probs`` (aligned with ``notes``) optionally carries a learned model's
    belief in each (string, fret) of a note; it enters the cost as
    ``model_evidence * -log p``, so the model steers the choice while the optimizer
    keeps chords playable and hand movement small.
    """
    tuning = tuple(tuning)
    weights = weights or TabCostWeights()
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))

    events: list[tuple[list[int], list[int], list[tuple[State, float]]]] = []
    for group in _onset_groups(order, notes, chord_tolerance):
        playable: list[int] = []
        options: list[list[Position]] = []
        probs: list[Mapping[Position, float] | None] = []
        for i in group:
            note = notes[i]
            if is_valid_position(note, tuning, num_frets):
                opts = [(note.string, note.fret)]
            else:
                opts = pitch_to_positions(note.pitch, tuning, num_frets)
            if opts:
                playable.append(i)
                options.append(opts)  # type: ignore[arg-type]
                probs.append(position_probs[i] if position_probs is not None else None)
        candidates = _candidates(options, weights, max_candidates, probs)
        events.append((group, playable, candidates))

    path = _viterbi([candidates for _, _, candidates in events], weights)

    result = list(notes)
    for (group, playable, _), state in zip(events, path):
        for i in group:
            result[i] = replace(notes[i], string=None, fret=None)
        for i, position in zip(playable, state):
            if position is not None:
                result[i] = replace(notes[i], string=position[0], fret=position[1])
    return result


def _onset_groups(order: list[int], notes: Sequence[Note], tolerance: float) -> Iterator[list[int]]:
    group: list[int] = []
    group_start = 0.0
    for i in order:
        if group and notes[i].start - group_start > tolerance:
            yield group
            group = []
        if not group:
            group_start = notes[i].start
        group.append(i)
    if group:
        yield group


def _candidates(
    options: list[list[Position]],
    weights: TabCostWeights,
    limit: int,
    probs: list[Mapping[Position, float] | None],
) -> list[tuple[State, float]]:
    """Conflict-free assignments of an event (distinct strings) with their local cost,
    cheapest first."""
    states: list[State] = []
    chosen: list[Position] = []
    used: set[int] = set()

    def search(k: int) -> None:
        if k == len(options):
            states.append(tuple(chosen))
            return
        for position in options[k]:
            if position[0] in used:
                continue
            chosen.append(position)
            used.add(position[0])
            search(k + 1)
            chosen.pop()
            used.discard(position[0])

    search(0)
    if not states:  # e.g. more simultaneous notes than strings
        states = [_greedy_state(options)]
    scored = [(s, _local_cost(s, weights) + _evidence_cost(s, probs, weights)) for s in states]
    scored.sort(key=lambda item: item[1])
    return scored[:limit]


def _evidence_cost(
    state: State, probs: list[Mapping[Position, float] | None], w: TabCostWeights
) -> float:
    cost = 0.0
    for position, note_probs in zip(state, probs):
        if position is not None and note_probs is not None:
            cost -= math.log(max(note_probs.get(position, 0.0), EVIDENCE_FLOOR))
    return w.model_evidence * cost


def _greedy_state(options: list[list[Position]]) -> State:
    used: set[int] = set()
    chosen: list[Position | None] = []
    for opts in options:
        free = [p for p in opts if p[0] not in used]
        position = min(free, key=lambda p: p[1]) if free else None
        if position is not None:
            used.add(position[0])
        chosen.append(position)
    return tuple(chosen)


def _local_cost(state: State, w: TabCostWeights) -> float:
    cost = w.unplaced * sum(1 for p in state if p is None)
    cost += w.open_string * sum(1 for p in state if p is not None and p[1] == 0)
    frets = [p[1] for p in state if p is not None and p[1] > 0]
    if frets:
        span = max(frets) - min(frets)
        center = sum(frets) / len(frets)
        cost += w.fret_height * abs(center - w.preferred_fret) + w.span * span
        cost += w.awkward * max(0, span - 4)
    return cost


def _hand_center(state: State) -> float | None:
    frets = [p[1] for p in state if p is not None and p[1] > 0]
    return sum(frets) / len(frets) if frets else None


def _mean_string(state: State) -> float | None:
    strings = [p[0] for p in state if p is not None]
    return sum(strings) / len(strings) if strings else None


def _viterbi(events: list[list[tuple[State, float]]], w: TabCostWeights) -> list[State]:
    """Minimum-cost path through the (state, local cost) candidates of consecutive events."""
    if not events:
        return []

    def features(
        candidates: list[tuple[State, float]],
    ) -> list[tuple[float, float | None, float | None]]:
        return [(cost, _hand_center(s), _mean_string(s)) for s, cost in candidates]

    def transition(a: tuple, b: tuple) -> float:
        cost = 0.0
        if a[1] is not None and b[1] is not None:
            cost += w.position_change * abs(a[1] - b[1])
        if a[2] is not None and b[2] is not None:
            cost += w.string_change * abs(a[2] - b[2])
        return cost

    prev = features(events[0])
    costs = [f[0] for f in prev]
    backpointers: list[list[int]] = []
    for states in events[1:]:
        current = features(states)
        new_costs, pointers = [], []
        for feat in current:
            best_j, best = 0, math.inf
            for j, prev_feat in enumerate(prev):
                total = costs[j] + transition(prev_feat, feat)
                if total < best:
                    best_j, best = j, total
            new_costs.append(best + feat[0])
            pointers.append(best_j)
        costs, prev = new_costs, current
        backpointers.append(pointers)

    index = min(range(len(costs)), key=costs.__getitem__)
    path = [events[-1][index][0]]
    for pointers, candidates in zip(reversed(backpointers), reversed(events[:-1])):
        index = pointers[index]
        path.append(candidates[index][0])
    return path[::-1]


def string_labels(tuning: Sequence[int] = STANDARD_TUNING) -> list[str]:
    """Pitch-class names per string; the highest string is lower-cased (``e``)."""
    labels = [NOTE_NAMES[p % 12] for p in tuning]
    if labels:
        labels[-1] = labels[-1].lower()
    return labels


def tab_to_ascii(
    notes: Sequence[Note],
    tuning: Sequence[int] = STANDARD_TUNING,
    columns_per_second: float = 8.0,
    line_width: int = 80,
    chord_tolerance: float = 0.05,
    capo: int = 0,
) -> str:
    """Render positioned notes as plain-text tablature (highest string on top).

    Columns follow the timing, but each event (a note or a chord) starts at least one
    dash after the previous one, so close notes never merge (5 then 6 must not read as
    fret 56). Chords stay vertically aligned and lines never break inside a number.
    With a capo, ``tuning`` includes it (frets count from the capo), the strings keep
    their own names and a "Capo N" line comes first.
    """
    placed = sort_notes(
        n
        for n in notes
        if n.string is not None and n.fret is not None and 0 <= n.string < len(tuning)
    )
    if not placed:
        return ""

    events: list[tuple[int, list[Note]]] = []
    cursor = 0
    for group in _onset_groups(list(range(len(placed))), placed, chord_tolerance):
        members = [placed[i] for i in group]
        col = max(int(round(members[0].start * columns_per_second)), cursor)
        events.append((col, members))
        cursor = col + max(len(str(n.fret)) for n in members) + 1

    total_cols = cursor + 2
    rows = [["-"] * total_cols for _ in tuning]
    for col, members in events:
        for note in members:
            for offset, char in enumerate(str(note.fret)):
                rows[note.string][col + offset] = char  # type: ignore[index]

    def inside_number(col: int) -> bool:  # a fret number continues from col - 1 into col
        return any(row[col - 1].isdigit() and row[col].isdigit() for row in rows)

    labels = string_labels([pitch - capo for pitch in tuning])
    label_width = max(len(label) for label in labels)
    blocks = []
    start = 0
    while start < total_cols:
        end = min(start + line_width, total_cols)
        while start + 1 < end < total_cols and inside_number(end):
            end -= 1
        lines = [
            f"{labels[s]:<{label_width}}|{''.join(rows[s][start:end])}|"
            for s in reversed(range(len(tuning)))
        ]
        blocks.append("\n".join(lines))
        start = end
    header = f"Capo {capo}\n\n" if capo else ""
    return header + "\n\n".join(blocks) + "\n"
