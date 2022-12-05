from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t{self.score:4.2f}\t{self.start:5d}\t{self.end:5d}"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens, blank_id=0, cross_attention=None):
    """
    Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
    that represents the probabilities of each source token being at a certain time step
    """
    num_frames = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frames):
        emission_t_tokens = emission[t, tokens]
        # for j, token in enumerate(tokens):
        # 	if token == 4:
        # 		emission_t_tokens[j] *= 100.0
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission_t_tokens,
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0, cross_attention=None):
    """
    Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path
    """
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def merge_repeats(path, transcript):
    """
    Merge repeated tokens into a single segment. Note: this shouldn't affect repeated characters from the
    original sentences (e.g. `ll` in `hello`)
    """
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def get_duration_from_emission(
    emission, tokens, transcript, blank_id=0, cross_attention=None
):
    trellis = get_trellis(
        emission, tokens, blank_id=blank_id, cross_attention=cross_attention
    )
    path = backtrack(
        trellis, emission, tokens, blank_id=blank_id, cross_attention=cross_attention
    )
    segments = merge_repeats(path, transcript)
    char_durations = []
    for segment in segments:
        char_durations.append(segment.end - segment.start)
    return char_durations, segments
