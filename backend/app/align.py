# Time-aligns the singer's take to the reference using dynamic time warping
# (DTW).
#
# In plain words: the singer never sings at exactly the reference's tempo -
# they rush some words and drag others. DTW finds, for every moment in the
# take, the best-matching moment in the reference, while requiring the
# matches to stay in order (you cannot match take-second-5 to a reference
# moment earlier than the one you matched take-second-4 to). It does this
# by building a cost matrix of "how different do these two moments sound"
# for every (take frame, reference frame) pair, then finding the cheapest
# path through that matrix from the start to the end. The result is a
# warping path: a list of (reference_time, take_time) pairs describing how
# the take's timeline bends to line up with the reference's timeline.
#
# We compare MFCCs (a standard timbre/phoneme-shape feature) rather than
# raw pitch, since DTW works best on features that change smoothly and
# distinctly over time, and MFCCs capture syllable-level structure even
# when the take is off pitch.

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

HOP_LENGTH = 512
N_MFCC = 20


@dataclass
class AlignmentResult:
    # warping path, oldest to newest, one entry per matched frame pair
    ref_times: np.ndarray
    take_times: np.ndarray

    def take_to_ref(self, take_time: float) -> float:
        """Map a moment in the take's timeline to the matching reference moment."""
        return float(np.interp(take_time, self.take_times, self.ref_times))


def _mfcc(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sample_rate, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    # normalize each coefficient so no single one dominates the distance
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-8
    return (mfcc - mean) / std


def align(
    ref_samples: np.ndarray,
    ref_sample_rate: int,
    take_samples: np.ndarray,
    take_sample_rate: int,
) -> AlignmentResult:
    ref_features = _mfcc(ref_samples, ref_sample_rate)
    take_features = _mfcc(take_samples, take_sample_rate)

    # librosa's dtw returns the cost matrix D and the optimal warping path
    # wp, as pairs of (ref_frame_index, take_frame_index), newest first.
    _cost, warp_path = librosa.sequence.dtw(X=ref_features, Y=take_features, metric="cosine")

    warp_path = warp_path[::-1]  # oldest to newest
    ref_frames = warp_path[:, 0]
    take_frames = warp_path[:, 1]

    ref_times = librosa.frames_to_time(ref_frames, sr=ref_sample_rate, hop_length=HOP_LENGTH)
    take_times = librosa.frames_to_time(take_frames, sr=take_sample_rate, hop_length=HOP_LENGTH)

    # the path can repeat a frame on plateaus; np.interp needs take_times
    # strictly increasing, so keep only the first occurrence of each step
    _, unique_idx = np.unique(take_times, return_index=True)
    unique_idx = np.sort(unique_idx)

    return AlignmentResult(ref_times=ref_times[unique_idx], take_times=take_times[unique_idx])
