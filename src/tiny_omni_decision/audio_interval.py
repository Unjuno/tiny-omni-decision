"""Exact decoded sample-clock bounds; no metadata padding or implicit resampling."""

from fractions import Fraction


def sample_interval(blocks, sample_rate, start_us, end_us):
    """blocks are (integer PTS, rational time base, decoded sample count).

    Return half-open indices in concatenated decoded PCM. Include only samples
    whose complete support lies inside the requested half-open time interval.
    """
    if type(sample_rate) is not int or sample_rate <= 0 or not blocks:
        raise ValueError("positive sample rate and decoded blocks required")
    if type(start_us) is not int or type(end_us) is not int or not 0 <= start_us < end_us:
        raise ValueError("integer microsecond interval required")
    origin, cursor, total = None, None, 0
    for pts, time_base, samples in blocks:
        if type(pts) is not int or type(samples) is not int or samples <= 0:
            raise ValueError("valid decoded block required")
        base = Fraction(time_base)
        if base <= 0:
            raise ValueError("positive time base required")
        t = pts * base
        if origin is None:
            origin = t
        if cursor is not None and t != cursor:
            raise ValueError("decoded audio gap or overlap")
        cursor = t + Fraction(samples, sample_rate)
        total += samples
    start, end = Fraction(start_us, 1_000_000), Fraction(end_us, 1_000_000)
    if start < origin or end > cursor:
        raise ValueError("requested interval exceeds decoded audio")
    first = (start - origin) * sample_rate
    last = (end - origin) * sample_rate
    begin = -(-first.numerator // first.denominator)
    stop = last.numerator // last.denominator
    if not 0 <= begin < stop <= total:
        raise ValueError("interval has no complete samples")
    return begin, stop
