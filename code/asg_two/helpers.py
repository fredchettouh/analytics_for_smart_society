def slidingwindowsegment(sequence, create_segment, compute_error, max_error,
                         seq_range=None):
    """
    Return a list of line segments that approximate the sequence.
    The list is computed using the sliding window technique.
    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error
    """
    if not seq_range:
        seq_range = (0, len(sequence) - 1)

    start = seq_range[0]
    end = start
    result_segment = create_segment(sequence, (seq_range[0], seq_range[1]))
    while end < seq_range[1]:
        end += 1
        test_segment = create_segment(sequence, (start, end))
        error = compute_error(sequence, test_segment)
        if error <= max_error:
            result_segment = test_segment
        else:
            break

    if end == seq_range[1]:
        return [result_segment]
    else:
        return [result_segment] + slidingwindowsegment(sequence, create_segment,
                                                       compute_error, max_error,
                                                       (end - 1, seq_range[1]))
