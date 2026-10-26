import editdistance


# character error rate
class CER:
    """Accumulator for character error rate."""

    def __init__(self) -> None:
        """Initialise counters to zero."""
        self.total_dist = 0
        self.total_len = 0

    def update(self, prediction: str, target: str) -> None:
        """Accumulate edit distance for one prediction/target pair.

        Args:
            prediction (str): Predicted text.
            target (str): Ground-truth text.
        """
        dist = float(editdistance.eval(prediction, target))
        self.total_dist += dist
        self.total_len += len(target)

    def score(self) -> float:
        """Return the current CER value.

        Returns:
            float: Character error rate accumulated so far.
        """
        return self.total_dist / self.total_len

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.total_dist = 0
        self.total_len = 0
