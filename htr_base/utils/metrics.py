import editdistance

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# character error rate
class CER:
    """Accumulator for character error rate."""

    def __init__(self) -> None:
        """Initialise counters to zero."""
        self.total_dist = 0
        self.total_len = 0
        
    def update(self, prediction: str, target: str) -> None:
        """Accumulate edit distance for one prediction/target pair."""
        dist = float(editdistance.eval(prediction, target))
        self.total_dist += dist
        self.total_len += len(target)

    def score(self) -> float:
        """Return the current CER value."""
        return self.total_dist / self.total_len

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.total_dist = 0
        self.total_len = 0
   
# word error rate     
# two supported modes: tokenizer & space
class WER:
    """Word error rate metric supporting two tokenisation modes."""

    def __init__(self, mode: str = 'tokenizer') -> None:
        """Create a WER accumulator.

        Args:
            mode (str): Either ``'tokenizer'`` or ``'space'`` controlling how words are split.
        """
        self.total_dist = 0
        self.total_len = 0
        
        if mode not in ['tokenizer', 'space']:
            raise ValueError('mode must be either "tokenizer" or "space"')
        
        self.mode = mode
    
    def update(self, prediction: str, target: str) -> None:
        """Update WER statistics with a new prediction."""
        if self.mode == 'tokenizer':
            target = word_tokenize(target)
            prediction = word_tokenize(prediction)
        elif self.mode == 'space':
            target = target.split(' ')
            prediction = prediction.split(' ')
        
        dist = float(editdistance.eval(prediction, target))
        self.total_dist += dist
        self.total_len += len(target)
        
    def score(self) -> float:
        """Return the current WER value."""
        return self.total_dist / self.total_len

    def reset(self) -> None:
        """Reset the internal counters."""
        self.total_dist = 0
        self.total_len = 0


