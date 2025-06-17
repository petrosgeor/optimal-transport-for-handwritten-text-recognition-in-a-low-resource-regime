import editdistance

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# character error rate
class CER:
    def __init__(self):
        self.total_dist = 0
        self.total_len = 0
        
    def update(self, prediction, target):            
        dist = float(editdistance.eval(prediction, target))
        self.total_dist += dist
        self.total_len += len(target)

    def score(self):
        return self.total_dist / self.total_len

    def reset(self):
        self.total_dist = 0
        self.total_len = 0
   
# word error rate     
# two supported modes: tokenizer & space
class WER:
    def __init__(self, mode='tokenizer'):
        self.total_dist = 0
        self.total_len = 0
        
        if mode not in ['tokenizer', 'space']:
            raise ValueError('mode must be either "tokenizer" or "space"')
        
        self.mode = mode
    
    def update(self, prediction, target):
        if self.mode == 'tokenizer':
            target = word_tokenize(target)
            prediction = word_tokenize(prediction)
        elif self.mode == 'space':
            target = target.split(' ')
            prediction = prediction.split(' ')
        
        dist = float(editdistance.eval(prediction, target))
        self.total_dist += dist
        self.total_len += len(target)
        
    def score(self):
        return self.total_dist / self.total_len

    def reset(self):
        self.total_dist = 0
        self.total_len = 0


def word_silhouette_score(features, words):
    """Return the average silhouette coefficient for *features* labelled by *words*.

    Parameters
    ----------
    features : torch.Tensor of shape ``(N, D)``
        Backbone descriptors.
    words : list[str]
        Ground-truth word for each descriptor.

    Returns
    -------
    float
        Mean silhouette score in ``[-1, 1]``.
    """
    from sklearn.metrics import pairwise_distances, silhouette_samples
    import torch

    label_map = {w: i for i, w in enumerate(sorted(set(words)))}
    labels = torch.tensor([label_map[w] for w in words])
    dist = pairwise_distances(features.numpy(), metric="euclidean")
    sil = silhouette_samples(dist, labels.numpy(), metric="precomputed")
    return float(sil.mean())
