import io,os
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from .preprocessing import load_image, preprocess
from wordfreq import top_n_list, word_frequency
from sklearn.manifold import MDS
import editdistance
import random

class HTRDataset(Dataset):
    def __init__(self,
        basefolder: str = 'IAM/',                # Root folder
        subset: str = 'train',                   # Dataset subset to load ('train', 'val', 'test')
        fixed_size: tuple =(128, None),          # Resize inputs to this size
        transforms: list = None,                 # List of augmentation transforms to apply on input
        character_classes: list = None,          # If None, computed automatically; else list of characters
        config=None                             # Configuration object with optional parameters
        ):
        self.basefolder = basefolder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes
        self.config = config
        self.k_external_words = 0
        self.n_aligned = 0
        if self.config is not None:
            self.k_external_words = int(getattr(self.config, 'k_external_words', 0))
            self.n_aligned = int(getattr(self.config, 'n_aligned', 0))
        # Load gt.txt from basefolder - each line contains image path and transcription
        data = []
        with open(os.path.join(basefolder, subset, 'gt.txt'), 'r') as f:
            for line in f:
                img_path, transcr = line.strip().split(' ')[0], ' '.join(line.strip().split(' ')[1:])
                data += [(os.path.join(basefolder, subset, img_path + '.png'), transcr)]
        self.data = data
        # Load images into memory and store transcriptions
        # imgs = []
        transcrs = []
        for img_path, transcr in self.data:
            transcrs.append(transcr)
            # img = load_image(img_path)
            # img = preprocess(img, (self.fixed_size[0], self.fixed_size[1]))
            # img = torch.tensor(img).float().unsqueeze(0)
            # imgs.append(img)
            
        # if len(imgs) > 0:
        #     self.images = torch.stack(imgs)
        # else:
        #     self.images = torch.empty((0, 1, self.fixed_size[0], self.fixed_size[1]))
        self.transcriptions = transcrs
        # External vocabulary and probabilities
        self.external_words = []
        self.external_word_probs = []
        if self.k_external_words > 0:
            self.external_words = [w for w in top_n_list('en', self.k_external_words)] # no white spaces added
            self.external_word_probs = [word_frequency(w.strip(), 'en') for w in self.external_words]
        self.external_word_embeddings = self.find_word_embeddings(self.external_words)
        # Check if each transcription is in external vocab
        self.is_in_dict = torch.zeros(len(self.transcriptions), dtype=torch.int32)
        for i, t in enumerate(self.transcriptions):
            if t in self.external_words:
                self.is_in_dict[i] = 1
        # Alignment tensor
        self.aligned = torch.full((len(self.transcriptions),), fill_value=-1, dtype=torch.int32)
        if self.n_aligned > 0:
            dict_indices = torch.nonzero(self.is_in_dict).view(-1)
            if len(dict_indices) < self.n_aligned:
                print(f'Warning: reducing n_aligned from {self.n_aligned} to {len(dict_indices)}')
                self.n_aligned = len(dict_indices)
            if self.n_aligned > 0 and len(dict_indices) > 0:
                perm = torch.randperm(len(dict_indices))[:self.n_aligned]
                chosen = dict_indices[perm]
                for idx in chosen.tolist():
                    word = self.transcriptions[idx]
                    if word in self.external_words:
                        self.aligned[idx] = self.external_words.index(word)
                    else:
                        print(f'Warning: word {word} not found in external vocabulary')
        if self.character_classes is None:
            res = set()
            for _, transcr in data:
                res.update(list(transcr))
            res = sorted(list(res))
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            self.character_classes = res
    def __getitem__(self, index):
        img_path = self.data[index][0]
        transcr = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        img = load_image(img_path)
        if self.subset == 'train':
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
            img = resize(image=img, output_shape=(nheight, nwidth)).astype(np.float32)
        img = preprocess(img, (fheight, fwidth))
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = torch.Tensor(img).float().unsqueeze(0)
        return img, transcr, self.aligned[index]
    def __len__(self):
        return len(self.data)
    def find_word_embeddings(self, word_list):
        """Compute 2D embeddings of words using pairwise Levenshtein distances."""
        if len(word_list) == 0:
            return torch.empty((0, 2))
        n = len(word_list)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = editdistance.eval(word_list[i], word_list[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
        emb = mds.fit_transform(dist_matrix)
        return torch.FloatTensor(emb)
