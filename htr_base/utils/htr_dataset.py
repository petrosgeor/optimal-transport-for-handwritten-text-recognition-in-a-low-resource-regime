import io, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from .preprocessing import load_image, preprocess
from wordfreq import top_n_list, word_frequency
from sklearn.manifold import MDS
import editdistance
import random
from typing import List
class HTRDataset(Dataset):
    def __init__(self,
        basefolder: str = 'IAM/',                # Root folder
        subset: str = 'train',                   # Dataset subset to load ('train', 'val', 'test', 'all')
        fixed_size: tuple =(128, None),          # Resize inputs to this size
        transforms: list = None,                 # List of augmentation transforms to apply on input
        character_classes: list = None,          # If None, computed automatically; else list of characters
        config=None,                            # Configuration object with optional parameters
        two_views: bool = False,                # Whether to return two views of each image
        concat_prob: float = 0.0                # Probability of concatenating a sample with itself
        ):
        self.basefolder = basefolder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes
        self.config = config
        self.two_views = two_views
        self.concat_prob = concat_prob
        self.k_external_words = 0
        self.n_aligned = 0
        self.word_emb_dim = 512
        if self.config is not None:
            self.k_external_words = int(getattr(self.config, 'k_external_words', 0))
            self.n_aligned = int(getattr(self.config, 'n_aligned', 0))
            self.word_emb_dim = int(getattr(self.config, 'word_emb_dim', 512))
        # Load gt.txt from basefolder - each line contains image path and transcription
        if subset not in {'train', 'val', 'test', 'all'}:
            raise ValueError("subset must be 'train', 'val', 'test' or 'all'")
        data = []
        subsets = ['train', 'val', 'test'] if subset == 'all' else [subset]
        for sub in subsets:
            gt_file = os.path.join(basefolder, sub, 'gt.txt')
            with open(gt_file, 'r') as f:
                for line in f:
                    img_path, transcr = line.strip().split(' ')[0], ' '.join(line.strip().split(' ')[1:])
                    data.append((os.path.join(basefolder, sub, img_path + '.png'), transcr))
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
        self.prior_char_probs = self.letter_priors()
        if self.character_classes is None:
            res = set()
            for t in transcrs:
                res.update(list(t))
            res = sorted(list(res))
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            self.character_classes = res
        # External vocabulary and probabilities
        self.external_words = []
        self.external_word_probs = []
        if self.k_external_words > 0:
            words = [w for w in top_n_list('en', self.k_external_words)]
            words = [w.lower() for w in words]
            words = self._filter_external_words(words)
            self.external_words = words[: self.k_external_words]
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
    def __getitem__(self, index):
        img_path = self.data[index][0]
        transcr1 = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        img = load_image(img_path)
        def build_view(im):
            if self.subset in {'train', 'all'}:
                nwidth = int(np.random.uniform(.75, 1.25) * im.shape[1])
                nheight = int((np.random.uniform(.9, 1.1) * im.shape[0] / im.shape[1]) * nwidth)
                im = resize(image=im, output_shape=(nheight, nwidth)).astype(np.float32)
            im = preprocess(im, (fheight, fwidth))
            if self.transforms is not None:
                im = self.transforms(image=im)['image']
            return torch.Tensor(im).float().unsqueeze(0)
        rand = random.random()
        if rand < self.concat_prob:
            transcr2 = transcr1
            img2 = img
            if self.two_views:
                v1_a = build_view(img.copy())
                v2_a = build_view(img2.copy())
                v1_b = build_view(img.copy())
                v2_b = build_view(img2.copy())
                img_a = torch.cat([v1_a, v2_a], dim=-1)
                img_b = torch.cat([v1_b, v2_b], dim=-1)
                img_a = F.interpolate(img_a.unsqueeze(0), size=(fheight, fwidth), mode='bilinear', align_corners=False).squeeze(0)
                img_b = F.interpolate(img_b.unsqueeze(0), size=(fheight, fwidth), mode='bilinear', align_corners=False).squeeze(0)
                transcr = f" {transcr1.strip()}   {transcr1.strip()} "
                return (img_a, img_b), transcr, self.aligned[index]
            else:
                img_a = build_view(img)
                img_b = build_view(img2)
                img_cat = torch.cat([img_a, img_b], dim=-1)
                img_cat = F.interpolate(img_cat.unsqueeze(0), size=(fheight, fwidth), mode='bilinear', align_corners=False).squeeze(0)
                transcr = f" {transcr1.strip()}   {transcr1.strip()} "
                return img_cat, transcr, self.aligned[index]
        else:
            if self.two_views:
                img1 = build_view(img.copy())
                img2 = build_view(img.copy())
                return (img1, img2), transcr1, self.aligned[index]
            else:
                img_tensor = build_view(img)
                return img_tensor, transcr1, self.aligned[index]
    def __len__(self):
        return len(self.data)
    def _filter_external_words(self, words: List[str]) -> List[str]:
        """Return words containing only known dataset characters."""
        allowed = set(self.character_classes)
        return [w for w in words if all(ch in allowed for ch in w)]

    @staticmethod
    def letter_priors(transcriptions: List[str] = None, *, n_words: int = 50000):
        """Return prior probabilities for ``a-z0-9``.

        If ``transcriptions`` is ``None`` the distribution is computed from the
        ``n_words`` most common English words provided by ``wordfreq``.
        Otherwise ``transcriptions`` are used directly.
        """

        letters = "abcdefghijklmnopqrstuvwxyz0123456789"
        counts = {c: 0 for c in letters}

        if transcriptions is None:
            corpus = top_n_list("en", n_words)
        else:
            corpus = transcriptions

        for t in corpus:
            for ch in t.lower():
                if ch in counts:
                    counts[ch] += 1

        total = float(sum(counts.values()))
        if total == 0:
            return {c: 0.0 for c in letters}

        return {c: counts[c] / total for c in letters}
    def find_word_embeddings(self, word_list, n_components: int = 512):
        """Compute embeddings of words using pairwise Levenshtein distances."""
        if len(word_list) == 0:
            if n_components is None:
                n_components = self.word_emb_dim
            return torch.empty((0, n_components))
        if n_components is None:
            n_components = self.word_emb_dim
        n = len(word_list)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = editdistance.eval(word_list[i], word_list[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=0)
        emb = mds.fit_transform(dist_matrix)
        return torch.FloatTensor(emb)

    def save_image(self, index: int, out_dir: str, filename: str = None) -> str:
        """Save the preprocessed image at *index* to *out_dir* and return its path."""
        img_path = self.data[index][0]
        img = load_image(img_path)
        img = preprocess(img, (self.fixed_size[0], self.fixed_size[1]))

        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            filename = os.path.basename(img_path)
        if not filename.lower().endswith(".png"):
            filename = f"{os.path.splitext(filename)[0]}.png"

        save_path = os.path.join(out_dir, filename)
        plt.imsave(save_path, img, cmap="gray")
