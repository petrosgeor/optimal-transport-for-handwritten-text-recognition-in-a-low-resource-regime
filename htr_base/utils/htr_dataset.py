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
from htr_base.utils.vocab import load_vocab
class HTRDataset(Dataset):
    """Dataset of handwritten text images with optional alignment data.

    Args:
        basefolder (str): Root folder containing ``train/``, ``val/`` and ``test/``.
        subset (str): Portion of the dataset to load.
        fixed_size (tuple): ``(height, width)`` used to resize images.
        transforms (list | None): Optional Albumentations pipeline.
        character_classes (list | None): Characters making up the vocabulary.
        config (Any): Optional configuration object with alignment parameters.
        two_views (bool): Return two augmented views when ``True``.

    Attributes:
        basefolder (str): Root folder containing dataset splits.
        subset (str): Selected subset to load.
        fixed_size (tuple): Target image size ``(height, width)``.
        transforms (list | None): Albumentations pipeline applied to images.
        character_classes (list[str]): Dataset vocabulary of characters.
        config (Any): Optional configuration object with alignment parameters.
        two_views (bool): Whether to return two augmented views per sample.
        n_aligned (int): Number of initially aligned instances.
        word_emb_dim (int): Dimension of word embeddings.
        use_wordfreq_probs (bool): Use language-model priors instead of empirical counts.
        data (list[tuple[str, str]]): Image paths and their transcriptions.
        transcriptions (list[str]): Text strings for each image.
        prior_char_probs (dict): Prior probabilities for each character.
        unique_words (list[str]): Unique words present in the dataset.
        unique_word_probs (list[float]): Prior probability of each unique word.
        unique_word_embeddings (torch.Tensor): Embeddings for unique words.
        is_in_dict (torch.IntTensor): ``1`` if a transcription is in ``unique_words``.
        aligned (torch.IntTensor): Alignment indices or ``-1`` when unknown.
    """

    def __init__(self,
        basefolder: str = 'IAM/',                # Root folder
        subset: str = 'train',                   # Dataset subset to load ('train', 'val', 'test', 'all', 'train_val')
        fixed_size: tuple =(128, None),          # Resize inputs to this size
        transforms: list = None,                 # List of augmentation transforms to apply on input
        character_classes: list = None,          # If None, computed automatically; else list of characters
        config=None,                            # Configuration object with optional parameters
        two_views: bool = False,                # Whether to return two views of each image
        ):
        """Load handwritten text images and optional alignment info.

        Args:
            basefolder (str): Root folder containing ``train/``, ``val/`` and ``test/``.
            subset (str): Portion of the dataset to load.
            fixed_size (tuple): ``(height, width)`` used to resize images.
            transforms (list | None): Optional Albumentations pipeline.
            character_classes (list | None): Characters making up the vocabulary.
            config (Any): Optional configuration object with alignment parameters.
            two_views (bool): Return two augmented views when ``True``.
        """
        self.basefolder = basefolder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes
        self.config = config
        self.two_views = two_views
        self.n_aligned = 0
        self.use_wordfreq_probs = False
        if self.config is not None:
            self.n_aligned = int(getattr(self.config, 'n_aligned', 0))
            self.word_emb_dim = int(getattr(self.config, 'word_emb_dim', 512))
            self.use_wordfreq_probs = bool(getattr(self.config, "use_wordfreq_probs", False))
        # Load gt.txt from basefolder - each line contains image path and transcription
        if subset not in {'train', 'val', 'test', 'all', 'train_val'}:
            raise ValueError("subset must be 'train', 'val', 'test', 'all' or 'train_val'")
        data = []
        if subset == 'all':
            subsets = ['train', 'val', 'test']
        elif subset == 'train_val':
            subsets = ['train', 'val']
        else:
            subsets = [subset]
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
        self.transcriptions = transcrs
        self.prior_char_probs = self.letter_priors()
        if self.character_classes is None:
            c2i, _ = load_vocab()
            self.character_classes = list(c2i.keys())
        # Vocabulary derived from dataset transcriptions
        self.unique_words, self.unique_word_probs = self.word_frequencies()
        if self.use_wordfreq_probs:
            self.unique_word_probs = self._estimated_word_probs(self.unique_words)
        self.unique_word_embeddings = self.find_word_embeddings(self.unique_words, n_components=self.word_emb_dim)

        # All transcriptions are present in ``unique_words``
        self.is_in_dict = torch.ones(len(self.transcriptions), dtype=torch.int32)
        # Alignment tensor
        self.aligned = torch.full((len(self.transcriptions),), fill_value=-1, dtype=torch.int32)
        if self.n_aligned > 0:
            total = len(self.transcriptions)
            if total < self.n_aligned:
                print(f'Warning: reducing n_aligned from {self.n_aligned} to {total}')
                self.n_aligned = total
            if self.n_aligned > 0:
                chosen = torch.tensor(self._select_seed_indices(), dtype=torch.long)
                for idx in chosen.tolist():
                    word = self.transcriptions[idx]
                    self.aligned[idx] = self.unique_words.index(word)
    def __getitem__(self, index):
        """Return one or two processed image tensors and its transcription.

        Args:
            index (int): Index of the sample to load.

        Returns:
            tuple: ``(image, text, aligned_id)`` or ``((img1, img2), text, aligned_id)``
            when ``two_views`` is enabled.
        """
        img_path = self.data[index][0]
        transcr1 = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        img = load_image(img_path)
        def build_view(im):
            if self.subset in {'train', 'all', 'train_val'}:
                nwidth = int(np.random.uniform(.75, 1.25) * im.shape[1])
                nheight = int((np.random.uniform(.9, 1.1) * im.shape[0] / im.shape[1]) * nwidth)
                im = resize(image=im, output_shape=(nheight, nwidth)).astype(np.float32)
            im = preprocess(im, (fheight, fwidth))
            if self.transforms is not None:
                im = self.transforms(image=im)['image']
            return torch.Tensor(im).float().unsqueeze(0)
        if self.two_views:
            img1 = build_view(img.copy())
            img2 = build_view(img.copy())
            return (img1, img2), transcr1, self.aligned[index]
        else:
            img_tensor = build_view(img)
            return img_tensor, transcr1, self.aligned[index]
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def _select_seed_indices(self) -> List[int]:
        """Select random indices to seed the alignment.

        The method samples up to ``n_aligned`` distinct words uniformly at
        random from the dataset transcriptions, then returns the index of the
        first occurrence of each sampled word (in the sampling order). The
        randomness follows Python's global ``random`` module and can be made
        reproducible by seeding it externally in the caller/test.

        Returns:
            list[int]: Indices corresponding to randomly sampled distinct words.
        """

        if self.n_aligned <= 0:
            return []

        # Collect distinct words in first-occurrence order and record their
        # first dataset index. This guarantees a single seed per word.
        first_index = {}
        ordered_unique_words = []
        for i, w in enumerate(self.transcriptions):
            if w not in first_index:
                first_index[w] = i
                ordered_unique_words.append(w)

        if not ordered_unique_words:
            return []

        k = min(self.n_aligned, len(ordered_unique_words))
        sampled_words = random.sample(ordered_unique_words, k=k)
        return [first_index[w] for w in sampled_words]

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
    
    def external_word_histogram(
        self,
        *,
        save_dir: str = "tests/figures",
        filename: str = "external_word_hist.png",
        dpi: int = 200,
    ) -> None:
        """
        Plot a histogram showing how many ground-truth transcriptions
        exactly match each word in ``unique_words`` and save it under
        *tests/figures/* by default.

        Parameters
        ----------
        save_dir : str | Path, default "tests/figures"
            Directory where the PNG will be written.  Created if missing.
        filename : str, default "external_word_hist.png"
            Name of the output file inside *save_dir*.
        dpi : int, default 200
            Resolution of the saved figure.

        Notes
        -----
        • The method **returns nothing**; its side-effect is a saved PNG.
        • An exception is raised if the dataset contains no ``unique_words``.
        """
        # ── sanity check ────────────────────────────────────────────────
        if not getattr(self, "unique_words", None):
            raise RuntimeError(
                "Dataset has no unique words – nothing to plot."
            )
        # ── count matches ───────────────────────────────────────────────
        from collections import Counter
        import os, matplotlib.pyplot as plt, torch

        hits = Counter()
        for t in self.transcriptions:                  # list of GT strings
            w = t.lower()
            if w in self.unique_words:               # dataset vocab list
                hits[w] += 1
        counts = [hits.get(w, 0) for w in self.unique_words]

        # ── build & save plot ───────────────────────────────────────────
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.figure(figsize=(max(8, len(self.unique_words) * 0.25), 4))
        plt.bar(range(len(self.unique_words)), counts)
        plt.xticks(
            range(len(self.unique_words)),
            self.unique_words,
            rotation=90,
            fontsize=8,
        )
        plt.ylabel("Frequency in dataset")
        plt.title("Ground-truth matches per unique word")
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        print(f"[external_word_histogram] Figure saved to: {save_path}")

    def word_frequencies(self) -> tuple[list[str], list[float]]:
        """Return unique words and their empirical probabilities.

        Returns:
            tuple[list[str], list[float]]: First element is the list of unique
            transcriptions. The second element gives the probability of each
            corresponding word.
        """

        from collections import Counter

        words = [t.strip().lower() for t in self.transcriptions]
        counts = Counter(words)
        total = sum(counts.values())
        unique = list(counts.keys())
        probs = [counts[w] / total for w in unique]
        return unique, probs

    def _estimated_word_probs(self, words):
        """Return probabilities for *words* using the wordfreq corpus.

        The values are ``word_frequency(w, "en")``; zeros are replaced with a
        tiny epsilon and the vector is normalised to sum to 1.

        Args:
            words (list[str]): Tokens to query the language model.

        Returns:
            list[float]: Normalised prior probabilities.
        """
        eps = 1e-12
        freqs = [max(word_frequency(w, "en"), eps) for w in words]
        total = float(sum(freqs))
        return [f / total for f in freqs]


class PretrainingHTRDataset(Dataset):
    """Lightweight dataset for image-only pretraining.

    If ``n_random`` is provided, ``random_seed`` ensures the same
    subset of images is selected each time.  When ``preload_images``
    is ``True`` the raw images are cached in memory for faster
    access. Labels can be filtered by length using ``min_length``
    and ``max_length`` before random subsampling occurs.
    """

    def __init__(
        self,
        list_file: str = '/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt',
        fixed_size: tuple = (64, 256),
        base_path: str = '/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px',
        transforms: list = None,
        n_random: int = None,
        random_seed: int = 0,
        preload_images: bool = False,
    ):
        """Create a dataset from an image list for synthetic pretraining.

        Args:
            list_file (str): Path to a text file with relative image paths.
            fixed_size (tuple): ``(height, width)`` for resizing.
            base_path (str): Root directory prepended to each path in ``list_file``.
            transforms (list | None): Optional Albumentations pipeline.
            n_random (int | None): If given, keep only ``n_random`` entries.
            min_length (int): Lower bound on label length to keep.
            max_length (int): Upper bound on label length to keep.
            random_seed (int): Seed controlling the random subset selection.
            preload_images (bool): Load all images into memory on init.
        """
        self.fixed_size = fixed_size
        self.base_path = base_path
        self.transforms = transforms
        self.preload_images = preload_images


        with open(list_file, 'r') as f:
            rel_paths = [line.strip() for line in f if line.strip()]

        def _valid(p):
            desc = os.path.basename(p).split('_')[1]
            return (not desc.isupper()) and desc.isalnum()

        filtered = [p for p in rel_paths if _valid(p)]
        if n_random is not None and n_random > 0:
            rng = random.Random(random_seed)
            filtered = rng.sample(filtered, min(n_random, len(filtered)))
        self.img_paths, self.transcriptions = self.process_paths(filtered)
        if self.preload_images:
            self.images = [load_image(p) for p in self.img_paths]

    def process_paths(self, filtered_list):
        """Convert relative image paths to absolute ones and extract labels.

        Args:
            filtered_list (list[str]): Relative paths as read from ``list_file``.

        Returns:
            tuple[list[str], list[str]]: Absolute paths and lowercase labels.
        """
        full_paths = [
            os.path.normpath(os.path.join(self.base_path, p.lstrip('./')))
            for p in filtered_list
        ]
        descriptions = []
        for p in filtered_list:
            desc = os.path.basename(p).split('_')[1]
            descriptions.append(desc.lower())
        return full_paths, descriptions

    def __len__(self):
        """Dataset length."""
        return len(self.img_paths)

    def __getitem__(self, index):
        """Return a preprocessed image tensor and its transcription."""
        if hasattr(self, 'images'):
            img = self.images[index]
        else:
            img = load_image(self.img_paths[index])
        fH, fW = self.fixed_size
        # nW = int(np.random.uniform(.75, 1.25) * img.shape[1])
        # nH = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nW)
        nW = max(1, int(np.random.uniform(.75, 1.25) * img.shape[1]))
        nH = max(1, int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nW))
        img = resize(image=img, output_shape=(nH, nW)).astype(np.float32)

        img = preprocess(img, (fH, fW))
        if getattr(self, 'transforms', None):
            img = self.transforms(image=img)['image']

        img_tensor = torch.Tensor(img).float().unsqueeze(0)
        trans = f" {self.transcriptions[index]} "
        # trans = self.transcriptions[index]
        return img_tensor, trans

    def save_image(self, index: int, out_dir: str, filename: str = None) -> str:
        """Save the preprocessed image at *index* to *out_dir* and return its path."""

        img_path = self.img_paths[index]
        img = load_image(img_path)
        img = preprocess(img, (self.fixed_size[0], self.fixed_size[1]))

        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            filename = os.path.basename(img_path)
        if not filename.lower().endswith(".png"):
            filename = f"{os.path.splitext(filename)[0]}.png"

        save_path = os.path.join(out_dir, filename)
        plt.imsave(save_path, img, cmap="gray")

        return save_path

    def loaded_image_shapes(self) -> List[tuple]:
        """Return the shapes of all preloaded images.

        Raises
        ------
        RuntimeError
            If ``preload_images`` was ``False`` when the dataset was built.
        """

        if not hasattr(self, "images"):
            raise RuntimeError("the images are not loaded yet")

        return [img.shape for img in self.images]
