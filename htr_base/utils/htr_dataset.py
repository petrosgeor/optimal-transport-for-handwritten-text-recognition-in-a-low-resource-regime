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
        word_prob_strategy (str | None): One of ``'empirical'``, ``'wordfreq'`` or ``'uniform'``
            controlling how ``unique_word_probs`` are computed. Defaults to ``'empirical'``.
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
        self.word_prob_strategy = None
        if self.config is not None:
            self.n_aligned = int(getattr(self.config, 'n_aligned', 0))
            self.word_emb_dim = int(getattr(self.config, 'word_emb_dim', 512))
            # word probability strategy: 'empirical' (default), 'wordfreq', or 'uniform'
            self.word_prob_strategy = getattr(self.config, "word_prob_strategy", None)
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
            if not os.path.isfile(gt_file):
                # Allow datasets without a 'val' split (e.g., CVL) or incomplete setups.
                # Skip missing subsets gracefully.
                continue
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # The first token is the filename; remaining tokens form the transcription.
                    parts = line.split()
                    img_token = parts[0]
                    transcr = ' '.join(parts[1:])
                    # If the filename in gt does not include an extension (GW/IAM),
                    # default to '.png'. If it includes an extension (e.g., CVL '.tif'),
                    # preserve it as-is.
                    name, ext = os.path.splitext(img_token)
                    filename = img_token if ext else (img_token + '.png')
                    data.append((os.path.join(basefolder, sub, filename), transcr))
        self.data = data
        # Load images into memory and store transcriptions
        # imgs = []
        transcrs = []
        for img_path, transcr in self.data:
            transcrs.append(transcr)
        self.transcriptions = transcrs
        self.prior_char_probs = self.letter_priors()
        if self.character_classes is None:
            # Infer dataset name from basefolder and load the matching vocabulary
            dataset_name = self.get_dataset_name()
            c2i, _ = load_vocab(dataset_name)
            self.character_classes = list(c2i.keys())
        # Vocabulary derived from dataset transcriptions
        self.unique_words, self.unique_word_probs = self.word_frequencies()
        # Apply configured prior strategy for word probabilities
        _strategy = (self.word_prob_strategy or "empirical").lower()
        if _strategy == "wordfreq":
            self.unique_word_probs = self._estimated_word_probs(self.unique_words)
        elif _strategy == "uniform":
            n = len(self.unique_words)
            self.unique_word_probs = ([1.0 / n] * n) if n > 0 else []
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

    def get_dataset_name(self) -> str:
        """Infer and return the dataset identifier from ``self.basefolder``.

        Args:
            None

        Returns:
            str: The dataset name, one of ``'IAM'``, ``'GW'`` or ``'CVL'``.

        Raises:
            ValueError: If the path in ``self.basefolder`` does not contain a
            recognizable dataset token.
        """
        base = str(self.basefolder)
        low = base.lower()
        if "iam" in low:
            return "IAM"
        if "gw" in low:
            return "GW"
        if "cvl" in low:
            return "CVL"
        raise ValueError(
            f"Cannot infer dataset name from basefolder={self.basefolder!r}. "
            "Expected the path to include 'IAM', 'GW' or 'CVL'."
        )

    def get_test_indices(self) -> torch.Tensor:
        """Return indices of samples that belong to the ``test`` split.

        The dataset must have been constructed with ``subset='all'`` so that
        the ``test`` split is present. Indices are returned as a 1‑D tensor
        of type ``torch.long`` and refer to positions in ``self.data``.

        Args:
            None

        Returns:
            torch.Tensor: 1‑D tensor with the positions of all ``test`` items
            within the concatenated dataset order.

        Raises:
            RuntimeError: If the dataset was not built with ``subset='all'``.
        """

        if getattr(self, 'subset', None) != 'all':
            raise RuntimeError(
                f"Cannot retrieve test indices: dataset subset is '{self.subset}', "
                "which does not include the 'test' split. Build with subset='all'."
            )

        test_prefix = os.path.join(self.basefolder, 'test') + os.sep
        indices = [i for i, (p, _) in enumerate(self.data) if str(p).startswith(test_prefix)]
        return torch.tensor(indices, dtype=torch.long)

    def _select_seed_indices(self) -> List[int]:
        """Select random **image indices** to seed the alignment.

        Behavior
        --------
        - Returns up to ``n_aligned`` **distinct dataset indices** (no duplicates).
        - For every subset except ``'all'``: sample uniformly at random from the
        items present in *this* dataset instance (e.g., 'train', 'val', 'test',
        or 'train_val'), **without replacement**.
        - For ``subset == 'all'``: sample uniformly at random **only from the
        'train' split**, **without replacement**. Returned indices are **global**
        (positions in the concatenated [train, val, test] dataset order).

        Rationale
        ---------
        We sample at the *image* level (not at the *word* level). This allows the
        selection to include multiple images that share the same word, while still
        guaranteeing that the **same image** is never selected twice.

        Returns
        -------
        list[int]
            A list with up to ``n_aligned`` distinct dataset indices.
        """
        if self.n_aligned <= 0:
            return []

        # Build the candidate pool of indices to sample from.
        if getattr(self, "subset", None) == "all":
            # Restrict to items that *belong to the 'train' split* and return their
            # positions in the global concatenation order.
            train_prefix = os.path.join(self.basefolder, "train") + os.sep
            candidates = [
                i for i, (path, _txt) in enumerate(self.data)
                if str(path).startswith(train_prefix)
            ]
        else:
            # Sample from whatever this dataset instance currently contains.
            candidates = list(range(len(self.data)))

        if not candidates:
            return []

        # Sample *indices* uniformly and WITHOUT replacement (no duplicate images).
        k = min(self.n_aligned, len(candidates))
        return random.sample(candidates, k=k)


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
        """Embed words using Landmark (Nyström) MDS under edit distance.

        This implementation avoids the O(n^3) cost of classical MDS by:
        1) Selecting m landmark words (m = min(n, 1000)).
        2) Running classical MDS on the m×m landmark distance matrix.
        3) Embedding the remaining words out-of-sample via double-centering.

        Args:
            word_list (list[str]): Words to embed.
            n_components (int): Target embedding dimensionality. If None,
                uses self.word_emb_dim.

        Returns:
            torch.FloatTensor: Embeddings of shape (n_words, p) with
            p = min(n_components, m - 1).
        """
        if n_components is None:
            n_components = getattr(self, 'word_emb_dim', 512)
        n = len(word_list)
        if n == 0:
            return torch.empty((0, int(n_components)))
        if n == 1:
            return torch.zeros((1, int(n_components)), dtype=torch.float32)

        # 1) choose landmarks: longest words first for stability
        m = min(n, 1000)
        order = sorted(range(n), key=lambda i: (-len(word_list[i]), word_list[i]))
        import numpy as _np
        L_idx = _np.sort(_np.array(order[:m], dtype=_np.int64))
        NL_idx = _np.setdiff1d(_np.arange(n, dtype=_np.int64), L_idx, assume_unique=True)

        # 2) compute D_LL and perform classical MDS on landmarks
        D_LL = _np.zeros((m, m), dtype=_np.float64)
        for ii in range(m):
            wi = word_list[int(L_idx[ii])]
            for jj in range(ii + 1, m):
                wj = word_list[int(L_idx[jj])]
                d = editdistance.eval(wi, wj)
                D_LL[ii, jj] = D_LL[jj, ii] = float(d)

        D2_LL = D_LL ** 2
        J = _np.eye(m) - _np.ones((m, m), dtype=_np.float64) / m
        B = -0.5 * (J @ D2_LL @ J)
        from numpy.linalg import eigh as _eigh
        vals, vecs = _eigh(B)  # ascending
        p = int(min(n_components, max(1, m - 1)))
        idx_desc = _np.argsort(vals)[::-1]
        vals = vals[idx_desc]
        vecs = vecs[:, idx_desc]
        vals_clamped = _np.clip(vals[:p], a_min=0.0, a_max=None)
        L_sqrt = _np.sqrt(vals_clamped)
        X_L = vecs[:, :p] * L_sqrt[_np.newaxis, :]

        # Precompute means for out-of-sample embedding
        r = D2_LL.mean(axis=1)               # (m,)
        g = float(D2_LL.mean())              # scalar
        ones_m = _np.ones(m, dtype=_np.float64)

        X = _np.zeros((n, p), dtype=_np.float64)
        X[L_idx] = X_L
        with _np.errstate(divide='ignore', invalid='ignore'):
            invL = _np.zeros_like(L_sqrt)
            nz = L_sqrt > 0
            invL[nz] = 1.0 / L_sqrt[nz]
        Q_inv = vecs[:, :p] * invL[_np.newaxis, :]

        # 3) out-of-sample for non-landmarks
        for i in NL_idx.tolist():
            w = word_list[int(i)]
            d_iL = _np.array([editdistance.eval(w, word_list[int(j)]) for j in L_idx], dtype=_np.float64)
            d2 = d_iL ** 2
            mean_i = float(d2.mean())
            b = -0.5 * (d2 - mean_i * ones_m - r + g * ones_m)
            X[int(i)] = b @ Q_inv

        return torch.from_numpy(X.astype(_np.float32))

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
        if self.get_dataset_name() == 'GW':        
            words = [t.strip().lower() for t in self.transcriptions]
        elif self.get_dataset_name() == 'IAM':
            words = self.transcriptions
        elif self.get_dataset_name() == 'CVL':
            # Preserve case for CVL (charset includes mixed case and umlauts)
            words = self.transcriptions
        else:
            # Fallback: do not alter original transcriptions
            words = self.transcriptions
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
