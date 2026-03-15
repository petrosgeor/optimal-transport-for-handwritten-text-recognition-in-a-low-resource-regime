import io, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from .preprocessing import load_image, preprocess, is_image_corrupted
from wordfreq import top_n_list, word_frequency
from sklearn.manifold import MDS
import editdistance
import random
from typing import List, Callable
from htr_base.utils.vocab import load_vocab


class HTRDataset(Dataset):
    """Dataset of handwritten text images with optional alignment data.

    Args:
        basefolder (str): Root folder containing train/, val/ and test/.
        subset (str): Portion of the dataset to load.
        fixed_size (tuple): (height, width) used to resize images.
        transforms (Callable | None): Albumentations-style callable returning a
            dict that includes the preprocessed image under the image key.
            None disables extra augmentations.
        character_classes (list | None): Characters making up the vocabulary.
        config (Any): Configuration object with alignment parameters. This
            dataset requires a configuration; passing None raises an error.
        two_views (bool): Return two augmented views when True.

    Attributes:
        basefolder (str): Root folder containing dataset splits.
        subset (str): Selected subset to load.
        fixed_size (tuple): Target image size (height, width).
        transforms (list | None): Albumentations pipeline applied to images.
        character_classes (list[str]): Dataset vocabulary of characters.
        config (Any): Optional configuration object with alignment parameters.
        two_views (bool): Whether to return two augmented views per sample.
        n_aligned (int): Number of initially aligned instances.
        word_emb_dim (int): Dimension of word embeddings.
        word_prob_strategy (str | None): One of empirical, wordfreq or uniform
            controlling how unique_word_probs are computed. Defaults to empirical.
        data (list[tuple[str, str]]): Image paths and their transcriptions.
        transcriptions (list[str]): Text strings for each image.
        prior_char_probs (dict): Prior probabilities for each character.
        unique_words (list[str]): Unique words present in the dataset.
        unique_word_probs (list[float]): Prior probability of each unique word.
        unique_word_embeddings (torch.Tensor): Embeddings for unique words.
        is_in_dict (torch.IntTensor): 1 if a transcription is in unique_words.
        aligned (torch.IntTensor): Alignment indices or -1 when unknown.
    """

    def __init__(
        self,
        basefolder: str = "IAM/",  # Root folder
        subset: str = "train",  # Dataset subset to load ('train', 'val', 'test', 'all', 'train_val')
        fixed_size: tuple = (128, None),  # Resize inputs to this size
        transforms: Callable | None = None,  # Albumentations-style callable or None
        character_classes: list = None,  # If None, computed automatically; else list of characters
        config=None,  # Configuration object with alignment parameters
        two_views: bool = False,  # Whether to return two views of each image
    ):
        """Load handwritten text images and optional alignment info.

        Args:
            basefolder (str): Root folder containing train/, val/ and test/.
            subset (str): Portion of the dataset to load.
            fixed_size (tuple): (height, width) used to resize images.
            transforms (Callable | None): Albumentations-style callable returning
                {image: np.ndarray}; None disables extra augmentation.
            character_classes (list | None): Characters making up the vocabulary.
            config (Any): Configuration object with alignment parameters. Passing
                None is not supported.
            two_views (bool): Return two augmented views when True.
        """
        self.basefolder = basefolder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes
        self.two_views = two_views

        (
            self.config,
            self.n_aligned,
            self.word_emb_dim,
            self.word_prob_strategy,
        ) = self._parse_and_store_config(config)
        subsets = self._resolve_subsets(subset)
        self.data = self._load_gt_data(basefolder, subsets)
        self.transcriptions = self._init_transcriptions(self.data)
        assert len(self.data) == len(self.transcriptions), "data/transcriptions length mismatch"

        self.prior_char_probs = self.letter_priors()
        self.character_classes = self._init_character_classes_if_needed(self.character_classes)
        self._validate_charset_coverage(self.character_classes)
        (
            self.unique_words,
            self.unique_word_probs,
            self.unique_word_embeddings,
        ) = self._init_word_stats_and_embeddings()
        assert len(self.unique_words) == len(self.unique_word_probs), (
            "word/probability length mismatch"
        )
        self.is_in_dict, self.aligned, self.n_aligned = self._init_alignment_state(
            n_aligned=self.n_aligned
        )
        assert self.aligned.shape[0] == len(self.transcriptions), "aligned length mismatch"

    def _parse_and_store_config(self, config) -> tuple[object, int, int, object]:
        """Validate configuration and return derived init values.

        Args:
            config (Any): Configuration object with alignment-related metadata.

        Returns:
            tuple[object, int, int, object]:
                ``(config, n_aligned, word_emb_dim, word_prob_strategy)``.
        """
        if config is None:
            raise ValueError(
                "HTRDataset requires a configuration object with alignment "
                "metadata (e.g., n_aligned, word_emb_dim)."
            )
        n_aligned = int(getattr(config, "n_aligned", 0))
        word_emb_dim = int(getattr(config, "word_emb_dim", 512))
        word_prob_strategy = getattr(config, "word_prob_strategy", None)
        return config, n_aligned, word_emb_dim, word_prob_strategy

    def _resolve_subsets(self, subset: str) -> list[str]:
        """Validate subset selection and return concrete split names.

        Args:
            subset (str): Requested subset selector.

        Returns:
            list[str]: Split names to load in order.
        """
        if subset not in {"train", "val", "test", "all", "train_val"}:
            raise ValueError("subset must be 'train', 'val', 'test', 'all' or 'train_val'")
        if subset == "all":
            return ["train", "val", "test"]
        if subset == "train_val":
            return ["train", "val"]
        return [subset]

    def _load_gt_data(self, basefolder: str, subsets: list[str]) -> list[tuple[str, str]]:
        """Read ground-truth files and return image/transcription tuples.

        Args:
            basefolder (str): Root directory containing split folders.
            subsets (list[str]): Split names to scan for ``gt.txt`` files.

        Returns:
            list[tuple[str, str]]: Parsed ``(image_path, transcription)`` pairs.
        """
        data = []
        for sub in subsets:
            gt_file = os.path.join(basefolder, sub, "gt.txt")
            if not os.path.isfile(gt_file):
                # Allow datasets without a 'val' split (e.g., CVL) or incomplete setups.
                # Skip missing subsets gracefully.
                continue
            with open(gt_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    # The first token is the filename; remaining tokens form the transcription.
                    parts = line.split()
                    img_token = parts[0]
                    transcr = " ".join(parts[1:])
                    # If the filename in gt does not include an extension (GW/IAM),
                    # default to '.png'. If it includes an extension (e.g., CVL '.tif'),
                    # preserve it as-is.
                    _, ext = os.path.splitext(img_token)
                    filename = img_token if ext else (img_token + ".png")
                    data.append((os.path.join(basefolder, sub, filename), transcr))
        return data

    def _init_transcriptions(self, data: list[tuple[str, str]]) -> list[str]:
        """Build transcriptions from parsed data tuples.

        Args:
            data (list[tuple[str, str]]): Parsed ``(image_path, transcription)``
                pairs.

        Returns:
            list[str]: Ordered transcriptions matching ``data``.
        """
        return [transcr for _, transcr in data]

    def _init_character_classes_if_needed(self, character_classes: list | None) -> list[str]:
        """Resolve character classes, inferring from dataset vocab when absent.

        Args:
            character_classes (list | None): Character classes provided by the
                caller, or ``None`` to infer from dataset vocab.

        Returns:
            list[str]: Character classes to use for this dataset.
        """
        if character_classes is not None:
            return character_classes
        dataset_name = self.get_dataset_name()
        c2i, _ = load_vocab(dataset_name)
        return list(c2i.keys())

    def _validate_charset_coverage(self, character_classes: list[str]) -> None:
        """Fail fast when dataset transcriptions contain characters outside the active charset.

        Args:
            character_classes (list[str]): Active charset used to encode dataset
                transcriptions.

        Raises:
            ValueError: If one or more transcriptions contain out-of-charset
                symbols.
        """
        allowed = set(character_classes)
        unknown_chars: set[str] = set()
        offending_samples: list[tuple[int, str, list[str]]] = []

        for idx, transcr in enumerate(self.transcriptions):
            missing = sorted({ch for ch in str(transcr) if ch not in allowed})
            if not missing:
                continue
            unknown_chars.update(missing)
            if len(offending_samples) < 5:
                offending_samples.append((idx, transcr, missing))

        if not unknown_chars:
            return

        unknown_desc = ", ".join(f"{repr(ch)} (U+{ord(ch):04X})" for ch in sorted(unknown_chars))
        sample_desc = " | ".join(
            f"idx={idx}, text={text!r}, missing={missing}"
            for idx, text, missing in offending_samples
        )
        raise ValueError(
            "Dataset transcriptions contain symbols not present in the active charset. "
            f"Missing symbols: {unknown_desc}. "
            f"Sample offending items: {sample_desc}"
        )

    def _init_word_stats_and_embeddings(self) -> tuple[list[str], list[float], torch.Tensor]:
        """Build unique words, priors, and embeddings for this dataset.

        Args:
            None

        Returns:
            tuple[list[str], list[float], torch.Tensor]:
                (unique_words, unique_word_probs, unique_word_embeddings) where
                unique_word_embeddings has shape (V, D).
        """
        unique_words, unique_word_probs = self.word_frequencies()
        _strategy = (self.word_prob_strategy or "empirical").lower()
        if _strategy == "wordfreq":
            unique_word_probs = self._estimated_word_probs(unique_words)
        elif _strategy == "uniform":
            n_words = len(unique_words)
            unique_word_probs = ([1.0 / n_words] * n_words) if n_words > 0 else []
        unique_word_embeddings = self.find_word_embeddings(
            unique_words,
            n_components=self.word_emb_dim,
        )
        return unique_words, unique_word_probs, unique_word_embeddings

    def _init_alignment_state(
        self, *, n_aligned: int
    ) -> tuple[torch.IntTensor, torch.IntTensor, int]:
        """Build alignment tensors and seed aligned indices when requested.

        Args:
            n_aligned (int): Requested number of initially aligned samples.

        Returns:
            tuple[torch.IntTensor, torch.IntTensor, int]:
                (is_in_dict, aligned, effective_n_aligned) where is_in_dict and
                aligned both have shape (N,), and effective_n_aligned may be reduced when the dataset is
                smaller than the requested seed count.
        """
        is_in_dict = torch.ones(len(self.transcriptions), dtype=torch.int32)
        aligned = torch.full((len(self.transcriptions),), fill_value=-1, dtype=torch.int32)
        if n_aligned > 0:
            total = len(self.transcriptions)
            if total < n_aligned:
                print(f"Warning: reducing n_aligned from {n_aligned} to {total}")
                n_aligned = total
            if n_aligned > 0:
                chosen = torch.tensor(
                    self._select_seed_indices(n_aligned=n_aligned),
                    dtype=torch.long,
                )
                for idx in chosen.tolist():
                    word = self.transcriptions[idx]
                    aligned[idx] = self.unique_words.index(word)
        return is_in_dict, aligned, n_aligned

    def __getitem__(self, index):
        """Return one or two processed image tensors and its transcription.

        Args:
            index (int): Index of the sample to load.

        Returns:
            tuple: (image, text, aligned_id) or ((img1, img2), text, aligned_id)
            when two_views is enabled.

        Notes:
            Transcriptions are returned in canonical form with boundary
            whitespace removed.
        """
        img_path = self.data[index][0]
        transcr1 = str(self.data[index][1]).strip()
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        img = load_image(img_path)

        def build_view(im):
            """Build one processed tensor view from a grayscale image.

            Args:
                im (np.ndarray): Input image array with shape (H, W).

            Returns:
                torch.Tensor: Processed image tensor with shape (1, fheight, fwidth).
            """
            if self.transforms is not None:
                nwidth = int(np.random.uniform(0.75, 1.25) * im.shape[1])
                nheight = int((np.random.uniform(0.9, 1.1) * im.shape[0] / im.shape[1]) * nwidth)
                im = resize(image=im, output_shape=(nheight, nwidth)).astype(np.float32)
            im = preprocess(im, (fheight, fwidth))
            if self.transforms is not None:
                im = self.transforms(image=im)["image"]
            return torch.Tensor(im).float().unsqueeze(0)

        if self.two_views:
            img1 = build_view(img.copy())
            img2 = build_view(img.copy())
            return (img1, img2), transcr1, self.aligned[index]
        else:
            img_tensor = build_view(img)
            return img_tensor, transcr1, self.aligned[index]

    def __len__(self):
        """Return the number of items in the dataset.

        Returns:
            int: Number of dataset samples.
        """
        return len(self.data)

    def get_dataset_name(self) -> str:
        """Infer and return the dataset identifier from self.basefolder.

        Args:
            None

        Returns:
            str: The dataset name, one of IAM, GW, CVL or BESSARION.

        Raises:
            ValueError: If the path in self.basefolder does not contain a
            recognizable dataset token.

        Notes:
            This mapping is intentionally hard-coded to the supported
            datasets and should remain as such unless the rest of the pipeline
            grows support for additional corpora.
        """
        base = str(self.basefolder)
        low = base.lower()
        if "iam" in low:
            return "IAM"
        if "gw" in low:
            return "GW"
        if "cvl" in low:
            return "CVL"
        if "bessarion" in low:
            return "BESSARION"
        raise ValueError(
            f"Cannot infer dataset name from basefolder={self.basefolder!r}. "
            "Expected the path to include 'IAM', 'GW', 'CVL' or 'BESSARION'."
        )

    def get_test_indices(self) -> torch.Tensor:
        """Return indices of samples that belong to the test split.

        The dataset must have been constructed with subset=all so that
        the test split is present. Indices are returned as a 1‑D tensor
        of type torch.long and refer to positions in self.data.

        Args:
            None

        Returns:
            torch.Tensor: 1‑D tensor with shape (K,) containing positions of
            test items in the concatenated dataset order.

        Raises:
            RuntimeError: If the dataset was not built with subset=all.
        """

        if getattr(self, "subset", None) != "all":
            raise RuntimeError(
                f"Cannot retrieve test indices: dataset subset is '{self.subset}', "
                "which does not include the 'test' split. Build with subset='all'."
            )

        test_prefix = os.path.join(self.basefolder, "test") + os.sep
        indices = [i for i, (p, _) in enumerate(self.data) if str(p).startswith(test_prefix)]
        return torch.tensor(indices, dtype=torch.long)

    def _select_seed_indices(self, n_aligned: int | None = None) -> List[int]:
        """Return distinct dataset indices used as initial alignment seeds.

        Sampling is performed at the image level (not the word level), so the
        result may include different images sharing the same transcription, but
        never the same image index twice.

        Behavior:
        - If ``n_aligned`` is ``None``, uses ``self.n_aligned``.
        - If the requested count is non-positive, returns an empty list.
        - For ``subset != "all"``, samples uniformly from all indices in the
          current dataset instance, without replacement.
        - For ``subset == "all"``, samples uniformly from train-split indices
          only (global positions in the concatenated train/val/test order),
          without replacement.
        - The output size is ``min(requested_count, number_of_candidates)``.

        Args:
            n_aligned (int | None): Number of indices to sample.

        Returns:
            list[int]: Distinct sampled indices. Returns an empty list when no
            candidates are available or the requested count is non-positive.
        """
        if n_aligned is None:
            n_aligned = self.n_aligned
        if n_aligned <= 0:
            return []

        # Build the candidate pool of indices to sample from.
        if getattr(self, "subset", None) == "all":
            # Restrict to items that *belong to the 'train' split* and return their
            # positions in the global concatenation order.
            train_prefix = os.path.join(self.basefolder, "train") + os.sep
            candidates = [
                i for i, (path, _txt) in enumerate(self.data) if str(path).startswith(train_prefix)
            ]
        else:
            # Sample from whatever this dataset instance currently contains.
            candidates = list(range(len(self.data)))

        if not candidates:
            return []

        # Sample *indices* uniformly and WITHOUT replacement (no duplicate images).
        k = min(n_aligned, len(candidates))
        return random.sample(candidates, k=k)

    @staticmethod
    def letter_priors(transcriptions: List[str] = None, *, n_words: int = 50000):
        """Return prior probabilities for a-z0-9.

        If transcriptions is None the distribution is computed from the
        n_words most common English words provided by wordfreq.
        Otherwise transcriptions are used directly.

        Args:
            transcriptions (List[str] | None): Optional corpus words used to
                estimate priors. If None, wordfreq top-n words are used.
            n_words (int): Number of wordfreq words used when transcriptions is None.

        Returns:
            dict[str, float]: Prior probabilities for alphanumeric characters.
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
            torch.FloatTensor: Embeddings of shape (n_words, n_components).
            When the landmark MDS rank is smaller than n_components, the
            trailing dimensions are zero-padded.
        """
        if n_components is None:
            n_components = getattr(self, "word_emb_dim", 512)
        target_dim = int(n_components)
        n = len(word_list)
        if n == 0:
            return torch.empty((0, target_dim))
        if n == 1:
            return torch.zeros((1, target_dim), dtype=torch.float32)

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

        D2_LL = D_LL**2
        J = _np.eye(m) - _np.ones((m, m), dtype=_np.float64) / m
        B = -0.5 * (J @ D2_LL @ J)
        from numpy.linalg import eigh as _eigh

        vals, vecs = _eigh(B)  # ascending
        p = int(min(target_dim, max(1, m - 1)))
        idx_desc = _np.argsort(vals)[::-1]
        vals = vals[idx_desc]
        vecs = vecs[:, idx_desc]
        vals_clamped = _np.clip(vals[:p], a_min=0.0, a_max=None)
        L_sqrt = _np.sqrt(vals_clamped)
        X_L = vecs[:, :p] * L_sqrt[_np.newaxis, :]

        # Precompute means for out-of-sample embedding
        r = D2_LL.mean(axis=1)  # (m,)
        g = float(D2_LL.mean())  # scalar
        ones_m = _np.ones(m, dtype=_np.float64)

        X = _np.zeros((n, target_dim), dtype=_np.float64)
        X[L_idx, :p] = X_L
        with _np.errstate(divide="ignore", invalid="ignore"):
            invL = _np.zeros_like(L_sqrt)
            nz = L_sqrt > 0
            invL[nz] = 1.0 / L_sqrt[nz]
        Q_inv = vecs[:, :p] * invL[_np.newaxis, :]

        # 3) out-of-sample for non-landmarks
        for i in NL_idx.tolist():
            w = word_list[int(i)]
            d_iL = _np.array(
                [editdistance.eval(w, word_list[int(j)]) for j in L_idx], dtype=_np.float64
            )
            d2 = d_iL**2
            mean_i = float(d2.mean())
            b = -0.5 * (d2 - mean_i * ones_m - r + g * ones_m)
            X[int(i), :p] = b @ Q_inv

        return torch.from_numpy(X.astype(_np.float32))

    def save_image(self, index: int, out_dir: str, filename: str = None) -> str:
        """Save the preprocessed image at index to disk.

        Args:
            index (int): Dataset index to export.
            out_dir (str): Output directory where the image is written.
            filename (str | None): Optional output filename. If None, derives
                filename from the source image path.
        """
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
        exactly match each word in unique_words and save it under
        *tests/figures/* by default.

        Args:
            save_dir (str): Directory where the PNG is written.
            filename (str): Name of the output image file.
            dpi (int): Output figure DPI.

        Notes:
        • The method **returns nothing**; its side-effect is a saved PNG.
        • An exception is raised if the dataset contains no unique_words.
        """
        # ── sanity check ────────────────────────────────────────────────
        if not getattr(self, "unique_words", None):
            raise RuntimeError("Dataset has no unique words – nothing to plot.")
        # ── count matches ───────────────────────────────────────────────
        from collections import Counter
        import os, matplotlib.pyplot as plt, torch

        hits = Counter()
        for t in self.transcriptions:  # list of GT strings
            w = t.lower()
            if w in self.unique_words:  # dataset vocab list
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

        if self.get_dataset_name() == "GW":
            words = [t.strip().lower() for t in self.transcriptions]
        elif self.get_dataset_name() == "IAM":
            words = self.transcriptions
        elif self.get_dataset_name() == "CVL":
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

        The values are word_frequency(w, en); zeros are replaced with a
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

    Selection is deterministic for a fixed list_file, filesystem, and
    random_seed: the candidate list is first filtered by filename validity,
    then deterministically shuffled and scanned left-to-right. Only entries
    that actually exist on disk are kept. When n_random is provided, the
    dataset contains exactly n_random existing images; otherwise all
    existing images from the filtered list are used. If the filesystem cannot
    supply n_random existing files, a RuntimeError is raised with a
    concise summary.

    If preload_images is True, images are loaded into memory. Should a
    corrupted file cause a read failure during preloading, the dataset attempts
    to replace it by scanning the remaining shuffled candidates; if no
    replacement can be found while satisfying n_random, a RuntimeError
    is raised.
    """

    def __init__(
        self,
        list_file: str = "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt",
        fixed_size: tuple = (64, 256),
        base_path: str = "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px",
        transforms: list = None,
        n_random: int = None,
        random_seed: int = 0,
        preload_images: bool = False,
    ):
        """Create a dataset from an image list for synthetic pretraining.

        Args:
            list_file (str): Path to a text file with relative image paths.
            fixed_size (tuple): (height, width) for resizing.
            base_path (str): Root directory prepended to each path in list_file.
            transforms (list | None): Optional Albumentations pipeline.
            n_random (int | None): If given, keep exactly n_random existing entries.
            random_seed (int): Seed controlling the deterministic shuffle.
            preload_images (bool): Load all images into memory on init.

        Raises:
            RuntimeError: When n_random is set and fewer existing images are
                available than requested, or when preloading cannot satisfy the
                requested count due to unreadable/corrupted files.
        """
        self.fixed_size = fixed_size
        self.base_path = base_path
        self.transforms = transforms
        self.preload_images = preload_images

        # 1) Build candidate list from file, preserving current validity filter
        with open(list_file, "r") as f:
            rel_paths = [line.strip() for line in f if line.strip()]

        def _valid(p: str) -> bool:
            """Return whether a relative path passes filename heuristics.

            Args:
                p (str): Relative path candidate from list_file.

            Returns:
                bool: True when the encoded descriptor token is accepted.
            """
            try:
                desc = os.path.basename(p).split("_")[1]
            except Exception:
                return False
            return (not desc.isupper()) and desc.isalnum()

        filtered = [p for p in rel_paths if _valid(p)]

        # Deterministic permutation using the provided seed
        rng = random.Random(random_seed)
        rng.shuffle(filtered)

        # 2) Scan permuted list, keep first n_random existing absolute paths
        want_n = n_random if (n_random is not None and n_random > 0) else None
        keep_paths: List[str] = []
        keep_labels: List[str] = []

        # Helper to convert one relative path to absolute path and label
        def _to_abs_and_label(rel: str) -> tuple[str, str]:
            """Convert a relative dataset path to absolute path and label.

            Args:
                rel (str): Relative path from list_file.

            Returns:
                tuple[str, str]: Absolute path and lowercase label string.
            """
            abs_path = os.path.normpath(os.path.join(self.base_path, rel.lstrip("./")))
            desc = os.path.basename(rel).split("_")[1].lower()
            return abs_path, desc

        # Gather from shuffled candidates
        for rel in filtered:
            abs_path, label = _to_abs_and_label(rel)
            # Keep only files that exist and are not corrupted (header check)
            if os.path.isfile(abs_path) and not is_image_corrupted(abs_path, fast=False):
                keep_paths.append(abs_path)
                keep_labels.append(label)
                if want_n is not None and len(keep_paths) >= want_n:
                    break

        if want_n is not None and len(keep_paths) < want_n:
            msg = (
                f"Requested n_random={want_n}, but only {len(keep_paths)} existing images "
                f"were found after filtering (listed={len(rel_paths)}, valid={len(filtered)}). "
                "Re-download the dataset or lower n_random."
            )
            raise RuntimeError(msg)

        # Persist selected items
        self.img_paths = keep_paths
        self.transcriptions = keep_labels

        # Optional 5) Robust preloading with fallback to remaining candidates
        if self.preload_images:
            self.images = []
            kept_set = set(self.img_paths)
            # Iterator over remaining candidates (beyond those already chosen)
            remaining_iter = (
                rel
                for rel in filtered
                if os.path.normpath(os.path.join(self.base_path, rel.lstrip("./"))) not in kept_set
            )

            def _next_replacement():
                """Return next valid replacement image candidate.

                Args:
                    None

                Returns:
                    tuple[str, str, np.ndarray] | None: Absolute path, label, and
                        loaded image array with shape (H, W), or None if exhausted.
                """
                for rel in remaining_iter:
                    abs_path, label = _to_abs_and_label(rel)
                    if os.path.isfile(abs_path):
                        try:
                            img = load_image(abs_path)
                        except Exception:
                            continue
                        return abs_path, label, img
                return None

            for i, p in enumerate(list(self.img_paths)):
                try:
                    img = load_image(p)
                    self.images.append(img)
                except Exception:
                    repl = _next_replacement()
                    if repl is None:
                        # Not enough replacements; enforce the invariant when n_random was requested
                        want = want_n if want_n is not None else len(self.img_paths)
                        raise RuntimeError(
                            f"Failed to preload image at index {i} and no replacement available. "
                            f"Requested {want} images but could not preload that many."
                        )
                    abs_path, label, img = repl
                    # Replace entry i to keep counts and labels consistent
                    self.img_paths[i] = abs_path
                    self.transcriptions[i] = label
                    self.images.append(img)

        # 6) Log a concise summary of selection
        total_listed = len(rel_paths)
        total_valid = len(filtered)
        total_kept = len(self.img_paths)
        print(
            f"[PretrainingHTRDataset] listed={total_listed} valid={total_valid} kept={total_kept} "
            f"(n_random={n_random if n_random is not None else 'all'}; preload={self.preload_images})"
        )

    def process_paths(self, filtered_list):
        """Convert relative image paths to absolute ones and extract labels.

        Args:
            filtered_list (list[str]): Relative paths as read from list_file.

        Returns:
            tuple[list[str], list[str]]: Absolute paths and lowercase labels.
        """
        full_paths = [
            os.path.normpath(os.path.join(self.base_path, p.lstrip("./"))) for p in filtered_list
        ]
        descriptions = []
        for p in filtered_list:
            desc = os.path.basename(p).split("_")[1]
            descriptions.append(desc.lower())
        return full_paths, descriptions

    def __len__(self):
        """Return the number of selected pretraining images.

        Returns:
            int: Number of entries in img_paths.
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """Return one preprocessed pretraining sample.

        Args:
            index (int): Sample index in [0, len(self) - 1].

        Returns:
            tuple[torch.Tensor, str]: Image tensor with shape (1, fH, fW) and
                the corresponding transcription string.
        """
        if hasattr(self, "images"):
            img = self.images[index]
        else:
            img = load_image(self.img_paths[index])
        fH, fW = self.fixed_size
        # nW = int(np.random.uniform(.75, 1.25) * img.shape[1])
        # nH = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nW)
        nW = max(1, int(np.random.uniform(0.75, 1.25) * img.shape[1]))
        nH = max(1, int((np.random.uniform(0.9, 1.1) * img.shape[0] / img.shape[1]) * nW))
        img = resize(image=img, output_shape=(nH, nW)).astype(np.float32)

        img = preprocess(img, (fH, fW))
        if getattr(self, "transforms", None):
            img = self.transforms(image=img)["image"]

        img_tensor = torch.Tensor(img).float().unsqueeze(0)
        trans = str(self.transcriptions[index]).strip()
        return img_tensor, trans

    def save_image(self, index: int, out_dir: str, filename: str = None) -> str:
        """Save one preprocessed pretraining image to disk.

        Args:
            index (int): Dataset index to export.
            out_dir (str): Output directory where the image is written.
            filename (str | None): Optional output filename. If None, derives
                filename from the source path.

        Returns:
            str: Path to the saved PNG image.
        """

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

        Returns:
            List[tuple]: Shape list where each element is (H, W).

        Raises:
            RuntimeError: If preload_images was False when the dataset was built.
        """

        if not hasattr(self, "images"):
            raise RuntimeError("the images are not loaded yet")

        return [img.shape for img in self.images]
