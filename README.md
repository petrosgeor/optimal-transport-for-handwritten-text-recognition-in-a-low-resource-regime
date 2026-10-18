# Optimal Transport for Handwritten Text Recognition in a Low-Resource Regime

Petros Georgoulas Wraight, Giorgos Sfikas, Ioannis Kordonis, Petros Maragos, George Retsinas

EUSIPCO 2026 · [arXiv:2509.16977](https://arxiv.org/abs/2509.16977)

Code for the paper. We start from a handful of labeled word images and a lexicon with word-frequency priors. Each round, we fine-tune a CTC recognizer on the labeled words, project its features into a word-embedding space, and use optimal transport to match unlabeled images to lexicon words. The most confident matches become pseudo-labels for the next round.

## Install

```bash
git clone https://github.com/petrosgeor/optimal-transport-for-handwritten-text-recognition-in-a-low-resource-regime.git
cd optimal-transport-for-handwritten-text-recognition-in-a-low-resource-regime
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The main training loop needs a CUDA GPU.

## Data

Each dataset goes under `htr_base/data/<DATASET>/processed_words/{train,val,test}/`. Every split folder holds the word images and a `gt.txt` with one `<image_id> <transcription>` per line. The prep scripts build this from the raw datasets:

```bash
python htr_base/prepare_gw.py --dataset_folder /path/to/washingtondb-v1.0 --output_path htr_base/data/GW/processed_words
python htr_base/prepare_iam.py --iam-root /path/to/IAM
python htr_base/prepare_cvl.py --cvl-root /path/to/CVL
```

## Pretrain the backbone

The recognizer is first pretrained on synthetic words (MJSynth, `90kDICT32px`). Set `list_file` in `alignment/alignment_configs/pretraining_config.yaml` to the dataset's `imlist.txt`, then run:

```bash
python -m alignment.pretraining
```

Checkpoints go to `saved_models/pretrained_models/<save_path>/`.

## Run

```bash
python -m alignment.trainer --config alignment/alignment_configs/trainer_config.yaml
```

The config sets the dataset (`dataset.basefolder`), the number of starting labels (`dataset.n_aligned`), and the pretrained checkpoint (`pretrained_backbone_path`). After each round the trainer adds a line to `round_metrics_file`: round, correct pseudo-labels, test CER (as a fraction), test WER (in %).

## Citation

```bibtex
@article{georgoulaswraight2025optimal,
  title   = {Optimal Transport for Handwritten Text Recognition in a Low-Resource Regime},
  author  = {Georgoulas Wraight, Petros and Sfikas, Giorgos and Kordonis, Ioannis and Maragos, Petros and Retsinas, George},
  journal = {arXiv preprint arXiv:2509.16977},
  year    = {2025}
}
```
