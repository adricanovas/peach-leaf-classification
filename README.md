# Quantifying Domain Shift in Peach Leaf Damage Classification

Code, orchard data, data partitions and per-leaf predictions of the article *"Quantifying Domain Shift
in Peach Leaf Damage Classification: An Evaluation of Attention Mechanisms and Fine-Tuning
Strategies"*, submitted to Applied Sciences (MDPI), under review.

Authors: Adrián Cánovas-Rodríguez, Miguel A. González-Illán, Maria Fernanda García-Cruz,
Pedro Nortes Tortosa, José Salvador Rubio-Asensio, Miguel A. Zamora-Izquierdo,
Juan Antonio Martínez Navarro and Antonio F. Skarmeta.

Eleven ImageNet-pretrained CNNs (MobileNetV2, MobileNetV3Large, EfficientNetB0,
DenseNet121, EfficientNetB3, ResNet50, InceptionV3, VGG16, VGG19, EfficientNetB5,
ResNet101), each with and without a terminal-stage CBAM, are trained on 1415 public
peach-leaf crops (six classes) and evaluated and adapted on 182 leaves from a commercial
orchard (four classes). Input size is 224 for all models. All splits are grouped by parent
photograph.

## Data

- Public data (not included in this repository): 1415 leaf crops from 372 photographs,
  taken from two CC BY 4.0 datasets: 1337 leaves / 346 photos from the Mendeley dataset
  (Ahmad, J.; Farman, H. *Dataset: Efficientnet-based robust recognition of peach plant
  diseases in field images*. Mendeley Data, V1, 2024.
  <https://data.mendeley.com/datasets/3pmj85snvw/1>) and 78 leaves / 26 photos from
  PlantDoc (Singh, D. et al. *PlantDoc: A Dataset for Visual Plant Disease Detection*.
  CoDS-COMAD 2020, pp. 249–253). Training expects the crops in
  `data/public_dataset/public_data/<class>/` with the file names listed in
  `splits/manifest.csv`. The released predictions and the result tables do not need them;
  training, `peachleaf.domain_stats` and the example-leaves figure do.
- `data/merged_dataset/local_data/<class>/`: 182 orchard leaf crops from 122 photographs
  (Jumilla, Murcia, Spain; campaigns 2024 and 2025). Also archived, with the original
  photographs and the COCO polygon masks, on Zenodo: <https://doi.org/10.5281/zenodo.20446064>.

| Class | Public | Orchard |
|---|---:|---:|
| abiotic_stress | 118 | 24 |
| bacterial_spot | 180 | 0 |
| chewing_insect | 45 | 7 |
| healthy | 951 | 107 |
| mechanical_stress | 58 | 44 |
| mite_presence | 63 | 0 |

## Layout

| Path | Content |
|---|---|
| `data/` | orchard leaf crops (see above) |
| `splits/` | `manifest.csv` (one row per leaf: path, domain, class, parent photograph, campaign, source), `splits.json` (every partition, seed 2026), `split_summary.csv` |
| `peachleaf/` | Python package: configuration (`config.py`), data loading and augmentation, models, training, experiment stages (`run.py`), statistics and analysis |
| `predictions/` | per-leaf softmax outputs of every trained model, `<stage>/<backbone>__<variant>/r<repeat>_f<fold>/preds.npz` with its `meta.json` (arguments, training history, versions); `cost/` holds the cost measurements. Embeddings and checkpoints are not included |
| `reports/` | result tables (CSV) produced by the analysis; `reports/figures/` has the figures of the paper |
| `notebooks/results_analysis.ipynb` | draws the figures of the paper and of the Supplementary Materials from `reports/` |
| `figure_sources/` | TikZ source of the CBAM diagram |

## Protocol

- Public data: photo-grouped hold-out test (283 leaves / 73 photos) and photo-grouped
  5-fold cross-validation repeated 3 times on the remaining development set (15 models
  per architecture).
- Orchard data: zero-shot evaluation on all 182 leaves; adaptation with photo-grouped
  5-fold CV repeated 3 times, with an inner grouped validation split for early stopping and
  strategy selection, so test folds are never used for any decision.
- The six-way softmax is kept everywhere; the two classes absent from the orchard are
  masked at evaluation.
- Every unit stores per-leaf probabilities, so metrics, bootstrap confidence intervals and
  paired tests are computed afterwards without retraining.

## Installation

Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Versions are pinned (TensorFlow 2.20, Keras 3.15). Experiments ran on NVIDIA L40S GPUs.

## Reproducing the tables and figures

From the released predictions, without retraining:

```bash
PEACH_RUNS_ROOT=predictions python -m peachleaf.analysis                 # result CSVs -> reports/
jupyter nbconvert --execute --inplace notebooks/results_analysis.ipynb   # figures -> reports/figures/
python -m peachleaf.domain_stats                                         # image statistics of the domain shift (CPU)
```

## Running the experiments

```bash
python -m peachleaf.splits                 # regenerate manifest and splits (deterministic)
python -m peachleaf.run source --backbone efficientnetb0 --variant cbam --repeat 1 --fold 1
python -m peachleaf.run adapt  --backbone efficientnetb0 --variant cbam --repeat 1 --fold 1
python -m peachleaf.tasks --list           # pending/total units of every task group
```

| Stage | Description |
|---|---|
| `source` | training on the public data (base and CBAM; ablation variants `se`, `cbam_channel`, `cbam_spatial`, `cbam_2stage` on three backbones) |
| `adabn` | label-free BatchNorm statistics alignment on orchard images |
| `adapt` | adaptation with orchard labels: linear probing (`lp`), fine-tuning of the last block (`ftl`), fine-tuning of all layers (`fta`) and LP-FT (`lpft`); WiSE-FT interpolations are also stored but not reported in the article |
| `scratch` | ImageNet → orchard only (no public data) |
| `joint` | joint training on public and orchard data |
| `curve` | learning curve versus number of labelled orchard leaves |
| `cost` | parameters, FLOPs, latency and memory |

Each unit writes `runs/<stage>/<backbone>__<variant>/r<repeat>_f<fold>/` with `preds.npz`,
`meta.json` and `DONE`. `PEACH_SMOKE=1` runs every stage with one epoch.
`python -m peachleaf.tasks` writes the pending units of each group as text files (one
`peachleaf.run` argument line per unit) that can be fed to any job scheduler; units that already have a `DONE` file are skipped.

## Licence

Code: MIT (`LICENSE`). Orchard dataset: see the Zenodo record. The public images keep the
CC BY 4.0 licence of their original datasets.

## Citation

If you use this code or data, please cite the article (the reference will be added upon
publication) and the original public datasets.
