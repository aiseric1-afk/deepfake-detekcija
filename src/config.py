"""Zajednicka podesavanja: putanje, uredjaj, sjeme slucajnosti.

Putanje se citaju iz varijabli okruzenja, tako da isti kod radi
lokalno, na Colabu i kod svakog clana tima:

    MPVI_DATA   gdje se cuva skup podataka      (podrazumijevano ./data)
    MPVI_CKPT   gdje se cuvaju tezine modela     (podrazumijevano ./checkpoints)
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("MPVI_DATA", ROOT / "data"))
CKPT_DIR = Path(os.environ.get("MPVI_CKPT", ROOT / "checkpoints"))
RESULTS_DIR = ROOT / "results"

for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# pretrenirane tezine (ResNet, EfficientNet) idu uz nase tezine, a ne u
# podrazumijevani kes na sistemskom disku
os.environ.setdefault("TORCH_HOME", str(CKPT_DIR / "torch_hub"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postavi_sjeme(seed: int = 42) -> None:
    """Reproducibilnost: isto sjeme -> isti rezultati."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
