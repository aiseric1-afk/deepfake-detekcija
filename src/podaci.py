"""Ucitavanje pripremljenih slika (256x256) prema indeksu iz faze M1.

Skup za jedan eksperiment = stvarne slike (wiki) + izabrani dijelovi
laznih slika, iz jednog skupa podjele (trening / validacija / test).
Koriste se samo redovi oznaceni kao 'zadrzano' — insight slike cija je
pozadina u drugom skupu su izbacene da ne bi bilo curenja.
"""
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import DATA_DIR

KORIJEN = DATA_DIR / "dff256"
PAKET = KORIJEN / "slike.bin"            # sve slike u jednom fajlu (src/pakovanje.py)
POMACI = KORIJEN / "pomaci.npy"
LAZNI_DIJELOVI = ["insight", "inpainting", "text2img"]


def ucitaj_indeks() -> pd.DataFrame:
    df = pd.read_csv(KORIJEN / "indeks.csv",
                     dtype={"pozadina_osoba": str, "pozadina_udaljenost": str, "razlog": str})
    df["red"] = np.arange(len(df))       # pozicija zapisa u paketu
    return df[df["zadrzano"] == 1].reset_index(drop=True)


def zagrij_kes(blok=64 * 2**20):
    """Procita paket redom jednom, pa ga operativni sistem drzi u RAM-u.
    Na HDD-u je to razlika izmedju ~70 i nekoliko stotina slika u sekundi."""
    if not PAKET.exists():
        return
    with open(PAKET, "rb") as f:
        while f.read(blok):
            pass


def izaberi(indeks, split, lazni, n_po_klasi=None, seed=0):
    """Redovi za jedan skup: stvarne + izabrani lazni dijelovi.

    n_po_klasi: ako je zadano, svaka klasa se svede na tacno toliko slika,
    da bi eksperimenti s razlicitim metodama bili uporedivi.
    """
    s = indeks[indeks["split"] == split]
    stvarne = s[s["dio"] == "wiki"]
    lazne = s[s["dio"].isin(lazni)]
    if n_po_klasi is not None:
        rng = np.random.default_rng(seed)
        def svedi(df):
            if len(df) <= n_po_klasi:
                return df
            return df.iloc[rng.choice(len(df), n_po_klasi, replace=False)]
        stvarne, lazne = svedi(stvarne), svedi(lazne)
    return pd.concat([stvarne, lazne]).reset_index(drop=True)


class SkupLica(Dataset):
    """Cita iz paketa ako postoji, inace iz pojedinacnih fajlova."""

    def __init__(self, redovi: pd.DataFrame, transformacija):
        self.putanje = redovi["putanja"].tolist()
        self.oznake = redovi["oznaka"].astype(np.int64).tolist()
        self.t = transformacija
        self.paket = PAKET.exists() and POMACI.exists()
        if self.paket:
            self.pomaci = np.load(POMACI)[redovi["red"].to_numpy()]
        self._mm = None                  # otvara se posebno u svakom procesu

    def __len__(self):
        return len(self.oznake)

    def _bajtovi(self, i):
        if self._mm is None:
            self._mm = np.memmap(PAKET, dtype=np.uint8, mode="r")
        pocetak, duzina = self.pomaci[i]
        return self._mm[pocetak:pocetak + duzina].tobytes()

    def __getitem__(self, i):
        izvor = io.BytesIO(self._bajtovi(i)) if self.paket else KORIJEN / self.putanje[i]
        with Image.open(izvor) as im:
            x = self.t(im.convert("RGB"))
        return x, self.oznake[i]


def transformacije(velicina=224, trening=True):
    """Slike su sacuvane kao 256x256; uzima se isjecak velicine 'velicina'."""
    skala = int(round(velicina * 256 / 224))
    if trening:
        return T.Compose([
            T.Resize(skala), T.RandomCrop(velicina),
            T.RandomHorizontalFlip(),          # lice ogledalno je i dalje lice
            T.ToTensor(),                       # [0, 1]; normalizacija je u modelu
        ])
    return T.Compose([T.Resize(skala), T.CenterCrop(velicina), T.ToTensor()])


def loader(redovi, velicina=224, trening=True, batch=64, radnika=3):
    return DataLoader(
        SkupLica(redovi, transformacije(velicina, trening)),
        batch_size=batch, shuffle=trening, num_workers=radnika,
        pin_memory=torch.cuda.is_available(), persistent_workers=radnika > 0,
        drop_last=trening,
    )
