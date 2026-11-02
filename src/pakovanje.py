"""Pakovanje svih pripremljenih slika u jedan fajl.

Zasto: na tvrdom disku (HDD) nasumicno citanje sitnih fajlova ide oko
70 fajlova/s, jer glava diska trazi svaki fajl posebno. Trening cita
slike nasumicnim redom, pa bi disk bio usko grlo.

Rjesenje: sve JPEG slike se nadovezu u jedan fajl 'slike.bin', a
'pomaci.npy' cuva gdje pocinje i koliko je dug svaki zapis. Jedan
veliki fajl se cita redom, a poslije prvog citanja operativni sistem
ga drzi u RAM-u.

Redoslijed zapisa je ISTI kao redoslijed redova u indeks.csv.

Pokretanje:  python -m src.pakovanje
"""
import csv

import numpy as np
from tqdm import tqdm

from .config import DATA_DIR

KORIJEN = DATA_DIR / "dff256"


def main():
    with open(KORIJEN / "indeks.csv", encoding="utf-8") as f:
        putanje = [r["putanja"] for r in csv.DictReader(f)]

    pomaci = np.zeros((len(putanje), 2), dtype=np.int64)     # (pocetak, duzina)
    pozicija = 0
    with open(KORIJEN / "slike.bin", "wb") as izlaz:
        for i, p in enumerate(tqdm(putanje, desc="pakovanje")):
            with open(KORIJEN / p, "rb") as f:
                b = f.read()
            izlaz.write(b)
            pomaci[i] = (pozicija, len(b))
            pozicija += len(b)
    np.save(KORIJEN / "pomaci.npy", pomaci)
    print(f"spakovano {len(putanje)} slika, {pozicija / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
