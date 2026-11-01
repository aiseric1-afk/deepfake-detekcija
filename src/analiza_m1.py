"""Faza M1 — analiza podataka prije treniranja.

Mjeri dvije stvari koje bi omogucile "precicu" umjesto ucenja laziranja:
  1. udio (skoro) crno-bijelih slika po dijelovima skupa
  2. da li insight slike koriste STVARNE fotografije iz wiki dijela kao
     pozadinu (provjera perceptualnim hesom)

Pokretanje:  python -m src.analiza_m1
"""
import io
import json
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import DATA_DIR, RESULTS_DIR

DIO = ["wiki", "insight", "inpainting", "text2img"]
IZVOR = DATA_DIR / "deepfakeface"


def otvori(z, ime):
    return Image.open(io.BytesIO(z.read(ime))).convert("RGB")


def je_sivo(im, prag=4.0):
    """Skoro crno-bijela: kanali R, G, B se gotovo ne razlikuju."""
    a = np.asarray(im.resize((64, 64)), dtype=np.float32)
    return float(np.abs(a - a.mean(axis=2, keepdims=True)).mean()) < prag


def dhash(im, velicina=8):
    """64-bitni razlikovni hes; slicne slike daju slicne heseve.
    Slika se prvo RAZVUCE na kvadrat, kao sto su to radili autori skupa."""
    g = np.asarray(im.convert("L").resize((velicina + 1, velicina)), dtype=np.int16)
    bitovi = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bitovi)            # 8 bajtova


def hamming(a, B):
    """Hammingova udaljenost jednog hesa a do svih heseva u B."""
    return np.unpackbits(np.bitwise_xor(B, a), axis=1).sum(axis=1)


def main(uzorak_sivo=3000, uzorak_insight=1000, seed=0):
    rng = np.random.default_rng(seed)
    zipovi = {d: zipfile.ZipFile(IZVOR / f"{d}.zip") for d in DIO}
    imena = {d: sorted(n for n in zipovi[d].namelist() if not n.endswith("/")) for d in DIO}
    rez = {}

    # --- 1. crno-bijele slike po dijelovima ---
    rez["sivo"] = {}
    for d in DIO:
        izbor = rng.choice(imena[d], size=uzorak_sivo, replace=False)
        n = sum(je_sivo(otvori(zipovi[d], i)) for i in tqdm(izbor, desc=f"sivo {d}"))
        rez["sivo"][d] = {"uzorak": uzorak_sivo, "sivih": int(n),
                          "udio": round(n / uzorak_sivo, 4)}
        print(f"{d:11s} skoro crno-bijelih: {n}/{uzorak_sivo} = {100*n/uzorak_sivo:.1f} %")

    # --- 2. da li su pozadine insight slika stvarne wiki fotografije? ---
    hes_wiki = np.stack([dhash(otvori(zipovi["wiki"], i))
                         for i in tqdm(imena["wiki"], desc="hes wiki")])
    izbor = rng.choice(imena["insight"], size=uzorak_insight, replace=False)
    kljuc = lambda n: n.split("/", 1)[1]
    wiki_po_kljucu = {kljuc(n): k for k, n in enumerate(imena["wiki"])}

    udaljenosti, ista_osoba = [], 0
    for i in tqdm(izbor, desc="insight -> wiki"):
        h = dhash(otvori(zipovi["insight"], i))
        dist = hamming(h, hes_wiki)
        najblizi = int(dist.argmin())
        udaljenosti.append(int(dist[najblizi]))
        if najblizi == wiki_po_kljucu.get(kljuc(i)):
            ista_osoba += 1
    u = np.array(udaljenosti)

    # kontrola: kolika je najmanja udaljenost za text2img, koje SIGURNO
    # nije nastalo iz wiki fotografija
    kontrola = []
    for i in rng.choice(imena["text2img"], size=300, replace=False):
        kontrola.append(int(hamming(dhash(otvori(zipovi["text2img"], i)), hes_wiki).min()))
    k = np.array(kontrola)

    rez["insight_pozadina"] = {
        "uzorak": uzorak_insight,
        "najmanja_udaljenost_medijan": float(np.median(u)),
        "udio_do_6": round(float((u <= 6).mean()), 4),
        "udio_do_10": round(float((u <= 10).mean()), 4),
        "najblizi_je_ista_osoba": ista_osoba,
        "kontrola_text2img_medijan": float(np.median(k)),
        "kontrola_text2img_udio_do_6": round(float((k <= 6).mean()), 4),
    }
    print("\ninsight -> najbliza wiki fotografija (Hammingova udaljenost od 64 bita)")
    print(f"   medijan {np.median(u):.0f}   do 6 bita: {100*(u<=6).mean():.1f} %"
          f"   do 10 bita: {100*(u<=10).mean():.1f} %")
    print(f"   najbliza je fotografija ISTE osobe (istog imena): {ista_osoba}/{uzorak_insight}")
    print(f"kontrola text2img: medijan {np.median(k):.0f}   do 6 bita: {100*(k<=6).mean():.1f} %")

    putanja = RESULTS_DIR / "m1_analiza.json"
    putanja.write_text(json.dumps(rez, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nupisano:", putanja)


if __name__ == "__main__":
    main()
