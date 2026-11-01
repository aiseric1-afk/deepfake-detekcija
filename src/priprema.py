"""Faza M1 — priprema podataka.

Sta radi, i zasto:

1. UJEDNACAVANJE GEOMETRIJE
   Lazne slike (insight, inpainting) nastale su tako sto je stvarna
   fotografija RAZVUCENA na 512x512, bez cuvanja proporcija. Zato i
   stvarne slike razvlacimo na 512x512 istim putem, pa sve smanjujemo
   na 256x256. Bez toga bi model mogao uciti "razvuceno = lazno".

2. POVEZIVANJE POZADINA
   InsightFace je lice osobe X zalijepio na STVARNU fotografiju druge
   osobe W iz istog skupa (analiza M1: 77 % slika). Pozadinu pronalazimo
   perceptualnim hesom.

3. PODJELA BEZ CURENJA
   Osobe se dijele 70/15/15 na trening/validaciju/test. Insight slika
   ostaje samo ako su i osoba X i osoba W (pozadina) u ISTOM skupu —
   inace bi ista fotografija bila i u treningu i u testu.

Izlaz:
   <MPVI_DATA>/dff256/<dio>/<podfolder>/<ime>.jpg     slike 256x256
   <MPVI_DATA>/dff256/indeks.csv                       sve slike i podjela
   results/m1_priprema.json                            statistika

Pokretanje:  python -m src.priprema
"""
import csv
import io
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import DATA_DIR, RESULTS_DIR

DIO = ["wiki", "insight", "inpainting", "text2img"]
OZNAKA = {"wiki": 0, "insight": 1, "inpainting": 1, "text2img": 1}
IZVOR = DATA_DIR / "deepfakeface"
CILJ = DATA_DIR / "dff256"
VELICINA = 256
JPEG_KVALITET = 95
PRAG_POZADINE = 6            # bita od 64; kontrola text2img: 0 % ispod 6
UDIO = (0.70, 0.15, 0.15)    # trening / validacija / test
SEED = 42

_zipovi = {}


def _init():
    for d in DIO:
        _zipovi[d] = zipfile.ZipFile(IZVOR / f"{d}.zip")


def dhash(im, velicina=8):
    g = np.asarray(im.convert("L").resize((velicina + 1, velicina)), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def _obradi(zadatak):
    """Jedna slika: hes, ujednacavanje geometrije, upis 256x256."""
    dio, ime = zadatak
    im = Image.open(io.BytesIO(_zipovi[dio].read(ime))).convert("RGB")
    sirina, visina = im.size
    h = dhash(im)
    if dio == "wiki":
        # isti put kao kod laznih: razvuci na 512x512, bez cuvanja proporcija
        im = im.resize((512, 512), Image.BICUBIC)
    im = im.resize((VELICINA, VELICINA), Image.BICUBIC)
    kljuc = ime.split("/", 1)[1]                       # "61/19242061_...jpg"
    izlaz = CILJ / dio / kljuc
    izlaz.parent.mkdir(parents=True, exist_ok=True)
    im.save(izlaz, "JPEG", quality=JPEG_KVALITET)
    return dio, kljuc, h, sirina, visina


# broj jedinica u svakom bajtu 0..255 — za brzo racunanje Hammingove udaljenosti
_BROJ_BITOVA = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_min(H, B, blok=256):
    """Za svaki hes u H (n, 8 bajtova): indeks i udaljenost najblizeg u B."""
    idx = np.empty(len(H), dtype=np.int64)
    dist = np.empty(len(H), dtype=np.int64)
    for s in range(0, len(H), blok):
        x = np.bitwise_xor(H[s:s + blok, None, :], B[None, :, :])    # (b, nB, 8)
        d = _BROJ_BITOVA[x].sum(axis=2, dtype=np.int32)              # (b, nB)
        idx[s:s + blok] = d.argmin(axis=1)
        dist[s:s + blok] = d.min(axis=1)
    return idx, dist


def main(procesa=7):
    CILJ.mkdir(parents=True, exist_ok=True)
    _init()
    zadaci = [(d, n) for d in DIO for n in sorted(_zipovi[d].namelist()) if not n.endswith("/")]

    # --- 1. obrada svih slika, paralelno ---
    rezultati = []
    with Pool(procesa, initializer=_init) as pool:
        for r in tqdm(pool.imap_unordered(_obradi, zadaci, chunksize=64),
                      total=len(zadaci), desc="obrada slika"):
            rezultati.append(r)

    po_dijelu = {d: {} for d in DIO}
    for dio, kljuc, h, w, hgt in rezultati:
        po_dijelu[dio][kljuc] = (h, w, hgt)

    osobe = sorted(po_dijelu["wiki"])                       # 30.000 kljuceva
    poz = {k: i for i, k in enumerate(osobe)}
    H_wiki = np.stack([po_dijelu["wiki"][k][0] for k in osobe])

    # --- 2. pozadine insight slika ---
    ins = sorted(po_dijelu["insight"])
    H_ins = np.stack([po_dijelu["insight"][k][0] for k in ins])
    idx, dist = hamming_min(H_ins, H_wiki)
    pozadina = {k: (osobe[i] if d <= PRAG_POZADINE else None, int(d))
                for k, i, d in zip(ins, idx, dist)}

    # kontrola: inpainting X mora biti blizu wiki X (nastao je iz nje)
    inp = sorted(po_dijelu["inpainting"])
    H_inp = np.stack([po_dijelu["inpainting"][k][0] for k in inp])
    H_isti = np.stack([po_dijelu["wiki"][k][0] for k in inp])
    d_inp = np.unpackbits(np.bitwise_xor(H_inp, H_isti), axis=1).sum(axis=1)

    # --- 3. podjela osoba 70/15/15 ---
    rng = np.random.default_rng(SEED)
    red = rng.permutation(len(osobe))
    n_tr = int(UDIO[0] * len(osobe))
    n_val = int(UDIO[1] * len(osobe))
    split = {}
    for r, i in enumerate(red):
        split[osobe[i]] = "trening" if r < n_tr else ("validacija" if r < n_tr + n_val else "test")

    # --- 4. indeks ---
    redovi, izbaceno = [], {"pozadina_u_drugom_skupu": 0}
    for dio in DIO:
        for kljuc in sorted(po_dijelu[dio]):
            s = split[kljuc]
            zadrzano, razlog, poz_osoba, poz_dist = True, "", "", ""
            if dio == "insight":
                w, d = pozadina[kljuc]
                poz_osoba, poz_dist = (w or ""), d
                if w is not None and split[w] != s:
                    zadrzano, razlog = False, "pozadina_u_drugom_skupu"
                    izbaceno["pozadina_u_drugom_skupu"] += 1
            _, sir, vis = po_dijelu[dio][kljuc]
            redovi.append({
                "putanja": f"{dio}/{kljuc}", "dio": dio, "oznaka": OZNAKA[dio],
                "osoba": kljuc, "split": s, "zadrzano": int(zadrzano), "razlog": razlog,
                "pozadina_osoba": poz_osoba, "pozadina_udaljenost": poz_dist,
                "izvorna_sirina": sir, "izvorna_visina": vis,
            })

    with open(CILJ / "indeks.csv", "w", newline="", encoding="utf-8") as f:
        pisac = csv.DictWriter(f, fieldnames=list(redovi[0]))
        pisac.writeheader()
        pisac.writerows(redovi)

    # --- 5. statistika ---
    tabela = {}
    for r in redovi:
        if not r["zadrzano"]:
            continue
        tabela.setdefault(r["split"], {}).setdefault(r["dio"], 0)
        tabela[r["split"]][r["dio"]] += 1

    stat = {
        "ukupno_slika": len(redovi),
        "osoba": len(osobe),
        "insight_s_pronadjenom_pozadinom": int(sum(1 for w, _ in pozadina.values() if w)),
        "insight_izbaceno_curenje": izbaceno["pozadina_u_drugom_skupu"],
        "inpainting_do_wiki_iste_osobe_medijan_bita": float(np.median(d_inp)),
        "inpainting_do_wiki_iste_osobe_udio_do_6": round(float((d_inp <= 6).mean()), 4),
        "zadrzano_po_skupu_i_dijelu": tabela,
    }
    (RESULTS_DIR / "m1_priprema.json").write_text(
        json.dumps(stat, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
