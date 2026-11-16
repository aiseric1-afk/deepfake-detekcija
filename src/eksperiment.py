"""Jedan eksperiment: treniraj na izabranim metodama laziranja, testiraj na
SVAKOJ metodi posebno.

Primjeri:
    # bazni mali CNN, trening na sve tri metode
    python -m src.eksperiment --ime m2_mali_cnn --model mali_cnn --trening insight inpainting text2img

    # red matrice M3: trening samo na insight, test na sve tri
    python -m src.eksperiment --ime m3_insight --model resnet18 --trening insight

Rezultat: results/<ime>.json  i tezine u MPVI_CKPT/<ime>.pt
"""
import argparse
import json
import time

import torch

from .config import CKPT_DIR, DEVICE, RESULTS_DIR, postavi_sjeme
from .metrike import izracunaj
from .model import napravi_model
from .petlja import predvidi, treniraj
from .podaci import LAZNI_DIJELOVI, izaberi, loader, ucitaj_indeks, zagrij_kes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ime", required=True)
    p.add_argument("--model", default="resnet18",
                   choices=["mali_cnn", "resnet18", "efficientnet_b0"])
    p.add_argument("--trening", nargs="+", default=LAZNI_DIJELOVI, choices=LAZNI_DIJELOVI)
    p.add_argument("--epohe", type=int, default=5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--velicina", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-po-klasi", type=int, default=None,
                   help="svedi svaku klasu u treningu na N slika")
    p.add_argument("--radnika", type=int, default=3,
                   help="na Windowsu svaki radnik trazi ~4 s za pokretanje")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true",
                   help="mjesovita preciznost; ukljuciti SAMO na karticama s tenzorskim jezgrima")
    a = p.parse_args()

    postavi_sjeme(a.seed)
    lr = a.lr or (1e-3 if a.model == "mali_cnn" else 3e-4)
    t = time.time()
    zagrij_kes()
    print(f"paket slika ucitan u kes za {time.time() - t:.0f} s", flush=True)
    indeks = ucitaj_indeks()

    tr = izaberi(indeks, "trening", a.trening, a.n_po_klasi, seed=a.seed)
    val = izaberi(indeks, "validacija", a.trening)
    print(f"trening: {len(tr)} slika  (stvarnih {int((tr.oznaka == 0).sum())}, "
          f"laznih {int((tr.oznaka == 1).sum())})   metode: {', '.join(a.trening)}")
    print(f"validacija: {len(val)} slika   model: {a.model}   lr: {lr}   uredjaj: {DEVICE}")

    model = napravi_model(a.model, pretreniran=a.model != "mali_cnn")
    tezine = CKPT_DIR / f"{a.ime}.pt"
    t0 = time.time()
    historija = treniraj(
        model,
        loader(tr, a.velicina, True, a.batch, a.radnika),
        loader(val, a.velicina, False, a.batch, a.radnika),
        DEVICE, epohe=a.epohe, lr=lr, putanja_tezina=tezine, amp=a.amp,
    )
    trajanje = round(time.time() - t0)

    # test: najbolje tezine; JEDAN prolaz kroz cijeli test skup, pa se za
    # svaku metodu racuna AUC na (stvarne + ta metoda)
    model.load_state_dict(torch.load(tezine, map_location=DEVICE))
    sve_test = izaberi(indeks, "test", LAZNI_DIJELOVI)
    p, y = predvidi(model, loader(sve_test, a.velicina, False, a.batch, a.radnika), DEVICE,
                    "test", amp=a.amp)
    dio_niz = sve_test["dio"].to_numpy()
    test = {}
    for dio in LAZNI_DIJELOVI:
        maska = (dio_niz == "wiki") | (dio_niz == dio)
        m = izracunaj(p[maska], y[maska])
        test[dio] = m
        oznaka = "  (vidjena u treningu)" if dio in a.trening else "  (NEVIDJENA)"
        print(f"TEST {dio:11s} AUC {m.get('roc_auc', float('nan')):.4f}   "
              f"tacnost {m['tacnost']:.4f}   F1 {m['f1']:.4f}   n={m['n']}{oznaka}")

    izvjestaj = {"ime": a.ime, "argumenti": vars(a), "lr": lr, "trajanje_s": trajanje,
                 "trening_slika": len(tr), "historija": historija, "test": test}
    (RESULTS_DIR / f"{a.ime}.json").write_text(
        json.dumps(izvjestaj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nupisano: results/{a.ime}.json   (trajanje {trajanje} s)")


if __name__ == "__main__":
    main()
