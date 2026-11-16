"""Petlja za treniranje i evaluaciju — ne zavisi od skupa podataka.

Mjesovita preciznost (AMP) je ISKLJUCENA po defaultu. Izmjereno na
GTX 1650 SUPER (bez tenzorskih jezgara): ResNet-18 trening FP32 255
slika/s, AMP 67 slika/s — AMP je skoro 4 puta SPORIJI. Na karticama s
tenzorskim jezgrima (npr. T4 na Colabu) ukljuciti ga opcijom --amp.
"""
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm as _tqdm

from .metrike import izracunaj

# trake napretka samo u terminalu; u log fajlu bi zagusile ispis
tqdm = partial(_tqdm, disable=not sys.stderr.isatty())


def _autocast(uredjaj, amp):
    return torch.autocast(device_type=uredjaj.type, enabled=amp and uredjaj.type == "cuda")


def treniraj_epohu(model, loader, optimizer, scaler, uredjaj, amp=False, opis="trening"):
    model.train()
    kriterij = nn.BCEWithLogitsLoss()
    ukupno, n = 0.0, 0
    for x, y in tqdm(loader, desc=opis, leave=False):
        x = x.to(uredjaj, non_blocking=True)
        y = y.float().to(uredjaj, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(uredjaj, amp):
            gubitak = kriterij(model(x).float(), y)
        scaler.scale(gubitak).backward()
        scaler.step(optimizer)
        scaler.update()
        ukupno += gubitak.item() * len(y)
        n += len(y)
    return ukupno / max(n, 1)


@torch.no_grad()
def predvidi(model, loader, uredjaj, opis="evaluacija", amp=False):
    """Vraca (vjerovatnoce_laznog, stvarne_oznake) kao numpy nizove."""
    model.eval()
    sve_p, sve_y = [], []
    for x, y in tqdm(loader, desc=opis, leave=False):
        x = x.to(uredjaj, non_blocking=True)
        with _autocast(uredjaj, amp):
            logit = model(x)
        sve_p.append(torch.sigmoid(logit.float()).cpu().numpy())
        sve_y.append(y.numpy())
    return np.concatenate(sve_p), np.concatenate(sve_y)


def evaluiraj(model, loader, uredjaj, opis="evaluacija", amp=False):
    p, y = predvidi(model, loader, uredjaj, opis, amp)
    return izracunaj(p, y)


def treniraj(model, loader_tr, loader_val, uredjaj, epohe=5, lr=1e-3,
             tezina_raspada=1e-4, putanja_tezina=None, amp=False, ispis=print):
    """Trenira model; cuva tezine iz epohe s najboljim AUC-om na validaciji."""
    model.to(uredjaj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tezina_raspada)
    raspored = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epohe)
    scaler = torch.amp.GradScaler(enabled=amp and uredjaj.type == "cuda")

    historija, najbolji_auc = [], -1.0
    for e in range(1, epohe + 1):
        t0 = time.time()
        gub = treniraj_epohu(model, loader_tr, optimizer, scaler, uredjaj, amp,
                             opis=f"epoha {e}/{epohe}")
        raspored.step()
        val = evaluiraj(model, loader_val, uredjaj, opis="validacija", amp=amp)
        val.update({"epoha": e, "gubitak_trening": gub, "sekunde": round(time.time() - t0, 1)})
        historija.append(val)
        ispis(f"epoha {e:2d}  gubitak {gub:.4f}  val AUC {val.get('roc_auc', float('nan')):.4f}"
              f"  tacnost {val['tacnost']:.4f}  ({val['sekunde']} s)", flush=True)
        if putanja_tezina and val.get("roc_auc", -1) > najbolji_auc:
            najbolji_auc = val["roc_auc"]
            torch.save(model.state_dict(), putanja_tezina)
    return historija
