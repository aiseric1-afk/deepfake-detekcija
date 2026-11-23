"""Grad-CAM: na koje dijelove slike model gleda kad odlucuje "lazno".

Postupak (Selvaraju i sar., 2017):
  1. propusti sliku kroz model i zapamti izlaz posljednjeg konvolucijskog
     sloja A (oblik: K kanala x h x w)
  2. izracunaj gradijent izlaza modela (logit "lazno") po A
  3. tezina svakog kanala = prosjek njegovog gradijenta po prostoru
  4. mapa = ReLU( suma_k tezina_k * A_k )   -> samo dijelovi koji
     POVECAVAJU ocjenu lazanosti
  5. mapa se rasiri na velicinu slike i normalizuje na [0, 1]

Toplo (crveno) = dio slike koji najvise gura odluku prema "lazno".
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import ciljni_sloj


class GradCAM:
    def __init__(self, model, ime_modela):
        self.model = model.eval()
        self.sloj = ciljni_sloj(model, ime_modela)
        self._akt = None
        self._grad = None
        self._kuke = [self.sloj.register_forward_hook(self._zapamti)]

    def _zapamti(self, modul, ulaz, izlaz):
        # aktivacije sloja, i kuka na samom tenzoru koja ce uhvatiti njegov
        # gradijent pri backward() — otpornije od kuke na modulu kad mreza
        # ima ReLU(inplace=True)
        self._akt = izlaz.detach()
        if izlaz.requires_grad:
            izlaz.register_hook(lambda g: setattr(self, "_grad", g.detach()))

    def __call__(self, x):
        """x: (1, 3, H, W) u opsegu [0, 1]. Vraca (mapa HxW u [0,1], p_lazno)."""
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)       # da bi gradijent tekao kroz mrezu
        logit = self.model(x)
        logit.sum().backward()

        tezine = self._grad.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1)
        mapa = F.relu((tezine * self._akt).sum(dim=1, keepdim=True)) # (1, 1, h, w)
        mapa = F.interpolate(mapa, size=x.shape[2:], mode="bilinear", align_corners=False)
        mapa = mapa[0, 0].cpu().numpy()
        if mapa.max() > 0:
            mapa = mapa / mapa.max()
        return mapa, float(torch.sigmoid(logit).item())

    def ukloni(self):
        for k in self._kuke:
            k.remove()


def preklopi(slika_hwc, mapa, alfa=0.45):
    """Slika (H, W, 3) u [0,1] + toplotna mapa -> preklopljena slika."""
    import matplotlib
    boja = matplotlib.colormaps["jet"](mapa)[..., :3]
    return np.clip((1 - alfa) * slika_hwc + alfa * boja, 0, 1)
