"""Modeli za binarnu klasifikaciju: stvarno (0) / lazno (1).

Svi modeli primaju slike u opsegu [0, 1]. Normalizacija je DIO modela,
a ne pripreme podataka — tako Grad-CAM i testovi otpornosti rade nad
"pravim" pikselima, bez racunanja unazad kroz normalizaciju.

Izlaz je JEDAN broj (logit). Vjerovatnoca laznog = sigmoid(logit).
"""
import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalizacija(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class MaliCNN(nn.Module):
    """Bazni model, treniran od nule. Sluzi kao donja granica za poredjenje."""

    def __init__(self):
        super().__init__()

        def blok(ulaz, izlaz):
            return nn.Sequential(
                nn.Conv2d(ulaz, izlaz, 3, padding=1, bias=False),
                nn.BatchNorm2d(izlaz),
                nn.ReLU(inplace=True),
                nn.Conv2d(izlaz, izlaz, 3, padding=1, bias=False),
                nn.BatchNorm2d(izlaz),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.znacajke = nn.Sequential(
            blok(3, 32), blok(32, 64), blok(64, 128), blok(128, 256),
        )
        self.glava = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.glava(self.znacajke(x)).squeeze(1)


def napravi_model(ime: str, pretreniran: bool = True) -> nn.Module:
    """ime: 'mali_cnn' | 'resnet18' | 'efficientnet_b0'"""
    if ime == "mali_cnn":
        jezgro = MaliCNN()
        return nn.Sequential(Normalizacija(), jezgro)

    if ime == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretreniran else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 1)            # 1000 klasa -> 1 logit
    elif ime == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretreniran else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(f"nepoznat model: {ime}")

    return nn.Sequential(Normalizacija(), _Ravnanje(m))


class _Ravnanje(nn.Module):
    """Pretvara izlaz oblika (B, 1) u (B,), da svi modeli vracaju isto."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).squeeze(1)


def ciljni_sloj(model: nn.Module, ime: str) -> nn.Module:
    """Posljednji konvolucijski sloj — na njemu se racuna Grad-CAM."""
    jezgro = model[1]
    if ime == "mali_cnn":
        return jezgro.znacajke[-1]
    if ime == "resnet18":
        return jezgro.m.layer4[-1]
    if ime == "efficientnet_b0":
        return jezgro.m.features[-1]
    raise ValueError(ime)
