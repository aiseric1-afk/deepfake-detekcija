# Detekcija deepfake lica i generalizacija na nepoznate metode lažiranja

Detektor stvarnih i lažnih slika lica zasnovan na dubokom učenju, s posebnim fokusom na
pitanje koje je ključno u praksi: **prepoznaje li detektor i metode lažiranja koje tokom
treninga nije vidio?**

Projekat iz predmeta *Metode i primjena vještačke inteligencije*, ETF Sarajevo, 2026/2027.

## Sadržaj

- [Problem](#problem)
- [Istraživačka pitanja](#istraživačka-pitanja)
- [Podaci](#podaci)
- [Metodologija](#metodologija)
- [Plan](#plan)
- [Instalacija](#instalacija)
- [Tim](#tim)
- [Licenca](#licenca)

## Problem

Zamjene lica i lica generisana difuzionim modelima koriste se za lažne profile, prevare i
krađu identiteta. Nove metode lažiranja nastaju brže od detektora — zato mjerimo ne samo
tačnost na poznatim metodama, nego i **generalizaciju na nepoznate**.

## Istraživačka pitanja

1. Koliko dobro detektor razlikuje stvarna od lažnih lica kada su u testu iste metode
   lažiranja kao u treningu?
2. Koliko tačnosti detektor gubi na metodama lažiranja koje nije vidio tokom treninga?
3. Da li trening na više metoda lažiranja smanjuje taj gubitak?
4. Na koje dijelove slike detektor obraća pažnju — na lice ili na pozadinu i artefakte
   formata?

## Podaci

[DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) — 120.000 slika:

| Dio | Slika | Sadržaj |
|---|---:|---|
| `wiki` | 30.000 | stvarna lica (IMDB-WIKI) |
| `insight` | 30.000 | zamjena lica (InsightFace) |
| `inpainting` | 30.000 | Stable Diffusion Inpainting |
| `text2img` | 30.000 | Stable Diffusion 1.5 |

Podaci se **ne čuvaju u repozitoriju**. Putanja se zadaje varijablom okruženja `MPVI_DATA`.

Dva rizika koja moramo kontrolisati:

- **Curenje podataka** — `insight` i `inpainting` slike nastale su prepravljanjem stvarnih
  slika, pa podjelu radimo po izvornoj slici, ne nasumično.
- **Prečica preko formata** — ako se stvarne i lažne slike razlikuju po veličini ili
  kompresiji, model može naučiti tu razliku umjesto lažiranja; slike ujednačavamo prije
  treninga.

## Metodologija

| | |
|---|---|
| Bazni model | mala konvolucijska mreža, trenirana od nule |
| Glavni modeli | transfer learning — ResNet-18 i EfficientNet-B0 (ImageNet) |
| Glavna metrika | ROC-AUC (ne zavisi od praga odluke); uz nju F1 i tačnost |
| Glavni eksperiment | trening na jednoj metodi lažiranja, test na sve tri → matrica 3 × 3 |
| Objašnjivost | Grad-CAM |
| Otpornost | JPEG kompresija i smanjivanje slike |

## Plan

| Sedmica | Faza | Rezultat |
|:---:|---|---|
| 3 | M0 — definicija problema, podaci, plan | prijava, projektna knjiga, repozitorij |
| 4 | M1 — analiza i priprema podataka | statistike, podjela bez curenja |
| 5 | M2 — bazni model i transfer learning | prvi rezultati, ROC kriva |
| 6 | M3 — unakrsna generalizacija | matrica 3 × 3 |
| 7 | M4 — poboljšanje generalizacije | poređenje jedne i dvije metode u treningu |
| 8 | M5 — objašnjivost i otpornost | Grad-CAM, test kompresije |
| 9 | M6 — demo aplikacija | radni demo |
| 10 | M7 — rezerva / dodatni test | — |
| 11 | M8 — izvještaj i prezentacija | izvještaj |

Detaljnije: [docs/PLAN.md](docs/PLAN.md).

## Instalacija

```bash
git clone <url-repozitorija>
cd <repozitorij>
pip install -r requirements.txt
```

Potreban je Python 3.10+ i PyTorch. Grafička kartica je preporučena, ali nije obavezna.

## Tim

| Član | Odgovornost |
|---|---|
| | Podaci |
| | Modeli |
| | Generalizacija i otpornost |
| | Objašnjivost i prezentacija |

## Licenca

Kod: MIT. Skup podataka DeepFakeFace ima vlastitu licencu — vidi stranicu skupa.
