# PLAN PROJEKTA — faze

**Svaka faza je zaokružena cjelina koja se može pokazati asistentu ili profesoru.**

Predavanja su četvrtkom. Konsultacije: ponedjeljak i četvrtak 08–09.

---

```
FAZA  SEDMICA    DATUM     STA SE RADI                              STA SE POKAZUJE
-----------------------------------------------------------------------------------------
M0    3          22.10.    prijava, projektna knjiga, GitHub,       definicija problema,
                           definicija problema, odabir podataka     izbor podataka, plan

M1    4          29.10.    preuzimanje, analiza podataka,           statistike skupa,
                           podjela BEZ curenja, ujednacavanje       primjeri slika, kako je
                           formata slika                            sprijeceno curenje

M2    5           5.11.    bazni CNN, pa transfer learning          prvi rezultati, krive
                           (ResNet / EfficientNet), metrike         ucenja, ROC kriva

M3    6          12.11.    UNAKRSNA GENERALIZACIJA:                 matrica 3x3 — glavni
                           treniraj na jednoj metodi, testiraj      nalaz projekta
                           na sve tri

M4    7          19.11.    poboljsanje generalizacije:              poredjenje: jedna metoda
                           trening na dvije metode, test na trecoj  naspram dvije u treningu

M5    8          26.11.    objasnjivost (Grad-CAM) i otpornost      toplotne mape: gdje model
                           na JPEG kompresiju i smanjivanje         gleda; pad tacnosti uz
                                                                    kompresiju

M6    9           3.12.    demo aplikacija                          radni demo

M7    10         10.12.    rezerva: novi generatori ili             dodatni nalaz
                           GAN lica (StyleGAN) kao cetvrti test

M8    11         17.12.    izvjestaj, README, prezentacija          nacrt izvjestaja

      12         24.12.    PREDAJA (okvirno)
      poslije    januar    usmeni — 3 pitanja po clanu
```

---

## Zašto ovim redom

Svaka faza koristi rezultat prethodne i daje **jedan novi nalaz**. Tako na svakim
konsultacijama postoji nešto konkretno — graf, tabela, demo — i projektna knjiga raste
sedmicu po sedmicu.

**M3 je srce projekta.** Ako detektor savršeno radi na metodi na kojoj je treniran, a pada na
ostalim, to je pravi nalaz iz stvarnog svijeta: detektori zaostaju za novim metodama
lažiranja. Takav rezultat nije neuspjeh projekta, nego njegov odgovor.

---

## Pravila za projektnu knjigu

```
1.  upis POSLIJE svakog radnog dana ili sastanka, s DATUMOM
2.  upisuju se i NEUSPJESNI eksperimenti — profesor ih izricito trazi
3.  svaki clan upisuje svoj dio i kratko objasnjenje
4.  datumi su STVARNI — knjiga je dnevnik, ne naknadna prica
```

## Pravilo za usmeni

Svaki član dobija **tri pitanja**, i svaki tačan odgovor nosi **trećinu** projektnih bodova.
Zato:

```
svaki clan mora razumjeti SVE faze, ne samo svoju
na kraju svake faze:  onaj ko je radio objasni ostalima u 10 minuta
```
