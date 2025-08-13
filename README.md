# PP_7 – COVID-19 CXR klasifikatorius (COVID vs non-COVID)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k0v3PlqBKV4Z7tYwZ6Jq2O7JnC-dNFok?usp=sharing)

Trumpai: šiame Colab užrašinėje automatiškai atsisiunčiamas **COVID-19 Radiography Dataset (Kaggle)**, apmokomas **EfficientNetB0** (transfer learning) ir įvertinamos metrikos **accuracy / sensitivity / specificity / ROC-AUC**. Tikslas – pasiekti >50% (aukščiau už atsitiktinį spėjimą).  
> Pastaba: edukacinis projektas, **ne klinikinis įrankis**.

---

## Duomenys
- Šaltinis: *COVID-19 Radiography Dataset* (Kaggle). Klasės: `COVID`, `Normal`, `Viral Pneumonia`, `Lung_Opacity`.
- Užduotis: **binarinė** – `COVID` (1) vs **non-COVID** (0 = Normal + Viral Pneumonia + Lung_Opacity).

## Kaip paleisti (Colab)
1. Atidarykite aukščiau esantį **Open in Colab** mygtuką.  
2. **Runtime → Change runtime type → GPU** (T4) → Save.  
3. **Run all**. Paprašius – įkelkite **`kaggle.json`** (Kaggle → Account → Create New API Token).  
4. Gale pamatysite metrikas, ROC/PR kreives ir konfuzijos matricą.

## Modelis
- Backbone: `EfficientNetB0` (ImageNet), galva: GAP → Dropout → Dense(1, sigmoid).  
- Optimizer: Adam/AdamW; LR 1e-4 → 1e-5 (fine-tune).  
- Split: ~70/15/15 (train/val/test), **Youden J** slenkstis iš `val`.  
- Augmentacijos: horizontal flip, brightness/contrast; `224×224` resize.  
- Stabilumo sprendimai Colab: mažesnis `batch_size`, ribotas `.prefetch()` ir `num_parallel_calls`.

## Rezultatai (įrašykite savo skaičius)
- **Accuracy:** …  
- **Sensitivity (TPR, COVID):** …  
- **Specificity (TNR, non-COVID):** …  
- **ROC-AUC:** …  
- **Confusion matrix:** `[[TN, FP], [FN, TP]] = [[…,…],[…,…]]`

## Refleksija (santrauka)
- Heterogeniški CXR šaltiniai → rizika išmokti artefaktus, ribota perkeltis.  
- Binarizacija supaprastina užduotį, bet mažina etiologinę specifiką.  
- Praktikai svarbus **threshold** (naudotas **Youden J**).  
- **Transfer learning** ženkliai pagreitina ir stabilizuoja mokymą.  
- `tf.data` materializavimas gali sprogdinti RAM; ribotas prefetch/parallelism padėjo.

## Repo struktūra (rekomenduojama)
