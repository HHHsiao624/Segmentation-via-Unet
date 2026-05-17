# Segmentation-via-Unet
使用 UNet 進行視網膜血管分割，並結合 AutoEncoder 進行影像重建。

**資料集：** [DRIVE Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction)

---

## 一、Segmentation via UNet

### 前處理 — 數據增強

| 參數 | 數值 |
|---|---|
| 亮度 (brightness) | 0.2 |
| 對比度 (contrast) | 0.4 |
| 飽和度 (saturation) | 0.2 |
| 色調 (hue) | 0.1 |

### 結果

| 指標 | 數值 |
|---|---|
| Mean IoU (MIOU) | 0.602451 |
| Average F1 | 0.747238 |

### 成果圖 

原始影像、切割結果、資料集標準答案
![成果圖]((best)test6/test6_1.png)
---

## 二、AutoEncoder

原始影像與重建影像對比。

| 指標 | 數值 |
|---|---|
| 平均 PSNR | 32.675 dB |

### 成果圖

![成果圖](results/example.png)
