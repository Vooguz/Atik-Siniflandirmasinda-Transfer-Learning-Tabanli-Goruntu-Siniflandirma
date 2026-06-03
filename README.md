# Atık Sınıflandırmasında Transfer Learning Tabanlı Görüntü Sınıflandırma

> 10 sınıflı çöp/atık görüntülerini sınıflandırmak için **transfer learning** ve **sıfırdan CNN** yaklaşımlarının karşılaştırıldığı bir derin öğrenme projesi.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Git LFS](https://img.shields.io/badge/Git%20LFS-models-green)

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Veri Seti](#-veri-seti)
- [Modeller](#-modeller)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum-neler-yapılmalı)
- [Çalıştırma](#-çalıştırma)
- [Yöntem Detayları](#-yöntem-detayları)
- [Değerlendirme](#-değerlendirme)
- [Yazarlar](#-yazarlar)

---

## 🎯 Proje Hakkında

Bu projede atık görüntüleri **10 farklı sınıfa** ayrılmaktadır. Üç farklı model eğitilip karşılaştırılmıştır:

1. **ResNet50** — ImageNet ağırlıklarıyla transfer learning + fine-tuning (son 30 katman)
2. **MobileNetV2** — ImageNet ağırlıklarıyla transfer learning (feature extraction)
3. **Sıfırdan CNN** — Önceden eğitilmemiş, basit bir konvolüsyonel ağ (baseline)

Ek olarak modelin kararlarını yorumlamak için **Grad-CAM** ile ısı haritaları (heatmap) üretilmektedir.

---

## 📦 Veri Seti

Veri seti `data/original/` altında her sınıf için bir klasör olacak şekilde düzenlenmiştir. Toplam **12.259 görüntü** ve **10 sınıf** bulunur:

| Sınıf | Görüntü Sayısı | Sınıf | Görüntü Sayısı |
|-------|:---:|-------|:---:|
| `clothes` | 1892 | `cardboard` | 1411 |
| `glass` | 1736 | `paper` | 1336 |
| `plastic` | 1597 | `metal` | 930 |
| `shoes` | 1449 | `battery` | 756 |
| | | `biological` | 699 |
| | | `trash` | 453 |

> ⚠️ **Sınıf dengesizliği:** En kalabalık sınıf (`clothes`, 1892) ile en az örnekli sınıf (`trash`, 453) arasında ~4 kat fark vardır. Bu nedenle eğitimde `class_weight` kullanılarak az örnekli sınıflar dengelenmiştir.

### Veri setini edinme

📥 **Veri seti repoya dahil edilmemiştir** (boyutu ~1.2 GB). Projeyi çalıştırmadan önce veri setini indirip aşağıdaki yapıda yerleştirmeniz gerekir:

```
data/
└── original/
    ├── battery/
    ├── biological/
    ├── cardboard/
    ├── clothes/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    ├── shoes/
    └── trash/
```

> 💡 Bu sınıf yapısı Kaggle'daki **"Garbage Classification"** türü atık veri setleriyle uyumludur. Kullandığınız veri setini bu klasör düzenine getirmeniz yeterlidir.

---

## 🧠 Modeller

Eğitilmiş model ağırlıkları **Git LFS** ile `models/` klasöründe saklanır:

| Dosya | Çatı | Açıklama |
|-------|------|----------|
| `resnet50_finetuned.keras` | TensorFlow/Keras | ResNet50, transfer learning + fine-tuning |
| `mobilenetv2.keras` | TensorFlow/Keras | MobileNetV2, transfer learning |
| `scratch_cnn.keras` | TensorFlow/Keras | Sıfırdan eğitilmiş CNN |
| `resnet18_feature_extraction.pth` | PyTorch | ResNet18, özellik çıkarma (feature extraction) |
| `resnet18_finetuned.pth` | PyTorch | ResNet18, fine-tuning |
| `resnet18_scratch.pth` | PyTorch | ResNet18, sıfırdan eğitim |

> ⚠️ **Git LFS notu:** Model dosyalarının gerçek içeriğini indirebilmek için klonlamadan önce `git lfs install` çalıştırın. Aksi halde `.keras`/`.pth` dosyaları yalnızca küçük "pointer" dosyaları olarak gelir. Bkz. [Kurulum](#-kurulum-neler-yapılmalı).

---

## 📁 Proje Yapısı

```
CNN-PROJE/
├── garbage_classification.ipynb   # Ana notebook (tüm eğitim + değerlendirme + Grad-CAM)
├── models/                        # Eğitilmiş model ağırlıkları (Git LFS)
├── data/                          # Veri seti (repoya dahil DEĞİL — siz ekleyeceksiniz)
│   ├── original/                  # Orijinal görüntüler (10 sınıf klasörü)
│   ├── standardized_256/          # 256px'e standartlaştırılmış sürüm
│   └── standardized_384/          # 384px'e standartlaştırılmış sürüm
├── requirements.txt               # Python bağımlılıkları
├── .gitignore
└── README.md
```

---

## ⚙️ Kurulum (Neler Yapılmalı?)

### 1. Depoyu klonlayın

```bash
# Önce Git LFS'i kurun (bir kez yapılır)
git lfs install

# Repoyu klonlayın — model dosyaları LFS ile otomatik iner
git clone https://github.com/Vooguz/Atik-Siniflandirmasinda-Transfer-Learning-Tabanli-Goruntu-Siniflandirma.git
cd Atik-Siniflandirmasinda-Transfer-Learning-Tabanli-Goruntu-Siniflandirma
```

> Repoyu LFS kurmadan klonladıysanız, sonradan `git lfs pull` ile model dosyalarını indirebilirsiniz.

### 2. Sanal ortam oluşturun ve etkinleştirin

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. Veri setini ekleyin

Veri setini indirip [Veri Seti](#-veri-seti) bölümündeki yapıya göre `data/original/` altına yerleştirin.

---

## ▶️ Çalıştırma

Jupyter Notebook'u açın ve hücreleri **sırayla** çalıştırın:

```bash
jupyter notebook garbage_classification.ipynb
```

Notebook akışı:

| Adım | İçerik |
|------|--------|
| 1 | Veri setini yükleme (`image_dataset_from_directory`, %80/%20 train/val) |
| 2 | Sınıf ağırlıklarının hesaplanması (dengesizlik için) |
| 3 | ResNet50 — transfer learning + fine-tuning |
| 4 | MobileNetV2 — transfer learning |
| 5 | Sıfırdan CNN |
| 6 | Değerlendirme + karşılaştırma (accuracy, confusion matrix, classification report) |
| 7 | Grad-CAM görselleştirmesi |

> ⏳ **Not:** Eğitim, donanıma bağlı olarak uzun sürebilir. GPU'lu bir ortam şiddetle önerilir. Eğitilen modeller otomatik olarak `models/` klasörüne kaydedilir.

---

## 🔬 Yöntem Detayları

- **Görüntü boyutu:** 224×224, **batch size:** 32
- **Preprocessing (önemli):** Her mimari kendi ön işlemesini bekler:
  - ResNet50 → `resnet50.preprocess_input` (modelin içine `Lambda` katmanı olarak gömülü)
  - MobileNetV2 → `mobilenet_v2.preprocess_input` (girdiyi `[-1, 1]` aralığına getirir)
  - Sıfırdan CNN → basit `Rescaling(1./255)`
- **Sınıf dengesizliği:** `class_weight` ile az örnekli sınıflar dengelenir.
- **Fine-tuning:** ResNet50'nin son 30 katmanı açılarak düşük öğrenme oranıyla (`1e-5`) yeniden eğitilir.
- **Grad-CAM:** ResNet50 için iki aşamalı (conv + classifier) yaklaşım kullanılır; böylece iç içe gömülü `resnet_base` modelinde graf kopması yaşanmaz ve modelin görüntünün hangi bölgesine "baktığı" görselleştirilir.

---

## 📊 Değerlendirme

Modeller validation seti üzerinde şu metriklerle karşılaştırılır:

- **Validation Accuracy / Loss** (üç model yan yana)
- **Confusion Matrix** (en iyi model için)
- **Classification Report** (sınıf bazında precision / recall / F1-score)
- **Grad-CAM** ısı haritaları (yorumlanabilirlik)

---

## 👥 Yazarlar

| Öğrenci No | Ad Soyad |
|------------|----------|
| 032390074 | Mert Arı |
| 032290038 | Oğuz Eren |
