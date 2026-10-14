# Studio 2 — Klasifikasi Sampah · Figo bilang nama kelompoknya 9 naga aja

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |Louis Hutabarat|123140052|                  |
| 2  |Pradana Figo Ariasya|123140063|                  |
| 3  |Giovan Lado|123140068|          |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny pretrained ImageNet (torchvision). Layer klasifikasi terakhir diganti agar menghasilkan 5 kelas: `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`.
- **Strategi pelatihan**: fine-tuning seluruh layer (tanpa freezing) dengan optimizer AdamW, scheduler CosineAnnealingLR, dan loss CrossEntropy selama 15 epoch. Checkpoint terbaik dipilih berdasarkan Macro-F1 validasi.
- **Data tambahan**: tidak ada
- **Ukuran input & preprocessing**: 224×224 dengan normalisasi standar ImageNet.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | |0.9790| |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat :

- Fine-tuning penuh ConvNeXt-Tiny pretrained langsung memberi hasil kuat: di epoch 1 Macro-F1 validasi sudah 0.9422 (akurasi 94.56%), dan terbaik 0.9790 di epoch 11 (akurasi 98.00%).
- Akurasi latih mencapai 100.00% sejak epoch 9, sedangkan akurasi validasi stagnan di kisaran 97.28%–98.00%, tanda overfitting ringan.
- Skor lokal dipilih dari split validasi yang sama dengan pemilihan checkpoint, jadi bisa sedikit lebih optimistis dibanding skor dari asisten.
- Rencana perbaikan: cek duplikat gambar antar split, analisis confusion matrix per kelas, augmentasi lebih kuat, dan early stopping.

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
