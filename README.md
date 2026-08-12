# Face Recognition Server

API sederhana untuk:
- menerima gambar wajah
- membuat embedding dengan ArcFace pretrained
- menyimpan embedding ke SQLite
- melakukan prediksi dengan cosine similarity
- memberi ID otomatis mulai dari 101

Tidak ada proses training.

## 1. Install

Disarankan Python 3.10-3.12.

```bash
python -m venv .venv
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```powershell
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

## 2. Jalankan server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Server:
```text
http://127.0.0.1:8000
```

Dokumentasi otomatis:
```text
http://127.0.0.1:8000/docs
```

## 3. Register wajah

Endpoint:
```text
POST /register
```

Kirim 1 file gambar yang berisi tepat 1 wajah.

Contoh curl:
```bash
curl -X POST "http://127.0.0.1:8000/register" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@wajah.jpg"
```

Hasil pertama:
```json
{
  "success": true,
  "id": 101
}
```

Berikutnya 102, 103, dst.

## 4. Prediksi

Endpoint:
```text
POST /predict
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

Contoh:
```json
{
  "success": true,
  "found": true,
  "id": 102,
  "similarity": 0.812345,
  "threshold": 0.5
}
```

Kalau tidak ditemukan:
```json
{
  "success": true,
  "found": false,
  "id": null,
  "similarity": 0.41,
  "threshold": 0.5
}
```

## 5. Database

File:
```text
face.db
```

Tabel:
```text
face_embeddings
- id          INTEGER PRIMARY KEY
- embedding   BLOB
```

## Catatan

`face_model.py` memakai pretrained ArcFace dari InsightFace sebagai implementasi yang langsung jalan.

Kalau beta lu sudah punya model/script ArcFace sendiri, cukup ganti isi:
```text
FaceModel.load()
FaceModel.get_embedding()
```

Database dan API tidak perlu diubah.

Threshold `0.50` hanya nilai awal. Nanti tentukan threshold yang sesuai dengan model dan dataset lu setelah pengujian.
