# Music Transcriber

Gitar kayıtlarını nota olaylarına, MIDI'ye ve gitar tab'ına dönüştüren uçtan uca bir otomatik müzik transkripsiyonu (AMT) projesi.

```
guitar.wav
   │
   ▼
Ön işleme (44.1 kHz, mono, normalize → log-mel / CQT)
   │
   ▼
AMT modeli: CNN → BiLSTM → onset / frame / offset kafaları
   │
   ▼
Nota olayları (pitch, onset, offset, confidence)
   │
   ├──────────────► MIDI
   ▼
Tab optimizasyonu (Viterbi, minimum maliyetli parmak yolu) ──► Gitar TAB
   │
   ▼
Müzik editörü (dalga formu, TAB, piyano rulosu, oynatma, düzenleme, MIDI/MusicXML/TAB dışa aktarma)
```

**Temel tasarım kararı:** Notaları ML modeli tespit ediyor. Hangi notanın hangi tel ve perdede çalınacağına ise algoritmik bir sistem (`packages/music-core/tab.py`) karar veriyor:

```
cost = |el pozisyonu - tercih edilen perde| + açık teller + akor genişliği + zorlayıcı açıklık  (olay başına)
     + pozisyon değişimi + tel değişimi                                                  (ardışık olaylar arası)
```

Ağırlıklar GuitarSet validation setinde grid search ile ayarlandı (`scripts/tune_tab.py`). Oyuncular çoğunlukla 5–9. perdelerde çalıyor. Gerçek notalar verildiğinde optimizer'ın tel ve perde isabeti test setinde 0.575'ten 0.717'ye çıktı. Kendi kayıtların için yeniden ayarlamak istersen:

```bash
python scripts/tune_tab.py --split val --set paths.splits_dir=ml/data/splits/guitarset_full
```

Optimizer'a ek olarak, tel ve perdeyi doğrudan öğrenen bir **tab kafası** da kullanılabilir. Tab kafası pozisyon seçiminde çok başarılı, ama paylaşılan katmanları perde tespitinden biraz uzaklaştırıyor. Bu yüzden en iyi sonuç iki modelin birlikte kullanılmasıyla alınıyor: notalar normal gitar modelinden (`MODEL_CHECKPOINT`), tel/perde olasılıkları tab kafalı modelden (`MODEL_TAB_CHECKPOINT`, `ml/configs/guitar_tab.yaml`) geliyor. Model olasılıkları optimizer'ın maliyetine `-log p` terimi olarak ekleniyor; böylece model yönlendiriyor, optimizer akorların çalınabilir olmasını ve el hareketinin az olmasını sağlıyor.

GuitarSet test sonuçları (oyuncu 05, eğitimde hiç görülmedi):

| Kurulum | Nota F1 | Onset + offset F1 | Tel/perde isabeti | Tam doğru nota* |
|---|---|---|---|---|
| Gitar modeli + optimizer | **0.900** | **0.786** | 0.70 | 0.59 |
| Tab kafalı model tek başına | 0.844 | 0.691 | 0.95 | 0.77 |
| **Gitar modeli + tab modeli** | **0.900** | **0.786** | **0.95** | **0.81** |

\* Referans notalar içinde hem doğru tespit edilip hem de doğru tel/perdeye konanların oranı.

## Dizin yapısı

| Dizin | İçerik |
|---|---|
| `apps/web` | Next.js arayüzü: yükleme, proje listesi, editör, ayarlar |
| `apps/api` | FastAPI + SQLAlchemy + Celery (transkripsiyon işleri) |
| `ml/configs` | `base.yaml` (ortak) ve `guitar.yaml` (GuitarSet) |
| `ml/preprocessing` | ses yükleme, log-mel/CQT, JAMS anotasyonları, augmentation |
| `ml/datasets` | GuitarSet keşfi, oyuncu bazlı split, PyTorch dataset |
| `ml/models` | `ConvStack`, `CNN`, `CRNN` (BiLSTM/GRU), çıkış kafaları |
| `ml/training` | eğitim döngüsü, loss, callback'ler |
| `ml/evaluation` | frame/nota/tab metrikleri, değerlendirme, görselleştirme |
| `ml/inference` | nota çözümleme (Onsets & Frames tarzı) ve `Predictor` |
| `packages/music-core` | `Note`, MIDI, zamanlama, tab optimizasyonu (`music_core` olarak import edilir) |
| `packages/shared-types` | API şemalarının TypeScript karşılıkları |
| `scripts` | veri indirme/hazırlama, eğitim, değerlendirme |

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[ml,dev]"
```

## Veri seti stratejisi

**GuitarSet** (360 kayıt, 6 oyuncu; pitch, onset/offset ve tel/perde anotasyonları) kullanılıyor. Split **oyuncu bazlı** yapılıyor, rastgele frame split kullanılmıyor. Böylece model, eğitimde hiç görmediği bir gitariste karşı test ediliyor:

| Split | Oyuncular |
|---|---|
| train | 00–03 |
| val | 04 |
| test | 05 |

```bash
python scripts/download_dataset.py                    # anotasyonlar + mono mic ses (Zenodo)

# V0: pipeline'ı hızlıca çalıştırmak için oyuncu başına 5 kayıt
python scripts/prepare_dataset.py --subset 5

# V1: tam veri seti
python scripts/prepare_dataset.py

# V2: + çevrimdışı augmentation (yalnızca train split'i)
#     pitch shift +1/-2, tempo 0.9x/1.1x, gürültü, reverb, EQ, gain,
#     overdrive, distortion, echo, telefon kaydı, büyük oda, düşük SNR gürültü.
#     Etiketler de dönüştürülür: pitch/perde kayar, onset/offset ölçeklenir.
python scripts/prepare_dataset.py --augment

# log-mel yerine CQT
python scripts/prepare_dataset.py --set features.type=cqt
```

Validation ve test setlerine hiçbir zaman augmentation uygulanmaz. İleride MAESTRO (genel AMT ön eğitimi) ve Slakh2100 gitar kanalları (~105 GB) eklenebilir.

## Eğitim ve değerlendirme

```bash
python scripts/train.py --config ml/configs/guitar.yaml
python scripts/train.py --config ml/configs/guitar.yaml --set model.rnn_type=gru training.epochs=30
python scripts/evaluate.py --checkpoint ml/checkpoints/guitar/best.pt --split test --plot

# Tab kafalı ikinci model ve birlikte değerlendirme
python scripts/train.py --config ml/configs/guitar_tab.yaml --set paths.splits_dir=ml/data/splits/guitarset_aug training.repeats=1
python scripts/evaluate.py --checkpoint ml/checkpoints/guitar/best.pt --tab-checkpoint ml/checkpoints/guitar_tab/best.pt --split test

python -m ml.inference.predict kayit.wav --tab-checkpoint ml/checkpoints/guitar_tab/best.pt --midi kayit.mid --tab kayit.txt
```

Değerlendirme şu metrikleri raporlar: frame F1, nota F1 (yalnız onset / onset+offset, `mir_eval`), tab doğruluğu (doğru tespit edilen notaların doğru tel ve perdeye yerleşme oranı) ve ortalama güven.

## Uygulamayı çalıştırma

### Docker

```bash
docker compose up --build
```

- Web: http://localhost:3000
- API: http://localhost:8000/docs

Model checkpoint'i `ml/checkpoints/guitar/best.pt` konumunda olmalıdır. Checkpoint yoksa yüklenen kayıtlar açıklayıcı bir hata mesajıyla **Başarısız** durumuna düşer.

### Lokal geliştirme

API ayarları `apps/api/.env` dosyasından okunuyor. Örnek dosyayı kopyalayıp hangi modelin kullanılacağını (`MODEL_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v4/best.pt`, tab modeli için `MODEL_TAB_CHECKPOINT`) orada seç. Göreli yollar proje köküne göre çözülüyor; veritabanı ve yüklenen sesler varsayılan olarak `apps/api` altında tutuluyor.

```bash
cp apps/api/.env.example apps/api/.env
```

```bash
# API: USE_CELERY=false iken işler Redis olmadan süreç içinde çalışır
cd apps/api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
# Web
cd apps/web
npm install
npm run dev
```

API ortam değişkenleri: `DATABASE_URL`, `REDIS_URL`, `USE_CELERY`, `STORAGE_DIR`, `MODEL_CHECKPOINT`, `MODEL_TAB_CHECKPOINT`, `MODEL_DEVICE`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_AUDIO_MINUTES`, `MAX_NOTES`, `JOB_TIMEOUT_MINUTES`, `EXPOSE_DOCS`. Web için: `NEXT_PUBLIC_API_URL`. Docker Compose'da veritabanı parolası `POSTGRES_PASSWORD` ile değiştirilebilir.

## Dışa aktarma ve ritim

- **MIDI**: notalar saniye cinsinden, tahmin edilen tempoyla.
- **MusicXML**: tek gitar partisi, iki dizek (standart nota + 6 telli TAB, her notanın tel/perdesi). Onset'ler tempoya göre 16'lık ızgaraya hizalanır, 4/4 ölçülere bölünür; ölçü çizgisini aşan notalar bağlanır, boşluklar sus olur. Çıktı resmi MusicXML 4.0 şemasına göre geçerlidir; MuseScore ve Guitar Pro (Dosya → İçe aktar) açar.
- **ASCII TAB**: düz metin; yakın notalar birleşmez, akorlar hizalıdır.

Tempo, modelin onset tahminlerinden bulunur (GuitarSet testinde yarım/iki katı hatalar dahil %82 isabet; ses tabanlı `beat_track` ile %70). Swing/triplet hisli parçalarda yanılabilir; editördeki **BPM** alanından düzeltilip kaydedildiğinde MIDI ve MusicXML ritmi bu tempoya göre yazılır.

## Gerçek kayıtlar ve şarkı modu

**Dayanıklılık ölçümü.** Gerçek dünya kayıtlarında referans nota yok. Bu yüzden görülmemiş oyuncu 05'in kayıtları, notaları değiştirmeyen efektlerle bozularak ölçülüyor (`scripts/benchmark_robustness.py`):

| Koşul | Nota F1 (efektsiz → efektli eğitim) | Tam doğru nota (efektsiz → efektli eğitim) |
|---|---|---|
| Temiz | 0.900 → 0.894 | 0.812 → 0.786 |
| Telefon kaydı | 0.873 → 0.880 | 0.719 → 0.738 |
| Gürültü (10 dB) | 0.876 → 0.876 | 0.712 → 0.729 |
| Büyük oda | 0.819 → 0.846 | 0.735 → 0.712 |
| Overdrive | 0.796 → 0.855 | 0.601 → **0.739** |
| Distortion | 0.774 → 0.845 | 0.551 → **0.713** |
| Echo | 0.672 → 0.764 | 0.762 → 0.749 |
| **Ortalama** | | 0.699 → **0.738** |

"Efektsiz" pitch shift, tempo, gürültü, reverb, EQ ve gain varyantlarıyla eğitilmiş model çiftidir. "Efektli" buna ek olarak overdrive, distortion, echo, telefon, büyük oda ve düşük SNR gürültü varyantlarıyla eğitilmiştir (`--augment`). Efektli eğitim elektro gitar efektlerinde büyük kazanç sağlıyor, temiz kayıtlarda ise 2.6 puan kaybettiriyor. Uygulamanın varsayılanı efektli çift. Echo'da F1 düşük, çünkü yankılar da nota olarak yazılıyor. Test efektleri eğitimdeki efekt türleriyle aynı; yalnızca kayıtların kendisi görülmemiş.

**Şarkı modu.** Davul, bas ya da vokal içeren kayıtlarda, yükleme sırasında "Şarkı / grup kaydı" seçeneği açılırsa gitar önce Demucs `htdemucs_6s` ile ayrılır. Model ağırlıkları ilk kullanımda indirilir. Seçenek editörde "Gitarı ayır" ile değiştirilip yeniden çözümlenebilir. Ölçüm, test gitarlarının MUSDB18 davul/bas/vokal kanallarıyla karıştırılmasıyla yapıldı:

| Kayıt | Ayırmasız: nota F1 / tam doğru | Şarkı modu: nota F1 / tam doğru |
|---|---|---|
| Yalnız gitar | 0.900 / 0.812 | 0.893 / 0.815 |
| Grup, gitar eşit seviyede | 0.523 / 0.514 | **0.847 / 0.743** |
| Grup, gitar 6 dB kısık | 0.384 / 0.328 | **0.780 / 0.635** |

Ayırma, yalnız gitar kayıtlarında sonucu neredeyse değiştirmiyor. Varsayılan olarak kapalı, çünkü işlem süresini artırıyor: GPU'da 22 saniyelik ses 2.6 saniyede ayrılıyor, CPU'da çok daha yavaş.

```bash
python -m ml.inference.predict sarki.wav --separate --tab-checkpoint ml/checkpoints/guitar_tab/best.pt --tab sarki.txt
```

## Güvenlik

**API'de kimlik doğrulama yok.** API tek kullanıcılı, yerel çalışma için tasarlandı. Bu yüzden Docker Compose tüm portları yalnızca `127.0.0.1`'e bağlıyor. Başka makinelerden erişilecekse önüne kimlik doğrulayan bir reverse proxy konulmalı.

Mevcut korumalar:

- **Yükleme doğrulaması:** Yalnızca izin verilen uzantılar kabul ediliyor ve dosyanın içeriği magic bytes ile kontrol ediliyor. `.wav` uzantılı bir HTML dosyası reddediliyor. Dosyalar rastgele adlarla ve gerçek formatlarına uygun uzantıyla kaydediliyor.
- **Boyut sınırları:** İstek gövdesi akış sırasında ölçülüyor, sınırı aşan yükleme diske tamamen yazılmadan kesiliyor ve 413 dönüyor. Ses süresi `MAX_AUDIO_MINUTES` ile, editörden kaydedilen nota sayısı `MAX_NOTES` ile sınırlı. Nota değerlerinin de üst sınırları var: NaN ve sonsuz değer yok, zaman en fazla 6 saat.
- **CSRF:** İzin verilen origin'lerin (`CORS_ORIGINS`) dışından gelen POST, PUT ve DELETE istekleri 403 ile reddediliyor. Origin başlığı göndermeyen istemciler (curl, betikler) bu kontrolden etkilenmiyor.
- **Başlıklar:** `nosniff`, `X-Frame-Options: DENY` ve API yanıtlarında `Content-Security-Policy: default-src 'none'` gönderiliyor.
- **Hata mesajları:** İç hata ayrıntıları yalnızca sunucu log'una yazılıyor, kullanıcıya genel bir mesaj gösteriliyor. Doğrulama hataları gönderilen değeri geri yansıtmıyor.
- **Dosya yolları:** Dosya adları temizleniyor. Silme ve sunma işlemleri yalnızca depolama klasörünün içindeki dosyalarla sınırlı.
- **İş kurtarma:** `JOB_TIMEOUT_MINUTES` süresinden uzun takılı kalan işler yeniden başlatılabiliyor. Süreç içi modda sunucu yeniden başlarken yarım kalan işler "başarısız" olarak işaretleniyor.
- **Konteynerler:** Konteynerler root olmayan bir kullanıcıyla çalışıyor ve model klasörü salt-okunur bağlanıyor.

## API

| Metot | Yol | Açıklama |
|---|---|---|
| `POST` | `/api/projects` | Ses yükle (`file`, isteğe bağlı `name`), transkripsiyonu başlat |
| `GET` | `/api/projects` | Projeleri listele |
| `GET` | `/api/projects/{id}` | Proje durumu |
| `DELETE` | `/api/projects/{id}` | Projeyi ve ses dosyasını sil |
| `POST` | `/api/projects/{id}/retranscribe` | Yeniden çözümle |
| `GET` | `/api/projects/{id}/audio` | Orijinal ses |
| `GET` | `/api/projects/{id}/transcription` | Notalar, tempo, akort, ortalama güven |
| `PUT` | `/api/projects/{id}/notes` | Editörde düzenlenen notaları (ve isteğe bağlı düzeltilmiş tempoyu) kaydet; tel/perdesi olmayan notalara optimizer pozisyon atar |
| `GET` | `/api/projects/{id}/midi` | MIDI dışa aktar |
| `GET` | `/api/projects/{id}/musicxml` | Nota + TAB içeren MusicXML (MuseScore, Guitar Pro) |
| `GET` | `/api/projects/{id}/tab` | ASCII TAB dışa aktar |

## Testler

```bash
pytest
```
