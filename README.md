# Music Transcriber

Gitar kayıtlarını nota olaylarına, MIDI'ye ve gitar tab'ına dönüştüren uçtan uca bir otomatik müzik transkripsiyonu (AMT) projesi. Gitar yoksa söylenen, mırıldanan ya da ıslıkla çalınan bir melodinin de TAB'ı çıkarılabilir (bkz. [Ses → gitar](#ses--gitar-söyle-tabını-al)). Bir şarkı verilirse gitarla çalınacak akorları (şemalar, slash akorlar, capo önerisi, ton değişimleri, ritim kalıbı) ve vokal melodinin TAB'ı çıkarılır (bkz. [Şarkı → gitar](#şarkı--gitar-akorlar-ve-melodi)). Ölçü ızgarası kayıttaki vuruşları izler; metronomsuz çalınmış kayıtlarda da ölçüler müziğe oturur (bkz. [Vuruş takibi](#vuruş-takibi-metronomsuz-kayıtlar)).

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
Çalma teknikleri (nota başına perde eğrisi: bend, vibrato, slide) ──► TAB / MusicXML işaretleri
   │
   ▼
Müzik analizi (vuruş takibi, ton, ölçü ızgarası, akorlar, ritim kalıbı) ──► MusicXML armür, ölçü, tempo, akorlar
   │
   ▼
Müzik editörü (dalga formu, TAB + akorlar, piyano rulosu, nota görünümü, sentezle dinleme, döngü,
               metronom, sürükle-bırak düzenleme, geri al, MIDI/MusicXML/TAB dışa aktarma)
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

Bu tablo yalnızca GuitarSet ile eğitilmiş ilk çiftin sonuçları. Uygulamanın varsayılanı artık elektro gitar ve oda kayıtlarıyla da eğitilmiş, C2'ye kadar inen V10 + V7 çifti (bkz. [Ek veri setleri](#ek-veri-setleri-elektro-gitar-ve-oda-mikrofonları), [Oda mikrofonları ve hizalanmış EGDB](#oda-mikrofonları-ve-hizalanmış-egdb-v10) ve [Farklı akortlar ve capo](#farklı-akortlar-ve-capo)).

## Dizin yapısı

| Dizin | İçerik |
|---|---|
| `apps/web` | Next.js arayüzü: yükleme, proje listesi, editör, ayarlar |
| `apps/api` | FastAPI + SQLAlchemy + Celery (transkripsiyon işleri) |
| `ml/configs` | `base.yaml` (ortak), `guitar.yaml` (GuitarSet), `guitar_tab.yaml` (tab kafası), `guitar_mixed.yaml` / `guitar_tab_mixed.yaml` (ek veri setleriyle), `guitar_mixed_low.yaml` (C2'ye inen nota aralığı), `guitar_contour.yaml` (perde eğrisi kafası) |
| `ml/preprocessing` | ses yükleme, log-mel/CQT, JAMS ve tel başına MIDI anotasyonları, augmentation |
| `ml/datasets` | GuitarSet, EGDB (sese hizalanmış etiketleriyle) ve Guitar-TECHS keşfi ve split'leri, PyTorch dataset; ölçümler için AAM (`aam.py`) ve GuitarSet vuruşları (`strums.py`) |
| `ml/models` | `ConvStack`, `CNN`, `CRNN` (BiLSTM/GRU), çıkış kafaları; akor tanıyıcı BTC (`btc.py`, yalnızca çıkarım) |
| `ml/training` | eğitim döngüsü, loss, callback'ler |
| `ml/evaluation` | frame/nota/tab metrikleri, değerlendirme, görselleştirme |
| `ml/inference` | nota çözümleme (Onsets & Frames tarzı), `Predictor`, kanal ayırma (Demucs), ses modu (`voice.py`), akor, bas ve vuruş tanıma (`chords.py`), sesten vuruş onset'leri (`rhythm.py`), şarkı modu (`song.py`) |
| `packages/music-core` | `Note`, MIDI, MusicXML, vuruş ızgarası (`timing.py`), tab optimizasyonu, ton/ölçü/akor analizi ve ton değişimleri (`analysis.py`), ritim kalıbı (`rhythm.py`), gitar akor şemaları, slash akorlar ve capo önerisi (`guitar_chords.py`); `music_core` olarak import edilir |
| `packages/shared-types` | API şemalarının TypeScript karşılıkları |
| `scripts` | veri indirme/hazırlama, eğitim, değerlendirme |
| `notebooks` | Colab'da eğitim (perde eğrisi modeli) |

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[ml,dev]"
pip install -e ".[train]"       # yalnızca eğitim verisi hazırlamak için: oda simülasyonu
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

Validation ve test setlerine hiçbir zaman augmentation uygulanmaz.

### Ek veri setleri: elektro gitar ve oda mikrofonları

GuitarSet yalnızca akustik gitar ve tek bir oda içeriyor. İki açık veri seti daha ekleniyor:

| Veri seti | İçerik | Etiket | Lisans | Test |
|---|---|---|---|---|
| [EGDB](https://ss12f32v.github.io/Guitar-Transcription/) | Tek gitaristin 240 elektro gitar klibi; DI kaydı ve 5 amfi tonu (Marshall, Fender Twin, Mesa, JC Jazz, Plexi); 10.8 saat | Tel başına MIDI | serbest kullanım | 216–240 numaralı klipler, her ton; JC Jazz ve Plexi hiç eğitilmiyor |
| [Guitar-TECHS](https://zenodo.org/records/14963133) | 3 gitarist; akor, gam, tekli nota, teknik ve müzik parçaları; her çekim DI, amfi mikrofonu, kafaya takılı ve 1.5 m öndeki oda mikrofonuyla; 20 saat | Tel başına MIDI pickup | CC BY 4.0 | Hiç görülmeyen 3. gitaristin müzik parçaları |

```bash
python scripts/download_egdb.py            # Google Drive, ~7 GB
python scripts/download_guitar_techs.py    # Zenodo, ~4 GB zip + ~7 GB açılmış
python scripts/prepare_dataset.py --dataset egdb
python scripts/prepare_dataset.py --dataset guitar_techs
python scripts/mix_splits.py --out ml/data/splits/mixed_notes guitarset=ml/data/splits/guitarset_fx guitar_techs=ml/data/splits/guitar_techs
python scripts/mix_splits.py --out ml/data/splits/mixed_tab guitarset=ml/data/splits/guitarset_fx egdb=ml/data/splits/egdb guitar_techs=ml/data/splits/guitar_techs

# GuitarSet modellerinden ince ayar (sıfırdan eğitmekten çok daha kısa)
python scripts/train.py --config ml/configs/guitar_mixed.yaml --init ml/checkpoints/guitar_v4/best.pt --set training.lr=2e-4 training.epochs=30 paths.checkpoint_dir=ml/checkpoints/guitar_v8
python scripts/train.py --config ml/configs/guitar_tab_mixed.yaml --init ml/checkpoints/guitar_v5/best.pt --set training.lr=2e-4 training.epochs=30 paths.checkpoint_dir=ml/checkpoints/guitar_v7

# Her test setinde, kayıt koşuluna göre karşılaştırma
python scripts/benchmark_datasets.py --model eski=ml/checkpoints/guitar_v4/best.pt+ml/checkpoints/guitar_v5/best.pt --model yeni=ml/checkpoints/guitar_v8/best.pt+ml/checkpoints/guitar_v7/best.pt
```

Etiketlerde bulunan ve düzeltilen sorunlar:

- **Tel bilgisinin yeri:** EGDB'de kanal numarasında (iz adları çoğu dosyada eksik ya da tutarsız). Guitar-TECHS'te ise iz adında ("e", "B", ... "E"), çünkü bazı çekimlerde bütün teller kanal 0'da.
- **Guitar-TECHS zamanlaması:** MIDI bazı çekimlerde DI kaydından 65 ms'ye kadar kayık, video seslerinin başlangıcı da 50 ms'ye kadar farklı. Etiketler önce DI'nın onset zarfına, sonra her kaydın DI'ya göre kaymasına (çapraz korelasyon) hizalanıyor.
- **EGDB'nin "RealData" klipleri:** kullanılmıyor. MIDI izlerinin adı `clip1_pred` vb.; makalenin model çıktısı gibi görünüyor, referans değil.
- **EGDB nota zamanlaması:** etiketlerin çoğu tablatürün nicelenmiş zamanlamasını taşıyor, performansınkini değil. Ortanca klipte onset'lerin yarısı tam 16'lık ızgarada; Guitar-TECHS'te bu oran %2. Birkaç klip (40, 69, 116) sesiyle hiç uyuşmuyor. EGDB ile eğitilen nota modeli, EGDB'nin kendi testinde bile daha kötü oldu (Marshall tonunda nota F1 0.768 → 0.729). Bu yüzden EGDB yalnızca tab modelini eğitiyor; tab modelinin kare başına tel/perde hedefleri bundan etkilenmiyor.
- **Guitar-TECHS akor çekimleri:** pickup kalın telleri geç yazıyor. La ve kalın mi tellerinde onset'lerin %10'u 120–130 ms gecikmeli. Ama bu çekimleri çıkarmak her test setinde sonucu kötüleştirdi, o yüzden eğitimde kalıyorlar.

Sonuç (nota modeli V8: GuitarSet + Guitar-TECHS; tab modeli V7: üçü birden; ikisi de V4/V5'ten 30 epoch ince ayar):

| Test (eğitimde görülmemiş) | Nota F1: V4+V5 → V8+V7 | Tel/perde isabeti: V4+V5 → V8+V7 |
|---|---|---|
| GuitarSet (akustik, mikrofon) | 0.894 → 0.896 | 0.925 → 0.935 |
| EGDB DI | 0.805 → 0.821 | 0.734 → 0.805 |
| EGDB, eğitimdeki amfi tonları (ort.) | 0.769 → 0.786 | 0.692 → **0.842** |
| EGDB, hiç görülmeyen tonlar (JC Jazz, Plexi) | 0.765 → 0.783 | 0.689 → **0.833** |
| Guitar-TECHS DI / amfi mikrofonu | 0.633 → 0.658 | 0.526 → 0.629 |
| Guitar-TECHS oda mikrofonları | 0.535 → **0.571** | 0.521 → 0.605 |

Nota F1'in EGDB'de düşük görünmesinin bir sebebi de yukarıdaki nicelenmiş referans zamanlaması. Guitar-TECHS'te ise gam ve akor egzersizlerinden gerçek müziğe genelleme ölçülüyor.

### Oda mikrofonları ve hizalanmış EGDB (V10)

En zayıf nokta, odada mikrofonla (telefon, kamera) kaydedilmiş elektro gitardı. Bunun için iki şey denendi, ikisi de V9'dan 20 epoch ince ayarla:

- **Simüle odalar ve codec'ler (V10a):** GuitarSet'in eğitim kayıtları, oturma odasından prova odasına kadar rastgele odalarda (image-source yöntemi, `pyroomacoustics`; mikrofon 0.5–4 m uzakta, yansıma süresi 0.2–1.0 s), mikrofon renklendirmesi ve oda gürültüsüyle (`room`, `roomfar`), ayrıca düşük bit hızlı Opus/AAC/MP3'le (`codec`, `roomcodec`; ffmpeg) yeniden kaydedildi. Eskiden kullanılan `reverb` yalnızca gürültü kuyruğuydu; bunda erken yansımalar ve mesafeye göre doğrudan/yansıyan ses oranı var.
- **EGDB'yi nota modeline katmak (V10 = V10a + bu):** EGDB etiketlerinin onset'leri sese göre kayık (GuitarSet ile eğitilmiş V9'a göre ortanca +3 ms, ama dörtte biri 25 ms'den uzak). `scripts/align_egdb.py` her etiketi, V9'un aynı perdede ±60 ms içinde bulduğu onset'e taşıyor (etiketlerin %80'i taşındı, ortanca 13 ms). Model onset bulamazsa etiket yerinde kalıyor, modelin bulup etiketin söylemediği nota eklenmiyor. Etiketleriyle sesi uyuşmayan 9 klip (V9'a göre onset F1 < 0.5; README'de daha önce şüphelenilen 40, 69, 116 dahil) atıldı. Amfi tonları (DI değil) ayrıca simüle odalardan geçirildi: odadaki amfi, tam hedeflenen koşul.

Önce denenen, CQT'deki perde başına enerji artışıyla hizalama işe yaramadı: bas notalarında CQT penceresi uzun olduğundan artış notadan önce görünüyor ve etiketler ~15 ms erkene kaydı (V9'un onset F1'i hizalanmış etiketlere karşı 0.787'den 0.765'e düştü).

```bash
pip install -e ".[train]"
python scripts/prepare_dataset.py --augment --variants room roomfar codec roomcodec --set paths.splits_dir=ml/data/splits/guitarset_room
python scripts/align_egdb.py --checkpoint ml/checkpoints/guitar_v9/best.pt
python scripts/prepare_dataset.py --dataset egdb_aligned --augment --variants room
python scripts/mix_splits.py --out ml/data/splits/mixed_room_egdb guitarset=ml/data/splits/guitarset_fx guitarset=ml/data/splits/guitarset_room guitar_techs=ml/data/splits/guitar_techs egdb_aligned=ml/data/splits/egdb_aligned
python scripts/train.py --config ml/configs/guitar_room_egdb.yaml --init ml/checkpoints/guitar_v9/best.pt --set training.lr=2e-4 training.epochs=20 paths.checkpoint_dir=ml/checkpoints/guitar_v10
python scripts/benchmark_datasets.py --model v9=ml/checkpoints/guitar_v9/best.pt+ml/checkpoints/guitar_v7/best.pt --model v10=ml/checkpoints/guitar_v10/best.pt+ml/checkpoints/guitar_v7/best.pt
```

Nota F1, eğitimde görülmemiş testler (tab modeli üçünde de V7):

| Test | V9 | V10a (simüle odalar) | V10 (+ hizalı EGDB) |
|---|---|---|---|
| GuitarSet (akustik, mikrofon) | 0.895 | 0.898 | 0.895 |
| EGDB, 6 ton ortalaması (orijinal etiketler) | 0.792 | 0.793 | 0.790 |
| EGDB DI, hizalı etiketler | 0.830 | 0.830 | 0.837 |
| Guitar-TECHS DI / amfi mikrofonu | 0.673 / 0.667 | 0.669 / 0.656 | **0.680 / 0.673** |
| Guitar-TECHS oda mikrofonları (kafada / 1.5 m önde) | 0.589 / 0.574 | 0.587 / 0.572 | **0.617 / 0.594** |
| Guitar-TECHS oda, tel/perde isabeti (kafada / önde) | 0.571 / 0.637 | 0.576 / 0.644 | 0.589 / 0.644 |

- Simüle odalar tek başına gerçek oda kayıtlarında işe yaramadı (V10a): akustik gitarı odaya koymak, odadaki bir elektro gitar amfisine benzemiyor.
- Hizalanmış EGDB ile (V10) gerçek oda mikrofonlarında nota F1 2–3 puan arttı, DI ve amfi mikrofonunda da ~1 puan. GuitarSet aynı kaldı, EGDB'nin amfi tonlarında ortalamada 0.002 düştü (ton başına 25 klip; gürültü düzeyinde).
- "EGDB hizalı" testi, hizalama V9'un onset'leriyle yapıldığı için V9'a da yakın: orijinal etiketlere göre her model için 0.01 kadar yüksek çıkıyor. Modelleri karşılaştırmak için asıl ölçüt diğer satırlar.

GuitarSet oyuncu 05, bozulmalar simüle edilerek ([Dayanıklılık ölçümü](#gerçek-kayıtlar-ve-grup-kayıtları); son iki koşul yeni) ve akortlar kaydırılarak (`scripts/benchmark_tunings.py`), nota F1:

| Koşul | V9 | V10a | V10 |
|---|---|---|---|
| Temiz | 0.895 | 0.898 | 0.895 |
| Telefon / gürültü 10 dB | 0.885 / 0.887 | 0.886 / 0.891 | 0.887 / 0.893 |
| Büyük oda (eski reverb) / overdrive / distortion / echo | 0.859 / 0.858 / 0.851 / 0.772 | 0.859 / 0.861 / 0.853 / 0.780 | 0.853 / 0.855 / 0.846 / 0.766 |
| Simüle oda mikrofonu (2–4 m) / sesli not (oda + codec) | 0.850 / 0.862 | 0.847 / 0.861 | 0.844 / 0.856 |
| Yarım ton / bir ton / iki ton aşağı / capo 2 | 0.881 / 0.878 / 0.856 / 0.875 | | 0.880 / 0.875 / 0.856 / 0.877 |

Simüle koşullarda V10, V9'dan çoğunlukla 0.3–0.6 puan geride, gürültü ve telefonda biraz önde; akortlarda aynı. Gerçek oda kayıtlarındaki 2–3 puanlık kazanç bundan daha ağır bastığı için uygulamanın varsayılan nota modeli artık V10. Simülasyonla yapılan bu ölçümler, simülasyonla eğitilen modeller için iyimser; gerçek kayıtlar (Guitar-TECHS) asıl ölçüt.

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

Farklı verilerle eğitilen modellerin olasılık dağılımı kayabilir; nota çözümleme eşiklerini doğrulama setinde denemek (ve istenirse checkpoint'e yazmak) için:

```bash
python scripts/tune_thresholds.py --checkpoint ml/checkpoints/guitar_v8/best.pt --split ml/data/splits/mixed_notes/val.txt
```

Değerlendirme şu metrikleri raporlar: frame F1, nota F1 (yalnız onset / onset+offset, `mir_eval`), tab doğruluğu (doğru tespit edilen notaların doğru tel ve perdeye yerleşme oranı) ve ortalama güven.

**Uzun eğitimleri kendi terminalinden başlat.** Bir editör ya da asistan oturumunun başlattığı süreçler, o oturum kapanınca onunla birlikte kapanabilir. PowerShell'de eğitimi ayrı bir pencerede başlatıp çıktısını dosyaya yazmak için:

```powershell
Start-Process python -ArgumentList "scripts/train.py --config ml/configs/guitar_mixed.yaml" -RedirectStandardOutput train.log -RedirectStandardError train.err -WindowStyle Minimized
```

Yarıda kalan bir eğitim `--resume ml/checkpoints/<klasör>/last.pt` ile kaldığı epoch'tan sürer.

## Uygulamayı çalıştırma

### Docker

```bash
docker compose up --build                                                  # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build  # NVIDIA GPU
```

- Web: http://localhost:3000 (üretim derlemesi, `next start`)
- API: http://localhost:8000/docs

Modeller `ml/checkpoints` klasöründen salt-okunur bağlanır. Varsayılan çift `guitar_v10` (notalar) ve `guitar_v7` (tel/perde); başka klasörler `NOTES_MODEL` ve `TAB_MODEL` ortam değişkenleriyle (ör. proje kökündeki `.env`) seçilir. Checkpoint yoksa yüklenen kayıtlar açıklayıcı bir hata mesajıyla **Başarısız** durumuna düşer. CPU imajı yeterince hızlı: 22 saniyelik bir kaydın notalarını CPU'da 0.7 s'de çözüyor, vuruş takibi ve akor tanıma 1–2 s daha ekliyor (BTC ve Beat This! indirilemezse gitar projeleri tek tempoyla ve yalnız notalardan akorla devam eder). GPU asıl kanal ayırmada (Demucs: grup kayıtları ve şarkılar) fark ediyor ve imajı birkaç GB büyütüyor. Demucs, BTC (akorlar, 12 MB) ve Beat This! (vuruşlar, 81 MB) ağırlıkları ilk kullanımda indirilip `model-cache` volume'unda saklanır.

### Lokal geliştirme

API ayarları `apps/api/.env` dosyasından okunuyor. Örnek dosyayı kopyalayıp hangi modelin kullanılacağını (`MODEL_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v10/best.pt`, tab modeli için `MODEL_TAB_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v7/best.pt`) orada seç. Göreli yollar proje köküne göre çözülüyor; veritabanı ve yüklenen sesler varsayılan olarak `apps/api` altında tutuluyor.

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

## Editörde dinleme ve çalışma

- **Dinle: Kayıt / Notalar / Kayıt + notalar.** Transkripsiyon tarayıcıda sentezlenmiş tel sesiyle (Karplus-Strong) çalınır. Kayıtla üst üste dinleyince yanlış nota kulakla hemen bulunur. Kayıt saat olarak kullanılır, bu yüzden hız değişince ve döngüde atlayınca senkron kalır.
- **Döngü:** Dalga formunda sürükleyerek ya da **A** / **B** tuşlarıyla (oynatma konumu) seçilir, **L** kaldırır. Hız ayarıyla birlikte zor bir pasajı yavaş ve tekrar tekrar çalışmak için.
- **Metronom** (**M**): tahmini vuruş ızgarasında tıklar, ölçü başları vurgulu. Tempo ve ölçü başının doğru olup olmadığını duymanın en kolay yolu.
- **Izgara ve akorlar:** TAB ve piyano rulosunda vuruş ve ölçü çizgileri (kayıttaki vuruşları izler), ölçü numaraları (MusicXML ölçüleriyle aynı), TAB'ın üstünde akor sembolleri ve ritim kalıbı görünür. Bir notayı düzenleyince ya da BPM, ölçü, ızgara veya tonu değiştirince bunlar kaydetmeden güncellenir (`POST /api/analysis`).
- **Gitar sesi indir:** Notalar (kaydedilmemiş düzenlemeler dahil) aynı sentezlenmiş gitar sesiyle tarayıcıda WAV'a çevrilip indirilir. Söylenen bir melodiyi gitar sesi olarak almanın yolu.
- **Nota görünümü:** "Notayı göster" ile MusicXML çıktısının kendisi (standart nota + TAB, armür, ölçü, akorlar) tarayıcıda [OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/) ile çizilir. Kaydedilmemiş düzenlemeler de görünür (`POST /api/render/musicxml`); dışa aktarmadan önce MuseScore'da nasıl görüneceğini kontrol etmek için.

## Düzenleme

Notalar piyano rulosunda ve TAB'da doğrudan fareyle düzenlenir:

| İşlem | Nasıl |
|---|---|
| Seçmek | Tıkla; **Shift/Ctrl** + tıkla ekler/çıkarır; boş alanda sürükleyerek dikdörtgenle seç; **Ctrl+A** hepsi |
| Taşımak | Piyano rulosunda seçili notaları sürükle (yatay: zaman, dikey: perde); TAB'da yatay sürükleme zamanı değiştirir |
| Başka tele almak | TAB'da perde rakamını yukarı/aşağı sürükle: nota aynı perdede kalır, başka telde çalınabiliyorsa oraya geçer |
| Uzatmak/kısaltmak | Piyano rulosunda notanın sağ kenarını sürükle |
| İnce ayar | **↑/↓** yarım ton (**Shift**: oktav), **←/→** 10 ms (**Shift**: 100 ms) |
| Kopyala/yapıştır | **Ctrl+C** / **Ctrl+V**; yapıştırılan notalar oynatma konumundan başlar |
| Silmek | **Delete** |
| Geri al / yinele | **Ctrl+Z** / **Ctrl+Y** (ya da **Ctrl+Shift+Z**) ve araç çubuğundaki düğmeler; bir sürükleme tek adım sayılır |

Nota panelinden ayrıca notanın tekniği (hammer-on, pull-off, slide: önceki notadan aynı telde, pena vurmadan), bend miktarı ve vibratosu seçilir; TAB bunları yay, çizgi ve `7b9`/`7~` etiketleriyle gösterir.

Kaydedilmiş hale geri alınınca "kaydedilmemiş değişiklik" uyarısı da kalkar. Boş alana tıklamak oynatma konumunu oraya taşır.

## Yükleme, kayıt ve ilerleme

- **Mikrofonla kayıt:** Yükleme sayfasında "Mikrofonla kaydet" tarayıcıda WAV kaydeder (en fazla 8 dakika, seviye göstergesiyle) ve dosya seçmişsin gibi yükler. Yankı giderme, gürültü bastırma ve otomatik kazanç kapalı: konuşma için yapılmış bu işlemler gitar notalarını bozuyor. Tarayıcılar mikrofona yalnızca `localhost` ya da HTTPS üzerinden izin verir.
- **İlerleme:** Çözümleme sürerken proje listesi ve editör aşamayı (yükleniyor, gitar ayrılıyor, notalar çözülüyor, son işlemler) ve yüzdeyi gösterir. Model, uzun kayıtları parça parça işlerken ilerlemeyi bildirir; API bunu en fazla yarım saniyede bir veritabanına yazar (`progress`, `stage`).

## Ses → gitar: söyle, TAB'ını al

Yüklerken "Ne kaydettiniz?" sorusuna **Ses (şarkı, mırıldanma, ıslık)** cevabı verilirse gitar modeli çalışmaz; tek sesli bir melodi notalara çevrilip seçilen akortta gitara yerleştirilir (`ml/inference/voice.py`). Sonrası gitar projeleriyle aynı: editör, TAB, MIDI/MusicXML ve **Gitar sesi indir** ile melodinin gitarla çalınmış hali (WAV). Eğitilmiş bir model yok, dolayısıyla eğitim de gerekmiyor:

1. **Perde takibi:** pYIN (librosa) sesin temel frekansını ve sesli olup olmadığını izler; YIN'in daha ince tahmini, ikisi uyuştuğunda onun yerine geçer.
2. **Kişisel akort:** Çoğu kişi tam A = 440 Hz'e göre söylemez. Kaydın ortalama sapması ölçülür ve yarım tonlar ona göre sayılır. 40 cent tiz söyleyen biri, her notası yarım ton yukarı yazılmadan transkribe edilir.
3. **Notalara bölme:** Kareleri tek tek en yakın yarım tona yuvarlamak vibratoyu ve uzun notalardaki kaymayı ayrı notalara böler. Bunun yerine, bir anın 200 ms öncesi ve sonrasının ortalama perdesi en az 0.7 yarım ton farklıysa oradan kesilir. Her nota, karelerinin medyan perdesini alır. Kısa parçalar (notalar arası kayış, notaya aşağıdan giriş) komşusuna katılır. Nota, perdesi yerine oturduğu yerde başlar: bir önceki notadan kayarak gelinen notada bu, kayışın notanın söylenen perdesine 0.2 yarım ton yaklaştığı (ya da onu geçtiği) andır. Aynı perdede tekrarlanan heceler ("da-da-da") ancak belirgin bir ünsüz izi varsa ayrılır: spektral değişim ile sesliliğin ve ses şiddetinin çukuru birlikte.
4. **Gitara yerleştirme:** Melodi gitarın aralığına sığmıyorsa (ıslık genelde bir iki oktav yukarıda) tam oktavlarla kaydırılır, editör bunu belirtir (`transpose`). Tel/perdeyi tab optimizer'ı seçer, tempo nota başlarından bulunur.

Arkada müzik varsa "Müzik eşliğinde söylenmiş" seçeneği önce Demucs ile vokali ayırır. Aynı seçenek gitar projelerinde gitarı ayırır.

**Editörde ses projeleri için:**

- **Söylenen perde:** Piyano rulosunda notaların üzerinde, söylenen perdenin eğrisi çizilir (`GET /api/projects/{id}/pitch`, saniyede 50 değer). Yanlış yazılmış bir nota, eğrinin başka bir satırda durmasından hemen anlaşılır; vibrato ve kayışlar da görünür.
- **Ritmi oturt** (1/8 ya da 1/16): Seçili notaların (seçim yoksa hepsinin) başını ve sonunu tempo ile ölçü başından çıkan vuruş ızgarasına oturtur. Metronomsuz söylenmiş bir melodiyi notaya dökmeden önce.
- **Gama oturt:** Tonun gamı dışında kalan notaları yarım ton yandaki gam notasına taşır; minörde yeden (A minörde G#) gamdan sayılır. İki yan da gamdaysa, yön söylenen perdeden (eğriden) seçilir, eğri yoksa bir önceki notaya doğru. Detone söylenmiş notaları toplu düzeltmek için.

İkisi de gitar projelerinde de çalışır ve her düzenleme gibi **Ctrl+Z** ile geri alınır.

**Ölçüm.** [VocalSet](https://zenodo.org/records/1442513) (20 profesyonel şarkıcı; gamlar, arpejler, uzun notalar ve üç kısa şarkı) ve her notanın başlangıç, bitiş ve perdesini işaretleyen [Annotated-VocalSet](https://zenodo.org/records/7061507) ile yapıldı (ikisi de CC BY 4.0). Ayarlar 6 şarkıcıda seçildi, aşağıdaki sonuçlar hiç kullanılmayan diğer 14 şarkıcıdan:

```bash
python scripts/benchmark_voice.py --vocalset ml/data/raw/vocalset/VocalSet11.zip --annotations ml/data/raw/vocalset/annotated_vocalset.zip
```

| Kayıtlar (14 şarkıcı) | Kayıt | Nota F1 (başlangıç ±50 ms) | Nota F1 (±100 ms) | Yalnız başlangıç F1 |
|---|---|---|---|---|
| Gamlar | 645 | 0.36 | 0.67 | 0.45 |
| Arpejler | 683 | 0.39 | 0.64 | 0.51 |
| Uzun notalar | 202 | 0.39 | 0.50 | 0.51 |
| Şarkılar | 77 | 0.41 | 0.60 | 0.50 |
| **Hepsi** | 1607 | **0.38** | **0.64** | 0.48 |

Etiketler notaların yazılı perdesini veriyor, ama kayıtların çoğu ondan yarım ton (bazen bir oktav) farklı söylenmiş. Bu yüzden nota F1 her kaydı kendi tonunda, ona en iyi uyan tam yarım ton kaydırmasıyla ölçüyor. ±50 ms ile ±100 ms arasındaki fark çoğunlukla, kayışla başlayan bir notanın tam olarak nerede başladığına dair tanım farkından geliyor. Legato notaları kayışın varışında başlatmak (ilk sürümde kayışın ortasındaydı) ±50 ms skorunu 0.32'den 0.38'e çıkardı.

**Perde takibi karşılaştırması** (ayar şarkıcıları, ±100 ms nota F1): pYIN 0.67, önceden eğitilmiş [CREPE](https://github.com/maxrmorrison/torchcrepe) (full) 0.63, CREPE tiny 0.53, [PESTO](https://github.com/SonyCSLParis/pesto) 0.64. Şarkılar müziğin üstüne karıştırılıp vokal Demucs ile geri ayrıldığında pYIN 0.61'den 0.57'ye düşüyor, CREPE de 0.58'de kalıyor. CREPE yalnızca üst üste kaydedilmiş ya da efektli vokallerde belirgin şekilde daha iyi (aşağıdaki MUSDB örneğinde karelerin %57'sini sesli buldu, pYIN %7). Bu tek durum için 89 MB'lık bir model eklemeye değmediğinden pYIN kaldı.

Sınırlar:

- Net ve ayrık söylenmiş melodilerde iyi çalışıyor. Hızlı pasajlar zorluyor.
- Aynı perdede sözle tekrarlanan notalar, ünsüz belirgin değilse tek nota olarak çıkıyor.
- Üst üste kaydedilmiş vokaller (armoni, dublaj) ve ağır efektli vokaller pYIN'e sesli görünmüyor. Bir MUSDB şarkısının vokal kanalında karelerin yalnızca %7'si sesli sayıldı.
- Tek sesli çalışıyor; aynı anda birden fazla nota (akor) çıkarmıyor.

## Şarkı → gitar: akorlar ve melodi

Yüklerken **Şarkı (melodi + akor)** seçilirse bir şarkının gitarla çalınacak hali çıkarılır (`ml/inference/song.py`): ölçü ölçü akorlar, her akorun parmak şeması, akorları kolaylaştıran capo ve vokal melodinin TAB'ı, akor isimleri üstünde. Şarkıda gitar olması gerekmiyor. Burada da eğitim yok, önceden eğitilmiş iki açık model kullanılıyor:

1. **Vokali ayırma:** Demucs `htdemucs_6s`.
2. **Vuruş ve ölçü:** [Beat This!](https://github.com/CPJKU/beat_this) (Foscarin ve ark., ISMIR 2024, MIT) vuruşları ve ölçü başlarını karışımın kendisinden bulur. Tempo vuruş aralıklarının medyanından, ölçü (2/4, 3/4, 4/4…) ölçü başları arasındaki vuruş sayısından gelir. Izgara sabit bir tempoya bağlı değil: metronomsuz çalınmış, hızlanıp yavaşlayan şarkılarda da ölçü çizgileri müziği izler.
3. **Akorlar:** [BTC](https://github.com/jayg996/BTC-ISMIR19) (Park ve ark., ISMIR 2019, MIT), Transformer tabanlı bir akor tanıyıcı. 170 sınıfı var: 12 kök × 14 tür (majör, minör, 7, maj7, m7, mMaj7, 6, m6, sus2, sus4, dim, dim7, m7b5, aug) ve "akor yok". `ml/models/btc.py` yalnızca çıkarım için yeniden yazılmış hali; ağırlıklar orijinal checkpoint'ten, SHA-256'sı doğrulandıktan sonra yükleniyor. Karışımın CQT'sinden kare kare akor olasılıkları çıkıyor. Bunların her vuruştaki ortalamasından Viterbi ile vuruş vuruş bir akor dizisi seçiliyor. Her akor değişimi bir ceza ödüyor, böylece tek vuruşluk sıçramalar eleniyor.
4. **Slash akorlar:** Karışımın bas bölgesinde (E1–A3) her vuruşun en alçak notası bulunur: harmonik toplamı en güçlü olanın yarısına ulaşan, perde ekseninde tepe olan en alçak nota. Bir akor iki vuruş ya da daha uzun süre üçlüsü, beşlisi ya da yedilisi üstünde ve bas her vuruşta net (vuruşun bas enerjisinin en az %80'i) çalınıyorsa slash akor olur: D/F#, G/B, C/E, Am/G, C/B.
5. **Melodi:** Ayrılan vokal, [ses modunun](#ses--gitar-söyle-tabını-al) perde takibiyle notalara çevrilip gitarın aralığına oturtulur.
6. **Ton ve ton değişimleri:** Melodinin notaları ve akorların sesleri, çaldıkları süreyle birlikte. Ton ölçü ölçü de izlenir: her ölçünün perde sınıfı profili iki yanındaki ikişer ölçüyle birlikte 24 ton profiliyle karşılaştırılır, Viterbi her ton değişimine bir ceza öder (`segment_keys`). Değişen şarkıların tonları bölüm bölüm saklanır.
7. **Gitar kanalı:** Şarkıda gitar varsa (Demucs gitar kanalı karışımın en az %10'u seviyesinde), gitar kanalı nota modeliyle, şarkının akort ve capo'suna göre çözülür. Akorları ritim kalıbının vuruşları olur ([Ritim kalıbı](#ritim-kalıbı)). Kimsenin şarkı söylemediği en az 4 saniyelik bölümlerde (intro, ara, solo, kapanış) notaları da TAB'a girer, vokal notalarından 0.25 s uzak kalarak.
8. **Capo:** 0–7 arası her capo için akorların çalınacak şekilleri bulunur. Çaldıkları süreyle ağırlıklı toplam zorluğu en düşük olan capo seçilir. Her capo perdesi küçük bir ceza öder, eşitlikte daha alçak capo kazanır. Açık akorlar kolay, barre şekilleri (E ve A telinden) perde yükseldikçe zorlaşır. Örneğin F majörde Cm–Bb–F çalan bir şarkı, capo 3 ile Am–G–D şekilleriyle çalınır. Capo elle de seçilebilir; **Otomatik (kolay akorlar)** bu öneriyi kullanır.

Standart akortta akor şekilleri bilinen açık ve barre şekilleridir. Slash akorlar ve diğer akortlar için şekil telleri üzerinde aranır: dört perdelik bir pencere, en alçak nota bas (slash akorda bas notası), tellerin arasında susturulan tel yok, dört parmağa sığmayan şekillerde barre; slash akorlarda en az dört tel. Örneğin D/F# `200232`, G/B `x20003`, C/E `032010`, Am/G `302010`.

**Editörde şarkı projeleri için:**

- **Akor tablosu:** Editörün üstünde her akorun şeması (capo varsa altında duyulan akor) ve ölçü ölçü akorlar. Çalan ölçü vurgulanır, bir ölçüye tıklamak oradan çalar. TAB'ın üstündeki akor isimleri de çalınacak şekillerdir. Ton değişen şarkılarda yeni ton, başladığı ölçünün üstünde yazar. Ölçüler editörün ızgarasından gelir, ölçü ya da ölçü başı düzeltilince tablo da değişir.
- **Akor düzeltme:** Bir akora tıklayıp kökünü, türünü ve basını (D/F# gibi) değiştir. Şema, TAB ve capo'lu şekil hemen güncellenir, **Kaydet** saklar.
- **Dinle: Notalar** melodiyle birlikte akorları çalar: ritim kalıbı bulunduysa onun 8'lik ya da 16'lıklarında, aşağı ve yukarı vuruşlarla; bulunmadıysa her vuruşta bir aşağı vuruş. Kayıtla üst üste dinleyince yanlış akor kulakla bulunur.
- **Akor şeması indir:** Düz metin. Başlıkta ton, capo, BPM, ölçü, varsa ritim (`D-DU-UDU`) ve ton değişimleri, ardından ölçü ölçü şekiller ve her şeklin dizilimi (`x02210`: kalın telden ince tele, `x` çalınmaz). MusicXML'de de melodinin üstüne şarkının akorları (basıyla), ton değiştiği ölçüde yeni armür yazılır.

**Ölçüm.** Akorlar mir_eval ile zaman oranı olarak (kök, majör/minör, 7'liler dahil), vuruş ve ölçü başları vuruş F-ölçüsüyle (70 ms) ölçüldü. BTC iki veriyi de görmedi. Beat This! ise GuitarSet'le eğitildi, bu yüzden oradaki vuruş skorları iyimser; AAM'deki vuruş skorları adil:

- **GuitarSet eşlik kayıtları** (180 kayıt, gerçek akustik gitar). Referans, çalınan değil nota kağıdındaki akorlar (7'lilerle).
- **[AAM](https://zenodo.org/records/5794629)** (Ostermann ve ark., 2023, CC BY 4.0): davul, bas, gitar, piyano ve melodi içeren, bilgisayarda üretilmiş grup şarkıları. Akor, vuruş ve ölçü etiketleri kesin, ama yalnızca majör/minör akor var. 70 şarkılık örnek `scripts/download_aam.py` ile uzak arşivden tek tek çekildi (şarkı başına ~15 MB). Etiketinde vuruş zamanları geriye giden 2 şarkı atlandı.

```bash
python scripts/download_aam.py --count 70
python scripts/benchmark_chords.py --dataset aam --penalty 0 1 2 4
python scripts/benchmark_chords.py --dataset guitarset --penalty 0 1 2 4
```

| Veri | Kayıt | Kök | Majör/minör | 7'liler dahil | Vuruş F | Ölçü başı F |
|---|---|---|---|---|---|---|
| GuitarSet eşlik | 180 | 0.813 | 0.777 | 0.651 | 0.960 | 0.936 |
| GuitarSet eşlik, oyuncu 05 | 30 | 0.659 | 0.635 | 0.522 | 0.989 | 0.956 |
| AAM | 68 | 0.945 | 0.928 | – | 0.941 | 0.939 |

- Gitar projelerinde akorları notalardan çıkaran yöntem oyuncu 05'te kök ve majör/minör için 0.68 / 0.63 veriyor, BTC 0.66 / 0.64. Gitar projeleri artık ikisini birlikte kullanıyor ([Akorlar: BTC + notalar](#akorlar-btc--notalar)).
- Akorları kare kare seçmek yerine vuruşlara oturtmak majör/minör skorunu AAM'de 0.912'den 0.928'e, GuitarSet'te 0.762'den 0.777'ye çıkardı. Dakikadaki akor değişimi de 32'den 27'ye ve 34'ten 24'e indi. Değişim cezası 0, 1, 2 ve 4 denendi, iki veride de 1 en iyisi.

**Slash akorlar.** Faydası için grup kaydı ve çevrim (bas) etiketi olan açık bir veri yok: AAM'de bas hep kökü çalıyor (zamanın %95.5'i; gerisi sus). Bu yüzden ölçülen iki şey var: AAM'de sahte slash akor ekleyip eklemediği, ve GuitarSet'te gitaristin *çalındığı* akorların etiketlerinde (zamanın %27'sinde bas kökten farklı, çoğu alternatif bas /5) ne kadarını yakaladığı. Skor majör/minör + bas (mir_eval `majmin_inv`):

```bash
python scripts/benchmark_chords.py --dataset aam --bass
python scripts/benchmark_chords.py --dataset guitarset --reference performed --bass
```

| Veri | Slash akorsuz | Slash akorlu |
|---|---|---|
| AAM (bas hep kökte) | 0.928 | 0.928 |
| GuitarSet çalınan akorlar (180) | 0.640 | 0.642 |
| GuitarSet çalınan akorlar, oyuncu 05 | 0.534 | 0.538 |

İlk denemede kural gevşekti (vuruşun bas enerjisinin yarısı yetiyordu, tek vuruş da sayılıyordu, ikili/dörtlü/altılı da kabul ediliyordu) ve iki veride de skoru düşürdü (AAM 0.928 → 0.862). Hataların çoğu bir yarım ton aşağıdaki notayı bas sanmaktı: alçak notaların enerjisi CQT'de alttaki bine taşıyor. En alçak notanın perde ekseninde tepe olması şartı ve kaydın akort sapmasının tahmini, AAM'de vuruş başına bas isabetini %72.5'ten %88.6'ya çıkardı (payı ≥%80 olan vuruşlarda %99). Tek gitarlı kayıtlarda bas hâlâ zor (GuitarSet'te vuruş başına %47.5), ama katı kuralla az ve çoğu doğru slash akor yazılıyor.

**Ton değişimleri.** AAM şarkıları bölümler arasında ton değiştiriyor (örnekte şarkı başına 3.4 kez, çoğu majör ile paralel minörü arasında), her bölümün tonu etiketli. GuitarSet eşlikleri tek tonlu, orada bulunan her değişim yanlış:

```bash
python scripts/benchmark_keys.py --dataset aam --penalty 1 2 4
python scripts/benchmark_keys.py --dataset guitarset --penalty 1 2 4
```

| Yöntem | AAM tam doğru / MIREX | AAM şarkı başına değişim | GuitarSet tam doğru | GuitarSet'te sahte değişim |
|---|---|---|---|---|
| Tek ton (önceki) | 0.403 / 0.540 | 0 | 0.817 | – |
| Bölümler, ceza 1 | 0.553 / 0.679 | 3.1 | 0.779 | kayıtların %15'i |
| **Bölümler, ceza 2** | **0.520 / 0.653** | 2.2 | **0.821** | %4 |
| Bölümler, ceza 4 | 0.503 / 0.634 | 1.1 | 0.828 | %0 |

Ceza 1 AAM'de en iyisi ama tek tonlu kayıtların %15'ine sahte modülasyon ekliyor; uygulama 2'yi kullanıyor. Ton, şarkının sesinden değil akorlarından bulunuyor (ölçümde melodi yok, uygulamada var).

**Süre:** 2 dakikalık bir şarkı bir dizüstü GPU'sunda (RTX 4060) yaklaşık 12 saniyede, 16 çekirdekli CPU'da yaklaşık 70 saniyede çözülüyor.

Sınırlar:

- Akor türleri 7'liler ve 6'larla sınırlı: 9, 11, 13 gibi genişletilmiş akorlar en yakın türle yazılıyor. BTC'nin sözlüğünde bunlar yok; GuitarSet'te çalınan akorların zamanının yalnızca ~%3'ü 9'lu ya da add9, bu yüzden kural tabanlı bir ekleme denenmedi.
- Slash akorların faydası ölçülemedi (yukarıda), yalnızca zarar vermediği ölçüldü. Alternatif bas (her vuruşta kök–beşli) bilerek slash akor yazılmıyor.
- Ton değişse de tek capo var: gitarist şarkı ortasında capo kaydırmaz. Akor isimleri editörde şarkının ana tonuna göre yazılıyor (Bb/A#), MusicXML'de bölümün tonuna göre.
- Ritim kalıbı yalnızca şarkıda gitar varsa çıkıyor (piyano eşlikli şarkılarda çıkmıyor), ve yönler duyulmuyor, sarkaç kuralından öneriliyor ([Ritim kalıbı](#ritim-kalıbı)). Şarkılarda ritim için referans veri yok; ölçüm gitar kayıtlarında.
- Melodi vokalden geliyor, vokalsiz bölümlerde gitar kanalından. Gitar kanalı grup kaydından ayrıldığı için tek başına kaydedilmiş gitardan daha az isabetli (bkz. [Gitarı ayırma](#gerçek-kayıtlar-ve-grup-kayıtları)); şarkı söylenirken çalınan gitar partisi TAB'a girmiyor. Ses modunun sınırları burada da geçerli: ayrılmış vokalde nota F1 (±100 ms) 0.57 civarında; armonili, dublajlı ya da ağır efektli vokaller zayıf. Vokalsiz ve gitarsız şarkılarda (örneğin piyano eşlikli enstrümantal) yalnızca akorlar çıkar.
- AAM bilgisayarda üretilmiş müzik. Gerçek kayıtlarda sonuçlar GuitarSet'e daha yakın beklenmeli.

## Ton, ölçü ve akorlar

Analiz notalardan ve kayıttan bulunanlardan yapılır (`packages/music-core/analysis.py`), notalar düzenlendikçe model yeniden çalışmadan tekrarlanır. Gitar ve şarkı projelerinde vuruşlar ve ölçü başları kaydın kendisinden gelir ([Vuruş takibi](#vuruş-takibi-metronomsuz-kayıtlar)); vuruş takibi olmayan projelerde (ses projeleri, model kurulu değilse) tek bir tempo kullanılır:

- **Tempo inceltme (tek tempoda):** Ses tabanlı tempo tahmini kaba bir çözünürlükte çalışır, ama %1'lik hata bile ızgarayı her 100 vuruşta bir vuruş kaydırır. Tahmin, notaların vuruş ve yarım vuruş periyodikliğini en iyi açıklayan tempoya (±%4 içinde) inceltilir.
- **Ton:** Nota sürelerinin perde sınıfı histogramı, Temperley–Kostka–Payne majör/minör profilleriyle karşılaştırılır. Armür ve nota/akor isimleri (F majörde Bb, A minörde G#) buna göre yazılır.
- **Ölçü ızgarası (tek tempoda):** Uzun ve bas notaların vuruşa düştüğü faz seçilir. Ölçü başı, armoninin en çok değiştiği vuruştur (akorlar genelde ölçü başında değişir).
- **Akorlar:** Her vuruşta çalan notaların kroması; majör, minör, 7, maj7, m7, power (5), sus2, sus4 ve dim şablonlarıyla eşleştirilir, Viterbi ile yumuşatılır, bas notası farklıysa slash akor (D/F#) yazılır. Gitar projelerinde buna kayıttan tanınan akorlar da katılır ([Akorlar: BTC + notalar](#akorlar-btc--notalar)). Tek sesli melodilere akor yazılmaz.

GuitarSet oyuncu 05'in model transkripsiyonları üzerinde (uygulamanın gördüğü girdi; V9 + V7):

| Ölçüm | Eşlik (comp) | Solo |
|---|---|---|
| Ton (MIREX ağırlıklı skor) | 0.82 | 0.76 |
| Tempo %0.3 içinde (inceltme öncesi → sonrası, tahmin %4 içindeyken) | %14 → %94 | |
| Vuruş / ölçü başı F: tek tempo → vuruş takibi | 0.634 / 0.531 → **0.984 / 0.950** | 0.312 / 0.078 → 0.393 / 0.291 |
| Akor kökü / majör-minör / 7'liler (zaman oranı): notalar → BTC + notalar | 0.678 / 0.627 / 0.444 → **0.682 / 0.656 / 0.525** | akor yazılmıyor |

Ölçü başı ve tempo editörde elle düzeltilebilir: bir notayı seçip **Seçili nota 1. vuruş**'a basmak ölçü çizgilerini o notanın vuruşuna hizalar. **Ölçü** (2/4–7/4) ve **Ton** seçimleri de aynı şekilde kaydedilir; "Otomatik"e dönünce tahmin kullanılır. **BPM** alanına bir tempo yazmak ızgarayı o sabit tempoya geçirir.

### Vuruş takibi (metronomsuz kayıtlar)

Tek bir tempo, metronomsuz çalınmış ve hızlanıp yavaşlayan kayıtlarda zamanla kayar. Gitar projelerinde de artık Beat This! vuruşları ve ölçü başlarını kaydın kendisinden buluyor (şarkı modundaki gibi). Izgara bu vuruşları izliyor: editördeki vuruş ve ölçü çizgileri, metronom, **Ritmi oturt** (her vuruşun kendi 8'lik/16'lıkları), MusicXML (tempo değiştiği ölçüye çalma temposu, %8'den büyük değişimde görünür metronom işareti) ve MIDI (vuruş başına tempo haritası; notalar saniye olarak kayıtla hizalı kalıyor, ilk ölçü çizgisinden önceki kısım anakruz ölçüsü). Ölçüler, takip edilen ölçü başlarının çoğunun düştüğü vuruştan başlayarak düzenli sayılır; tracker'ın arada kaçırdığı bir ölçü başı ızgarayı bozmuyor. Ölçü sayısı (2/4, 3/4…) ölçü başları arasındaki vuruş sayısından geliyor. Editördeki **Izgara** seçimi bunu kapatıp tek tempoya döndürüyor (Beat This! bazen yanlış metrik seviyeyi seçiyor; örneğin bir GuitarSet kaydında 98 yerine 143 BPM).

Ölçüm (`scripts/benchmark_grid.py`): GuitarSet oyuncu 05, vuruş ve ölçü başı F-ölçüsü (70 ms). GuitarSet metronomla kaydedildiği için ikinci koşulda kayıtlar her 4 ölçüde rastgele bir oranla (0.88–1.12, sürüklenerek) esnetildi ve referans vuruşlar da onunla birlikte taşındı:

```bash
python scripts/benchmark_grid.py --players 05
```

| Koşul | Kayıt | Vuruş F: tek tempo → takip | Ölçü başı F: tek tempo → takip |
|---|---|---|---|
| Eşlik, metronomlu | 30 | 0.634 → **0.984** | 0.531 → **0.950** |
| Eşlik, tempo değişken | 30 | 0.496 → **0.914** | 0.323 → **0.878** |
| Solo, metronomlu | 30 | 0.312 → 0.393 | 0.078 → 0.291 |
| Solo, tempo değişken | 30 | 0.299 → 0.311 | 0.066 → 0.220 |

Beat This! GuitarSet'le de eğitildi, bu yüzden metronomlu skorlar iyimser; esnetilmiş kayıtlar onun için de yeni. Sololar ikisi için de zor. F-ölçüsü yarım ya da iki kat tempo hatalarını sıfır sayıyor; tek tempo yönteminin düşük vuruş F'sinin bir kısmı bundan.

### Akorlar: BTC + notalar

Gitar projelerinde transkripsiyon sırasında BTC'nin her vuruş için en olası 6 akoru ve log-olasılıkları saklanıyor. Analiz bunları notaların akor şablonlarına uyumuyla birleştiriyor: her vuruşta BTC log-olasılığı + 4 × (notaların kroması ile akorun tonları arasındaki kosinüs benzerliği), ardından Viterbi (değişim cezası 1). Notalar düzenlendikçe akorlar da değişiyor, ama kaydın kendisi de oy veriyor. Ağırlık BTC'nin görmediği GuitarSet oyuncuları 00–03'te seçildi (1, 2, 4, 8 denendi; 4 kök ve majör/minörde en iyiydi), sonuç oyuncu 05'te. Tek sesli kayıtlara (sololar) yine akor yazılmıyor.

| Yöntem (oyuncu 05 eşlikleri, nota kağıdı akorları) | Kök | Majör/minör | 7'liler dahil |
|---|---|---|---|
| Notalar, tek tempo ızgarası (önceki) | 0.678 | 0.627 | 0.444 |
| Notalar, takip edilen vuruşlar | 0.688 | 0.641 | 0.449 |
| Yalnız BTC | 0.659 | 0.635 | 0.522 |
| **BTC + notalar** | **0.682** | **0.656** | **0.525** |

Kayıttan tanınan akorlar BTC'nin sözlüğüyle geliyor (6, m6, dim7, m7b5, aug, m(maj7) dahil). Vuruş takibi olmadan (model kurulu değilse) akorlar eskisi gibi yalnız notalardan çıkıyor.

### Ritim kalıbı

Akorların hangi 8'lik ya da 16'lıklarda vurulduğu (`packages/music-core/rhythm.py`). Vuruşlar transkripsiyondan geliyor: 60 ms içinde başlayan en az iki nota bir vuruş (bir akor yeniden vurulduğunda model çoğu zaman tellerin yalnızca bir kısmını yeniden başlatıyor, bu yüzden üç nota şartı vuruşların bir kısmını kaçırıyordu). Vuruşlar ızgaranın 16'lıklarına yerleştiriliyor; vuruş gücünün %10'undan azı 8'liklerin arasındaysa kalıp 8'lik yazılıyor. Her ölçünün vurulan hücreleri bulunuyor, şarkının kalıbı da en az iki vuruşlu ölçülerin en az yarısında vurulan hücreler (tek bir fazla ya da eksik vuruş kalıbı değiştirmiyor). Editör kalıbı TAB'ın üstünde "1 & 2 & …" sayımının altında oklarla gösteriyor, **Dinle: Notalar** şarkı akorlarını ölçü ölçü bu vuruşlarla çalıyor, akor şeması `D-DU-UDU` diye yazıyor. Tek sesli kayıtlarda (sololar) kalıp çıkmıyor.

Yönler duyulmuyor, öneriliyor: el her hücrede aşağı ve yukarı gidip gelir (sarkaç kuralı), çift hücreler aşağı, tek hücreler yukarı. GuitarSet'in her teli ayrı etiketlendiği için vuruşların gerçek yönü de biliniyor (tellerin çalma sırası: kalından inceye aşağı):

```bash
python scripts/benchmark_strums.py
python scripts/benchmark_strums.py --players 05 --notes-cache ml/data/interim/notes05
```

| Ölçüm (GuitarSet, en az 20 vuruşlu eşlikler) | Sonuç |
|---|---|
| Sarkaç kuralı doğru yön (154 kayıt, 8127 vuruş) | %78 (funk %93, rock %80, singer-songwriter %74, caz %69, bossa nova %67) |
| Vuruş bulma F (50 ms), oyuncu 05: transkripsiyondan / doğrudan sesten | **0.903** / 0.707 |
| Vurulan 16'lıklar F, oyuncu 05: transkripsiyondan / sesten | **0.891** / 0.706 |
| Kalıp: hücre F / tam doğru, oyuncu 05: transkripsiyondan / sesten | **0.889 / %60** / 0.664 / %12 |

Sesten doğrudan onset almak (her tek nota, her ölü vuruş da onset) kalıp için fazla gürültülü çıktı; bu yüzden şarkı modunda da gitar kanalı nota modeliyle çözülüyor. Rock'ta çoğu oyuncu her 8'liği aşağı vuruyor; kural orada yukarı önerir (kalıp doğru, yönler değil). Bossa nova parmakla çalınıyor, yön kavramı zayıf.

## Çalma teknikleri: bend, vibrato, slide

Nota modeli hangi perdenin ne zaman çaldığını yarım ton çözünürlüğüyle buluyor. Notanın nasıl çalındığı ise perdenin zaman içindeki tam seyrinde görünüyor (`ml/inference/techniques.py`). Model eğitilmedi; kurallar her notanın kayıttan çıkarılan perde eğrisine uygulanıyor. Eşikler [IDMT-SMT-Guitar](https://zenodo.org/records/7544110)'ın (Kehling ve ark., 2014; CC BY-NC-ND) çalgıcının kendi etiketlediği bend, slide ve vibrato notalarıyla seçildi:

- **Perde eğrisi:** notanın çevresindeki aday perdeler için (10 cent adımla, 2.5 yarım ton aşağıdan 3.5 yarım ton yukarıya) spektrumun ilk 6 harmoniğindeki enerji toplanıyor. Aynı anda çalan diğer notaların perdeleri maskeleniyor. Kare kare en güçlü adayı seçmek yerine Viterbi ile süreklilik aranıyor: perde değişimi yarım ton başına sabit bir bedel ödüyor, böylece nota sönerken başka bir tele atlamalar eleniyor. Bu bedel bilerek doğrusal: karesel bedel, atlamaları küçük adımlara bölüp bend'e benzeyen rampalar üretti. Kaydın akort sapması (gitarlar çoğu zaman A = 440 Hz'in biraz yanındadır; IDMT'nin bir çekiminde tellerin çoğu 35–45 cent tiz) notaların eğrilerinden ölçülüp düzeltiliyor; bütün eşikler gitarın kendi akortuna göre.
- **Bend:** perde, basılan notanın en az 0.75 yarım ton üstünde en az 60 ms tutuluyor, oraya en az 30 ms'de çıkıyor (yeni basılan bir nota sıçrar) ve yolda iki kare arasında bend'in dörtte birinden büyük bir adım atmıyor. Son şart slide'ı ayırıyor: slide perde perde tırmanır (IDMT'de en büyük adım yolun ortanca %31'i, bend'de %16); bu şart olmadan IDMT'nin slide'larının %47'si bend sanılıyordu. Basılan perde notanın başının en alçak kısmından okunuyor, çünkü çalgıcı bükmeye pena ile birlikte başlayabiliyor (IDMT'nin lick'lerinde bend'li notaların ilk 40 ms'si ortanca 0.15–0.5 yarım ton yukarıdaydı). Hızlı bir bend'i nota modeli bazen bükülmüş perdeden yazıyor (IDMT'de bend'lerin beşte biri); bu notalar basılan perdeye geri çekiliyor. Miktar yarım ton yarımlarına yuvarlanıyor; TAB'da `7b9` (7. perde, 9. perdenin sesine bükülmüş), kesirli bend'lerde `7b`.
- **Vibrato:** yavaş seyri çıkarılmış eğride 4–9 Hz'lik düzenli bir salınım (o periyotta öz-ilinti ≥ 0.2), tepe-tepe en az 0.15 yarım ton (IDMT'deki vibratoların ortanca derinliği 0.22), notanın en az 0.4 saniyesi duyulmuş. 4 Hz'den yavaş vibrato (IDMT'nin yavaş çekimleri, 2–3 Hz) aranmıyor: yakaladığından çok yanlış işaretledi. TAB'da `7~`.
- **Slide:** art arda iki nota, ikincisi birincinin bittiği yerde başlıyor (en çok 12 ms boşluk: ses hiç kesilmediği için nota modeli ikinci notayı birincinin bittiği karede başlatıyor; IDMT'de slide sanılan pena geçişlerinde boşluk ortanca 46 ms'ydi); arada perde 20–80 ms iki notanın arasında kalıyor ve ikinciye doğru ilerliyor. Slide tek telde çalınır, ama parmak yolu optimizer'ı bunu bilmeden iki notayı sık sık iki ayrı tele koyuyor (IDMT'de bulunan slide'ların %40'ında); slide bulununca ikisi bir tele alınıyor (ikinci not birincinin teline, olmazsa tersi). Bir perdelik slide legatodan ayrılamıyor, aranmıyor. TAB'da `5/7`, `7\5`.
- **Hammer-on / pull-off tespit edilmiyor.** Bunların bıraktığı iz (pena vuruşundan daha zayıf bir atak) GuitarSet'te pena ile çalınmış notalardan ayrılmadı: aynı perdenin yeniden vurulduğu geçişlerle perde değişen geçişlerin yüksek frekanslı atak dağılımı neredeyse aynıydı (ortanca oran 0.71'e 0.63). IDMT'de bu teknikler yok; etiketli bir kayıt olmadan doğrulanamadıkları için yalnızca editörden elle işaretlenebiliyorlar (TAB'da `5h7`, `7p5`).

Ölçüm (`scripts/benchmark_techniques.py`): IDMT'nin 2. alt kümesi, tek çalgıcı, üç elektro gitar (AR, FS, LP), 261 çekim; kromatik dizilerin yanında tek tek bend, slide ve vibrato çekimleri ve bunları karıştıran lick'ler. Eşikler AR ve FS ile seçildi; aşağıdaki sayılar hiç bakılmadan bırakılan LP gitarında. "IDMT notalarıyla" yalnızca perde eğrisi ve kuralları ölçüyor; "modelin notalarıyla" uygulamanın gerçekte yaptığını (V10 + V7 transkripsiyonu, ardından teknikler).

```bash
python scripts/benchmark_techniques.py --dataset idmt --guitars LP --notes reference
python scripts/benchmark_techniques.py --dataset idmt --guitars LP --notes model
```

Veri: Zenodo'daki `IDMT-SMT-GUITAR_V2.zip` (1.3 GB) `ml/data/raw/idmt` altına açılır; yalnızca 1–3. alt kümeler gerekiyor (akor ve ritim için olan 4. alt küme kullanılmıyor). Yükleyici `ml/datasets/idmt.py`.

| IDMT, LP gitarı | Bend P / R | Vibrato P / R | Slide P / R |
|---|---|---|---|
| IDMT'nin notalarıyla | 0.87 / 0.56 | 0.87 / 0.48 | (notalar ayrı değil) |
| **Modelin notalarıyla** | **0.78 / 0.25** | **0.82 / 0.33** | **0.83 / 0.14** |

Slide'da ilk kurallar (aynı tel, 120 ms'ye kadar boşluk) AR+FS'de 0.16 kesinlik veriyordu; boşluk ve süre şartlarıyla 0.85. Normal notaların hemen hiçbiri işaretlenmiyor (bend %0, vibrato %0.1). İşaretlenenler çoğunlukla doğru, ama bend, vibrato ve slide'ların çoğu kaçıyor. Kaçırmanın büyük kısmı nota modelinden geliyor: IDMT'nin bend'lerinin dörtte birinin yakınında model hiç nota yazmıyor, AR+FS'de slide'ların üçte ikisinde model, kuralın bakabileceği iki ayrı nota üretmiyor. IDMT tek çalgıcı ve tek kayıt düzeni; LP testi ancak gitar farkını sınıyor.

GuitarSet'te teknik etiketi yok, ama her tel heksafonik pickup ile ayrı ayrı izlenmiş. Aynı kurallar bu referans eğriye uygulanınca etiket, mikrofon kaydından çıkarılan eğriye uygulanınca tahmin oluyor; yani ölçülen, karışık bir mikrofon kaydında eğrinin ne kadar iyi izlendiği (`--dataset guitarset`):

| GuitarSet sololar, mikrofon kaydı | Sonuç |
|---|---|
| Perde eğrisi, oyuncu 05: ortanca hata / 25 cent içinde | 3.5 cent / %92 |
| Bend, oyuncu 00–04 | işaretlenenlerin %79'u pickup eğrisinde de bend* |
| Vibrato, oyuncu 00–04 | P 0.69, R 0.65 |
| Süre | dakikalık kayıt başına ~0.3 s (CPU) |

\* Bend'de yalnızca kesinlik anlamlı: notanın başının en alçak kısmını basılan perde sayan kural, pickup eğrilerindeki atak çukurlarını da bend'e çeviriyor, bu yüzden pickup'tan çıkan referans bend'ler şişiyor (oyuncu 00–04'te 190). GuitarSet'te gerçek bend az. Teknikler şarkı modunda ayrılmış gitar kanalında işaretlenmiyor; eğri yalnızca tek başına kaydedilmiş gitarda ölçüldü.

Teknikler TAB'da (editör ve ASCII) ve MusicXML'de (her iki dizekte `hammer-on`/`pull-off`, `slide`, `bend`, vibrato için dalgalı çizgi; çıktı MusicXML 4.0 şemasına göre geçerli) yazılıyor, MIDI'de ve editördeki **Dinle: Notalar** sentezinde duyuluyor:

- **MIDI:** pitch bend tüm kanalı büktüğü için teknikli dosyalarda her tel kendi kanalında (Guitar Pro'nun da yazdığı "MIDI gitar" düzeni), bend aralığı bir oktav (RPN 0). Bend pena vuruşundan 40 ms sonra başlıyor ve 100 ms'de bükülmüş perdeye çıkıyor, vibrato 5.5 Hz'de ±20 cent, slide sonraki notanın başladığı yerde biten 60 ms'lik bir kayma (ilk nota o ana kadar sürüyor). Hammer-on ve pull-off %30 daha yumuşak çalınıyor. Tekniksiz dosyalar eskisi gibi tek kanallı.
- **Sentez:** hammer-on, pull-off ve slide yeni bir pena sesi başlatmıyor; aynı telde çalan ses perdesini değiştiriyor (sıçrayarak ya da kayarak). Bend ve vibrato da aynı biçimlerle sesin perdesinde. Bu biçimler Python ile TypeScript'te ortak fixture'la eşit tutuluyor.

Editörün nota panelinden her notanın tekniği, bend miktarı ve vibratosu değiştirilebiliyor.

### Perde eğrisi modeli (Colab'da eğitim)

Kuralların kaçırdıklarının çoğu nota modelinden geliyor: bend'li ve slide'lı geçişleri ayrı notalara bölmüyor. Sıradaki adım, perdeyi yarım ton yerine yarım tonun üçte biri (33 cent) çözünürlükle izleyen bir **kontur kafası** (Basic Pitch'teki gibi). Bend, vibrato ve slide bu çıkışta hareket eden bir aktivasyon olarak görünür; teknikler elle yazılmış harmonik toplamı izleyicisi yerine buradan okunabilir.

- **Hedef:** GuitarSet'in heksafonik pickup'tan çıkan `pitch_contour` anotasyonları (her tel, 5.8 ms'de bir). Her nokta kendi kesirli perdesine 1 bin genişliğinde bir Gauss olarak işleniyor, 157 bin (C2–E6). Teller aynı karede birleşiyor. Perde kaydırma ve tempo değişimi augmentation'larında eğri de kayıyor (`variant_contours`).
- **Model** (`ml/configs/guitar_contour.yaml`): aynı CNN + BiLSTM, girişi 36 bin/oktav CQT ve iki kat kare hızı (hop 512, ~86 kare/s; 9 Hz'lik vibratoyu izlemeye yetiyor). Son conv bloğu frekansı küçültmüyor (`freq_pools: [2, 2, 1]`). Kontur kafasının yanında nota kafaları da eğitiliyor. Kontur yalnızca GuitarSet'te var, model yalnızca onunla eğitiliyor; doğrulamada izlenen skor `val_contour_f1`.
- **Veri:** 360 kayıt ve eğitim oyuncularının 240 kaydı için 7 varyant (`ps+1`, `ps-2`, `tempo0.9`, `noise`, `overdrive`, `room`, `codec`).

Veri bu bilgisayarda hazırlanıyor, eğitim ücretsiz Colab'daki T4'te ([`notebooks/train_contour_colab.ipynb`](notebooks/train_contour_colab.ipynb)):

```bash
python scripts/prepare_dataset.py --config ml/configs/guitar_contour.yaml --augment \
    --variants ps+1 ps-2 tempo0.9 noise overdrive room codec
python scripts/pack_dataset.py --config ml/configs/guitar_contour.yaml   # -> guitarset_contour.zip
```

`guitarset_contour.zip` Drive'da `MyDrive/music_transcriber/` altına yükleniyor, notebook baştan sona çalıştırılıyor. Checkpoint'ler her epoch Drive'a yazıldığı için oturum koparsa notebook yeniden çalıştırılınca eğitim `last.pt`'den devam ediyor. En iyi skor (`best.pt`), erken durdurma sayacı ve `history.csv` de devam eden çalıştırmada korunuyor.

**İlk sonuç: kurallara henüz yetmiyor, uygulama harmonik toplamda kalıyor.** Model 80 epoch'luk eğitimin 76.'sında en iyi skora ulaştı (doğrulama oyuncusu 04: `val_contour_f1` 0.779, nota karesi F1 0.887). `ContourTracker` (`ml/inference/techniques.py`) modelin çıkışından notanın eğrisini harmonik toplamla aynı yolla okuyor (aday perdeler, diğer notaların maskelenmesi, sürekli yol; binler arası için log-olasılıkta parabol). `scripts/benchmark_techniques.py --tracker contour` ikisini karşılaştırıyor:

| | Harmonik toplam | Kontur modeli |
|---|---|---|
| GuitarSet oyuncu 05 sololar, pickup'a göre ortanca hata / 25 cent içinde | 3.5 cent / %91.5 | 3.9 cent / %91.8 |
| Vibrato derinliği (tepe-tepe, pickup'ta vibratolu notalar, pickup 0.34) | 0.28 yarım ton | 0.08 yarım ton |
| IDMT LP, IDMT notalarıyla: bend F / vibrato F | 0.68 / 0.62 | 0.00 / 0.14 |
| IDMT AR+FS, kurallar gevşetilince: bend F / vibrato F | 0.75 / 0.65 | 0.30 / 0.49 |

Ortanca hata iyi görünüyor, çünkü karelerin çoğunda perde sabit. Ama eğri binler arasında akmıyor, bin bin basamak atlıyor: bir bin 33 cent, tipik bir vibrato ±17 cent, yani salınım çıkışı bir bin bile oynatmıyor. Bend'de ise basamaklar "slide gibi tırmanıyor" şartına takılıyor. Bundan da çok, notanın başı (pena atağı) boş ya da yukarıda çıkıyor: IDMT LP'nin 57 bend'inin 28'i "başlangıç bir perdeye oturmuyor" diye eleniyor. Temiz Gauss bir çıkışta `ContourTracker` binler arasındaki bend'i birkaç cent içinde izliyor (`tests/test_contour.py`); sorun okumada değil, modelin çıkışının biçiminde. Kafa, frekansı 512 boyutlu tek bir vektöre sıkıştıran katmandan sonra geliyor; Basic Pitch'te olduğu gibi frekans ekseni boyunca konvolüsyonla çalışan bir kontur kafası ve daha ince binler (yarım ton başına 5, 20 cent) sıradaki eğitim turunun adayları.

**2. tur** (`ml/configs/guitar_contour5.yaml`, `ml/models/contour.py`): kontur kafası artık gömme vektörünü değil spektrogramın kendisini okuyor ve frekans eksenini sonuna kadar koruyor. CQT oktav başına 60 bin (yarım ton başına 5, çıkışın çözünürlüğü), harmonik yığını (Basic Pitch gibi: spektrum 0.5, 1, 2, 3, 4, 5. harmoniğin aralığı kadar kaydırılıp kanal olarak üst üste konuyor, böylece bir temel frekansın bininde harmonikleri hizalanıyor), iki 5x5 konvolüsyon, frekansta 13 yarım tonluk geniş bir konvolüsyon ve 5x5 bir çıkış. Her çıkış bini kendi perdesinde hesaplanıyor, binler arasında kayan bir perde aktivasyonu da kaydırıyor. 261 bin (C2–E6). Nota kafaları CRNN yolunda kalıyor. Kafa 35 bin parametre; 16'lık batch 4060'ta 0.95 s, 5.6 GB.

Bu kafa perdeyi doğru bine koyuyor ama emin olmadan: küçük bir denemede karelerin %80'inde en güçlü bin hedefteydi, olasılıklar ise 0.5'in altında kaldı. 0.5 eşikli doğrulama skoru bu yüzden sıfırdı; `best.pt` ilk epoch'ta kalır, eğitim erken dururdu. `val_contour_f1` artık 0.1–0.5 eşiklerinden en iyisinde ölçülüyor (`val_contour_threshold`). Eğri izleyici en güçlü yolu izlediği için bu güvensizlik eğriyi etkilemiyor.

Aynı küçük denemede (GuitarSet'in 8 kaydı, 40 kısa epoch) oyuncu 05'in sololarında:

| GuitarSet oyuncu 05 sololar | 1. tur (tam eğitim) | 2. tur kafası (küçük deneme) | 1. tur kafası (küçük deneme) |
|---|---|---|---|
| Vibrato derinliği (pickup 0.34) | 0.08 | 0.16 | 0.03 |
| Pickup'ta vibratolu 42 notadan işaretlenen | 7 | 21 | 1 |
| 25 cent içinde | %91.8 | %92.6 | %80.4 |

Neredeyse hiç eğitilmeden bile vibratoyu tam eğitilmiş 1. turdan iyi izliyor. Veri 1. turla aynı kayıtlar ve varyantlar; zip yaklaşık 4 GB (`guitarset_contour5.zip`). Notebook'ta `ROUND = 2` varsayılan.

**2. turun sonucu: 1. turdan çok iyi, harmonik toplamı hâlâ geçmiyor; uygulama harmonik toplamda kalıyor.** Colab'da en iyi skor 66. epoch'ta (`ml/checkpoints/guitar_contour5/best.pt`: `val_contour_f1` 0.740, artık 0.5 eşiğinde; nota karesi F1 0.886). Ölçerken iki şey değişti:

- Model saniyede 86 kare veriyor (hop 512), kurallar ise harmonik toplamın 172 karesinde seçilmişti. Aynı bend yarı kare hızında kareler arasında iki kat büyük adım atıyor ve "slide gibi tırmanıyor" şartına takılıyor, medyan filtre de iki kat uzun bir süreyi düzlüyor. `ContourTracker.from_audio` modelin çıkışını artık zamanda `hop_length` hızına yeniden örnekliyor (AR+FS bend F 0.44 → 0.53).
- Kalan bend kayıplarının çoğu (AR+FS'de 41) notanın başının daha çalınırken bükülmüş olmasından geliyor. Harmonik toplamın 46 ms'lik penceresi pena atağındaki basılı perdeyi bir süre tutuyor, kontur modeli tutmuyor: başlangıç notanın bir yarım ton üstüne yuvarlanıyor ve "başlangıç bir perdeye oturmuyor" diye eleniyor. Yeni `bend_above` ayarı başlangıcın notanın bu kadar yarım ton üstünde olmasına izin veriyor, basılı perde o zaman notanın kendisi sayılıyor. (GuitarSet pickup'ına göre iki izleyicinin eğrisi arasında zaman kayması yok, ikisi de en iyi +2–3 karede.)

| | Harmonik toplam | Kontur, 1. tur | Kontur, 2. tur |
|---|---|---|---|
| GuitarSet oyuncu 05 sololar: ortanca hata / 25 cent içinde | 3.5 cent / %91.5 | 3.9 cent / %91.8 | 6.4 cent / %94.0 |
| GuitarSet oyuncu 05: vibrato F (P / R) | 0.57 (0.46 / 0.74) | – | 0.46 (0.35 / 0.69) |
| IDMT AR+FS, IDMT notalarıyla: bend F, `bend_above` 0 → 1 | 0.75 → 0.78 | – | 0.53 → 0.75 |
| IDMT AR+FS: vibrato F | 0.65 | – | 0.56 |
| IDMT LP, IDMT notalarıyla: bend F, `bend_above` 0 → 1 / vibrato F | 0.68 → 0.70 / 0.62 | 0.00 / 0.14 | 0.42 → 0.56 / 0.52 |
| IDMT LP, modelin notalarıyla: bend F (`bend_above` 1) / vibrato F / slide F | 0.33 / 0.47 / 0.23 | – | 0.35 / 0.45 / 0.05 |

Bend'de, başlangıç kuralı gevşetilince kontur modeli harmonik toplama yaklaşıyor (modelin notalarıyla LP'de biraz önde, ama 17'ye 16 işaret). Vibratoda geride: normal notaların daha çoğunu titreşiyor sayıyor (LP'de %0.8, harmonik toplamda %0.1), eğri sabit bir notada da biraz oynuyor. Slide'da çok geride. `bend_above` varsayılan olarak 1: harmonik toplamda da AR+FS'de ve LP'de bend'i biraz artırıyor, modelin notalarıyla (uygulamanın yaptığı) bir şey değiştirmiyor.

### Modelin notalarında neler kayboluyor

IDMT LP'de, modelin notalarıyla etiketli her tekniğin nereye gittiği (harmonik toplam, `bend_above` 1, aşağıdaki `bend_extend` öncesi):

| IDMT LP | Bulunan | Nota var, kural "hayır" | Model nota yazmamış | Model başka perdeden yazmış |
|---|---|---|---|---|
| Bend (57) | 12 | 17 | 21 | 6 (bükülmüş perdeden, geri perdelenmemiş) |
| Slide (37) | 5 | 4 | 9 (ilk nota) | 7 (ilk nota) |
| Vibrato (54) | 18 | 32 | – | 4 (onset 50 ms'den uzak) |

Kalan 11 slide'ın 8'i tek perdelik kayma: tasarım gereği ölçülmüyor. Diğer 3'ünde iniş notası yok ya da iki nota arasındaki boşluk kuralın dışında.

- **Vibrato'da kayıp nota modelinde değil, kuralda:** notaların %95'i eşleşiyor. Kural, IDMT'nin kendi nota sınırlarıyla bile vibratoların yarısını kaçırıyor. Modelin notaları %20–25 daha kısa; bu da LP'de 6 vibrato daha kaybettiriyor.
- **Notası hiç yazılmayan bend'lerin hepsi Lick11'den.** Muted çekimlerde (`MN`) bend'ler 0.1–0.15 s süren notalar ve modelin frame olasılığı neredeyse sıfır. Diğer çekimlerde bend legato çalınıyor (önceki notadan pena vurulmadan). Frame olasılığı komşu perdelere dağılıyor (0.2–0.5), onset ise ≈ 0. Decoder onset istediği için nota yazmıyor.
- **Denenen ve bırakılan: frame olasılığından nota kurtarma.** Notası olmayan yerlerde frame olasılığı ±1 yarım ton toplandı, çalan notalardan ve onların üst harmoniklerinden uzak bölgeler aday nota sayıldı. Bükülünce kesilen notaların uzatılmış hâli de aday oldu.
  - Adayların hepsi eklenince AR+FS'de nota F1 0.792'den 0.787'ye ve altına düştü. Eklenenlerin çoğu, var olan bir notanın içinde yeniden başlayan kopyalardı.
  - Yalnızca kuralların bend ya da slide bulduğu adaylar tutulunca bend F AR+FS'de 0.51'den 0.54'e, LP'de 0.33'ten 0.35'e çıktı. Ama LP'de eklenen 6 bend'in 4'ü yanlıştı, GuitarSet oyuncu 05'te (pickup'a göre) eklenen 7 bend'in hepsi yanlıştı (P 0.40 → 0.17).
  - Uygulamaya alınmadı.
- **Kesilen notayı ileri okumak (`bend_extend`, uygulamada).** Model, perde basılan notanın bininden ayrılınca notayı bitiriyor; çoğu zaman bend tutulmadan önce. "Nota var, kural hayır diyor" grubunun ve bükülmüş perdeden yazılan notaların çoğunda kural, eğri çok kısa olduğu için ya da bend'in tepesini görmediği için bend bulamıyordu.
  - Bend'i olmayan bir nota, en çok 0.2 s ileriye kadar yeniden okunuyor. Perdesine yakın bir sonraki notayı geçmiyor, akorlarda denenmiyor. Bend notanın içinde başlamışsa nota bend sayılıyor ve bitişi uzatılıyor.
  - "Bend nota bittikten sonra başlıyorsa sayma" şartı olmadan GuitarSet'in akor kayıtlarında yanlış bend'ler çıkıyordu.
  - 0.1, 0.2 ve 0.3 s, AR+FS'de birbirinden en çok bir nota farklı.

| Modelin notalarıyla, bend | Önce | `bend_extend` 0.2 |
|---|---|---|
| IDMT AR+FS (ayar): P / R / F | 0.89 / 0.36 / 0.51 | 0.91 / 0.46 / **0.62** |
| IDMT LP (test): P / R / F | 0.75 / 0.21 / 0.33 | 0.78 / 0.25 / **0.37** |
| GuitarSet oyuncu 05, solo + akor: işaretlenen / pickup'ta da bend | 5 / 2 | 5 / 2 |
| GuitarSet oyuncu 00–04, sololar: işaretlenen / pickup'ta da bend | 17 / 3 | 28 / 4 |

Vibrato ve slide değişmiyor, IDMT'nin normal notalarında yeni bend yok. GuitarSet sololarında eklenen 11 bend'in 10'u, aynı pencerede pickup eğrisinde bend değil. Bunlar çoğunlukla notanın sonunda parmak kalkarken perdenin kısa bir sıçraması ve ardından gelen çınlama (13 858 eşleşen notada %0.08). Ses gücü, bend miktarı ve zamanlama bunları gerçek bend'lerden ayırmadı.

`benchmark_techniques.py --notes model` artık notaları değil modelin olasılıklarını önbelleğe alıyor (`ml/data/interim/techniques/*.npz`). Böylece decoder'daki bir değişiklik modeli yeniden çalıştırmadan ölçülebiliyor; decoder ayarları `--inference onset_threshold=0.4` gibi değiştirilebiliyor.

## Farklı akortlar ve capo

Yüklerken (ya da editörde yeniden çözümlerken) akort ve capo seçilir. Akortlar: Standart, yarım ton aşağı, bir ton aşağı, Drop D, Drop C, C standart, DADGAD, Open G ve Open D. Capo 0–12 arasında seçilebilir. Perdeler capodan itibaren sayılır. TAB, MusicXML ve ASCII TAB "Capo N" yazar, tel adları akortun kendisini gösterir. Akort ya da capo değişince notalar aynı kalır ama tel ve perdeler değişir. Bu yüzden editör, ayar değişince yeniden çözümlemeyi önerir.

**Tab kafası başka akortta.** Tab modeli yalnızca standart akortla eğitildi: bir teli, o telde duyduğu perdelerden tanıyor. Bu yüzden onun sınıflarını doğrudan yeni akortun perdeleri gibi okumak, notaları yanlış tele koyuyor. Bunun yerine her aday pozisyon için tab kafasına, bu perdenin *standart akortta* o telde hangi sınıfa düştüğü soruluyor. Standart akortta o telde çalınamayan notalarda (Drop D'de kalın teldeki D2 gibi) telin etkinliği kullanılıyor. GuitarSet testi kaydırılarak ölçüldüğünde, kayıt başına ortalama tel/perde isabeti bir ton aşağıda %65'ten %93'e, iki ton aşağıda %35'ten %80'e çıktı.

**C2'ye inen nota modeli (V9).** V8'in aralığı E2'de (MIDI 40) başlıyor. Bu yüzden C standart ve Drop C'nin en kalın notalarını hiç yazamıyor. V9, `guitar_mixed_low.yaml` ile V8'den 20 epoch ince ayarlandı ve aralığı C2'ye (36) iniyor. Alçak notalar, GuitarSet'in 2 ve 4 yarım ton aşağı kaydırılmış kopyalarından (`ps-2`, `ps-4`) geliyor. Kayıtlı E2–E6 hedefleri yeniden hazırlanmıyor, eksik alt satırlar eğitim sırasında notalardan üretiliyor. Checkpoint'e eklenen satırlar, V8'in en alçak satırının kopyasıyla başlıyor.

```bash
python scripts/train.py --config ml/configs/guitar_mixed_low.yaml --init ml/checkpoints/guitar_v8/best.pt --set training.lr=2e-4 training.epochs=20 paths.checkpoint_dir=ml/checkpoints/guitar_v9
python scripts/benchmark_tunings.py --model v8=ml/checkpoints/guitar_v8/best.pt+ml/checkpoints/guitar_v7/best.pt --model v9=ml/checkpoints/guitar_v9/best.pt+ml/checkpoints/guitar_v7/best.pt
```

GuitarSet testi, her tel aynı miktarda kaydırılarak (tab modeli ikisinde de V7):

| Akort | Nota F1: V8 → V9 | Tel/perde isabeti: V8 → V9 | Tam doğru nota: V8 → V9 |
|---|---|---|---|
| Standart | 0.896 → 0.895 | 0.939 → 0.940 | 0.794 → 0.795 |
| Yarım ton aşağı | 0.881 → 0.881 | 0.946 → 0.944 | 0.784 → 0.786 |
| Bir ton aşağı | 0.877 → 0.878 | 0.927 → 0.931 | 0.756 → 0.766 |
| İki ton aşağı (C) | 0.858 → 0.856 | 0.785 → 0.790 | 0.610 → **0.631** |
| Capo 2 | 0.874 → 0.875 | 0.831 → 0.828 | 0.678 → 0.680 |

V9 standart akortta V8 ile aynı sonucu veriyor. Diğer test setlerinde de aynı; yalnızca Guitar-TECHS'te nota F1'i 0.01–0.02 daha iyi. E2'nin altındaki notalar ise hâlâ zayıf. İki ton aşağıda test notalarının %2.6'sı bu aralığa düşüyor: V8 bunların hiçbirini bulamıyor, V9 üçte birini buluyor (isabet 0.38). Doğrulama setinde bu satırlar için daha düşük bir onset eşiği de denendi, ama bulduğu her doğru nota kadar yanlış nota ekledi. Daha iyisi için gerçekten alçak akortla çalınmış kayıtlarla eğitim gerekiyor.

## Dışa aktarma

- **MIDI**: notalar saniye cinsinden (kayıtla hizalı kalır), ölçü ve ton bilgisiyle; çalma teknikleri pitch bend olarak, tel başına bir kanalda ([Çalma teknikleri](#çalma-teknikleri-bend-vibrato-slide)). Vuruş takibinde her vuruşun temposu yazılır (tempo haritası), böylece DAW'daki vuruş ve ölçüler müziğinkilerle çakışır; ilk ölçü çizgisinden önceki kısım tam vuruşlu bir anakruz ölçüsüdür. Tek tempoda tek tempo yazılır.
- **MusicXML**: tek gitar partisi, iki dizek (standart nota + 6 telli TAB, her notanın tel/perdesi) ve akor sembolleri (slash akorlarda basıyla). Onset'ler vuruş ızgarasının 16'lıklarına hizalanır; ilk ölçü, ilk notadan önceki ölçü başında başlar. Ölçü çizgisini aşan notalar bağlanır, boşluklar sus olur. Vuruş takibinde tempo değiştiği ölçülere çalma temposu (%2'den büyük değişim) ve metronom işareti (%8'den büyük) yazılır; ton değiştiren şarkılarda yeni armür. Çıktı resmi MusicXML 4.0 şemasına göre geçerlidir; MuseScore ve Guitar Pro (Dosya → İçe aktar) açar.
- **ASCII TAB**: düz metin; yakın notalar birleşmez, akorlar hizalıdır. Çalma teknikleri alışılmış harflerle yazılır (`5h7`, `7p5`, `5/7`, `7b9`, `7~`; bkz. [Çalma teknikleri](#çalma-teknikleri-bend-vibrato-slide)).

Gitar ve şarkı projelerinde vuruşlar kayıttan takip edilir ([Vuruş takibi](#vuruş-takibi-metronomsuz-kayıtlar)); BPM alanı ortanca vuruşun temposunu gösterir. Vuruş takibi kapatılınca ya da yokken (ses projeleri) tempo, modelin onset tahminlerinden bulunur (GuitarSet testinde yarım/iki katı hatalar dahil %82 isabet; ses tabanlı `beat_track` ile %70). Editördeki **BPM** alanından düzeltilip kaydedildiğinde ızgara, metronom, MIDI ve MusicXML bu sabit tempoya göre yazılır.

## Gerçek kayıtlar ve grup kayıtları

**Dayanıklılık ölçümü.** Gerçek dünya kayıtlarında referans nota yok. Bu yüzden görülmemiş oyuncu 05'in kayıtları, notaları değiştirmeyen efektlerle bozularak ölçülüyor (`scripts/benchmark_robustness.py`):

| Koşul | Nota F1: efektsiz → efektli → + ek veri | Tam doğru nota: efektsiz → efektli → + ek veri |
|---|---|---|
| Temiz | 0.900 → 0.894 → 0.896 | 0.812 → 0.786 → 0.794 |
| Telefon kaydı | 0.873 → 0.880 → 0.885 | 0.719 → 0.738 → 0.759 |
| Gürültü (10 dB) | 0.876 → 0.876 → 0.884 | 0.712 → 0.729 → 0.742 |
| Büyük oda | 0.819 → 0.846 → 0.857 | 0.735 → 0.712 → 0.729 |
| Overdrive | 0.796 → 0.855 → 0.861 | 0.601 → 0.739 → 0.757 |
| Distortion | 0.774 → 0.845 → 0.853 | 0.551 → 0.713 → 0.733 |
| Echo | 0.672 → 0.764 → 0.775 | 0.762 → 0.749 → 0.765 |
| **Ortalama** | | 0.699 → 0.738 → **0.754** |

- **Efektsiz (V2+V3):** pitch shift, tempo, gürültü, reverb, EQ ve gain varyantlarıyla eğitilmiş çift.
- **Efektli (V4+V5):** bunlara ek olarak overdrive, distortion, echo, telefon, büyük oda ve düşük SNR gürültü varyantlarıyla eğitilmiş çift (`--augment`).
- **+ ek veri (V8+V7):** EGDB ve Guitar-TECHS'in gerçek elektro gitar ve mikrofon kayıtları eklenmiş hali. Efektli eğitimin temiz kayıtlarda kaybettirdiği puanın bir kısmını geri alıyor ve her koşulda V4+V5'ten iyi. V9, V8'in aralığı C2'ye genişletilmiş hali; uygulamanın varsayılanı V10, V9'un oda kayıtları ve hizalanmış EGDB ile ince ayarı.

Echo'da F1 düşük, çünkü yankılar da nota olarak yazılıyor. Test efektleri eğitimdeki GuitarSet efekt türleriyle aynı; yalnızca kayıtların kendisi görülmemiş.

**Gitarı ayırma.** Davul, bas ya da vokal içeren kayıtlarda, yükleme sırasında "Grup kaydı" seçeneği açılırsa gitar önce Demucs `htdemucs_6s` ile ayrılır. Model ağırlıkları ilk kullanımda indirilir. Seçenek editörde "Gitarı ayır" ile değiştirilip yeniden çözümlenebilir. Ölçüm, test gitarlarının MUSDB18 davul/bas/vokal kanallarıyla karıştırılmasıyla yapıldı:

| Kayıt | Ayırmasız: nota F1 / tam doğru | Gitar ayrılarak: nota F1 / tam doğru |
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
- **Başlıklar:** `nosniff`, `X-Frame-Options: DENY` ve API yanıtlarında `Content-Security-Policy: default-src 'none'` gönderiliyor. Web uygulamasının `Permissions-Policy` başlığı mikrofonu yalnızca uygulamanın kendi sayfalarına açıyor, kamera ve konumu kapatıyor.
- **Hata mesajları:** İç hata ayrıntıları yalnızca sunucu log'una yazılıyor, kullanıcıya genel bir mesaj gösteriliyor. Doğrulama hataları gönderilen değeri geri yansıtmıyor.
- **Dosya yolları:** Dosya adları temizleniyor. Silme ve sunma işlemleri yalnızca depolama klasörünün içindeki dosyalarla sınırlı.
- **İş kurtarma:** `JOB_TIMEOUT_MINUTES` süresinden uzun takılı kalan işler yeniden başlatılabiliyor. Süreç içi modda sunucu yeniden başlarken yarım kalan işler "başarısız" olarak işaretleniyor.
- **Konteynerler:** Konteynerler root olmayan bir kullanıcıyla çalışıyor ve model klasörü salt-okunur bağlanıyor.

## API

| Metot | Yol | Açıklama |
|---|---|---|
| `POST` | `/api/projects` | Ses yükle (`file`; isteğe bağlı `name`, `source` (`guitar` / `voice` / `song`), `separate_guitar`, `tuning`, `capo`; şarkılarda `capo=-1` en kolay akorları veren capoyu seçer), transkripsiyonu başlat |
| `GET` | `/api/projects` | Projeleri listele |
| `GET` | `/api/projects/{id}` | Proje durumu; çözümleme sürerken `progress` (0–1) ve `stage` |
| `DELETE` | `/api/projects/{id}` | Projeyi ve ses dosyasını sil |
| `POST` | `/api/projects/{id}/retranscribe` | Yeniden çözümle; `?source=`, `?separate_guitar=`, `?tuning=`, `?capo=` ile ayarlar değiştirilebilir |
| `GET` | `/api/projects/{id}/audio` | Orijinal ses |
| `GET` | `/api/projects/{id}/pitch` | Ses projelerinde söylenen perdenin eğrisi (`frame_rate`, `values`); gitar projelerinde 404 |
| `GET` | `/api/projects/{id}/transcription` | Notalar, tempo, akort, ortalama güven, ses modunda oktav kaydırması (`transpose`); kayıttan takip edilen `beats`, `downbeats` ve ızgaranın onları izleyip izlemediği (`beat_grid`); şarkılarda akorlar (`chords`: başlangıç, bitiş, Harte etiketi, slash akorlarda `D:maj/3`), ton değişimleri (`keys`) ve `capo_auto` |
| `PUT` | `/api/projects/{id}/notes` | Editörde düzenlenen notaları ve isteğe bağlı tempo, ölçü (`beats_per_measure`), ton (`key`), ölçü başı (`downbeat`), vuruş takibi (`beat_grid`) ve şarkı akorları (`chords`) düzeltmelerini kaydet; tel/perdesi olmayan notalara optimizer pozisyon atar |
| `GET` | `/api/projects/{id}/midi` | MIDI dışa aktar |
| `GET` | `/api/projects/{id}/musicxml` | Nota + TAB içeren MusicXML (MuseScore, Guitar Pro) |
| `GET` | `/api/projects/{id}/tab` | ASCII TAB dışa aktar |
| `GET` | `/api/projects/{id}/chordsheet` | Şarkı projelerinde düz metin akor şeması (capo'lu şekiller, ölçü ölçü, parmak dizilimleri); diğer projelerde 404 |
| `POST` | `/api/analysis` | Verilen notaların tonu, ızgarası (`beats`, numaralı `bars`, `tracked`), akorları, ton değişimleri (`keys`) ve ritim kalıbı (`rhythm`), kaydedilmemiş düzenlemeler için. `project_id` verilirse projenin kayıttan bulunanları (vuruşlar `beat_grid` ile, tanınan akorlar, tonlar, vuruşlar) da katılır |
| `POST` | `/api/chords/voicings` | Akor etiketlerinin (`labels`) bir akort (`tuning_name`) ve capo için çalınacak şekli, adı ve parmak dizilimi; `key` isimlerin yazılışını seçer (Bb / A#) |
| `POST` | `/api/render/musicxml` | Verilen notaların MusicXML'i (editördeki nota görünümü için; `tuning`, `capo`, `title` ve analiz alanları) |
| `GET` | `/api/models` | Yeni transkripsiyonların kullandığı modeller (`version`), ses modu (`voice_version`) ve şarkı modu (`song_version`) yöntemleri |

Her proje, onu çözen modelin sürümünü (`model_version`: checkpoint klasörü ve dosya parmak izi) ve kullanıcının düzenleme kaydedip kaydetmediğini (`edited`) taşır. Model değişince proje listesi eski modelle çözülenleri işaretler; düzenlenmemiş olanlar tek tuşla güncel modelle yeniden çözümlenebilir. Düzenlenmiş projeler, düzenlemeler kaybolacağı için yalnızca editörden tek tek yenilenir.

## Testler

```bash
pytest                        # Python: API, analiz, MusicXML, veri setleri, model...
cd apps/web && npm test       # Web (Vitest)
```

Web uygulaması nota adlandırmayı, ölçü numaralandırmayı ve TAB etiketlerini TypeScript'te `music_core` ile aynı şekilde yapıyor. İkisinin ayrışmaması için Python'un ürettiği ortak bir fixture (`tests/fixtures/music_web.json`) her iki tarafta da test ediliyor. Python tarafı bilerek değiştirildiğinde fixture'ı yeniden üretmek için:

```bash
python tests/test_web_fixtures.py
```

GitHub Actions (`.github/workflows/ci.yml`) her push ve pull request'te ruff, pytest, TypeScript tip denetimi, Vitest ve Next.js üretim derlemesini çalıştırır. Veri seti ya da model checkpoint'i gerekmez.
