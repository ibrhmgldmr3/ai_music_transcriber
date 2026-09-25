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
Müzik analizi (ton, vuruş/ölçü ızgarası, akor sembolleri) ──► MusicXML armür, ölçü, akorlar
   │
   ▼
Müzik editörü (dalga formu, TAB + akorlar, piyano rulosu, sentezle dinleme, döngü, metronom,
               düzenleme, MIDI/MusicXML/TAB dışa aktarma)
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

Bu tablo yalnızca GuitarSet ile eğitilmiş ilk çiftin sonuçları. Uygulamanın varsayılanı artık elektro gitar verisiyle de eğitilmiş V8 + V7 çifti (bkz. [Ek veri setleri](#ek-veri-setleri-elektro-gitar-ve-oda-mikrofonları)).

## Dizin yapısı

| Dizin | İçerik |
|---|---|
| `apps/web` | Next.js arayüzü: yükleme, proje listesi, editör, ayarlar |
| `apps/api` | FastAPI + SQLAlchemy + Celery (transkripsiyon işleri) |
| `ml/configs` | `base.yaml` (ortak), `guitar.yaml` (GuitarSet), `guitar_tab.yaml` (tab kafası), `guitar_mixed.yaml` / `guitar_tab_mixed.yaml` (ek veri setleriyle) |
| `ml/preprocessing` | ses yükleme, log-mel/CQT, JAMS ve tel başına MIDI anotasyonları, augmentation |
| `ml/datasets` | GuitarSet, EGDB ve Guitar-TECHS keşfi ve split'leri, PyTorch dataset |
| `ml/models` | `ConvStack`, `CNN`, `CRNN` (BiLSTM/GRU), çıkış kafaları |
| `ml/training` | eğitim döngüsü, loss, callback'ler |
| `ml/evaluation` | frame/nota/tab metrikleri, değerlendirme, görselleştirme |
| `ml/inference` | nota çözümleme (Onsets & Frames tarzı) ve `Predictor` |
| `packages/music-core` | `Note`, MIDI, MusicXML, zamanlama, tab optimizasyonu, ton/ölçü/akor analizi (`music_core` olarak import edilir) |
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

Modeller `ml/checkpoints` klasöründen salt-okunur bağlanır. Varsayılan çift `guitar_v8` (notalar) ve `guitar_v7` (tel/perde); başka klasörler `NOTES_MODEL` ve `TAB_MODEL` ortam değişkenleriyle (ör. proje kökündeki `.env`) seçilir. Checkpoint yoksa yüklenen kayıtlar açıklayıcı bir hata mesajıyla **Başarısız** durumuna düşer. CPU imajı yeterince hızlı: 22 saniyelik bir kaydı CPU'da 0.7 s'de çözüyor. GPU asıl şarkı modunda (Demucs) fark ediyor ve imajı birkaç GB büyütüyor. Demucs ağırlıkları ilk kullanımda indirilip `model-cache` volume'unda saklanır.

### Lokal geliştirme

API ayarları `apps/api/.env` dosyasından okunuyor. Örnek dosyayı kopyalayıp hangi modelin kullanılacağını (`MODEL_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v8/best.pt`, tab modeli için `MODEL_TAB_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v7/best.pt`) orada seç. Göreli yollar proje köküne göre çözülüyor; veritabanı ve yüklenen sesler varsayılan olarak `apps/api` altında tutuluyor.

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
- **Izgara ve akorlar:** TAB ve piyano rulosunda vuruş ve ölçü çizgileri, ölçü numaraları (MusicXML ölçüleriyle aynı) ve TAB'ın üstünde akor sembolleri görünür. Bir notayı düzenleyince ya da BPM, ölçü veya tonu değiştirince bunlar kaydetmeden güncellenir (`POST /api/analysis`).

## Ton, ölçü ve akorlar

Hepsi yalnızca notalardan çıkarılır (`packages/music-core/analysis.py`), model yeniden çalışmaz:

- **Tempo inceltme:** Ses tabanlı tempo tahmini kaba bir çözünürlükte çalışır, ama %1'lik hata bile ızgarayı her 100 vuruşta bir vuruş kaydırır. Tahmin, notaların vuruş ve yarım vuruş periyodikliğini en iyi açıklayan tempoya (±%4 içinde) inceltilir.
- **Ton:** Nota sürelerinin perde sınıfı histogramı, Temperley–Kostka–Payne majör/minör profilleriyle karşılaştırılır. Armür ve nota/akor isimleri (F majörde Bb, A minörde G#) buna göre yazılır.
- **Ölçü ızgarası:** Uzun ve bas notaların vuruşa düştüğü faz seçilir. Ölçü başı, armoninin en çok değiştiği vuruştur (akorlar genelde ölçü başında değişir).
- **Akorlar:** Her vuruşta çalan notaların kroması; majör, minör, 7, maj7, m7, power (5), sus2, sus4 ve dim şablonlarıyla eşleştirilir, Viterbi ile yumuşatılır, bas notası farklıysa slash akor (D/F#) yazılır. Tek sesli melodilere akor yazılmaz.

GuitarSet oyuncu 05'in model transkripsiyonları üzerinde (uygulamanın gördüğü girdi):

| Ölçüm | Eşlik (comp) | Solo |
|---|---|---|
| Ton (MIREX ağırlıklı skor) | 0.82 | 0.76 |
| Tempo %0.3 içinde (inceltme öncesi → sonrası, tahmin %4 içindeyken) | %14 → %94 | |
| Ölçü başı doğru (tempo doğruyken) | %73 | %8 |
| Akor kökü / majör-minör doğru (zaman oranı) | 0.64 / 0.59 | akor yazılmıyor |

Ölçü başı sololarda neredeyse tahmin edilemiyor, bu yüzden editörde elle düzeltilebilir: bir notayı seçip **Seçili nota 1. vuruş**'a basmak, ölçü çizgilerini o notaya hizalar. **Ölçü** (2/4–7/4) ve **Ton** seçimleri de aynı şekilde kaydedilir; "Otomatik"e dönünce tahmin kullanılır.

## Dışa aktarma

- **MIDI**: notalar saniye cinsinden (kayıtla hizalı kalır), tempo, ölçü ve ton bilgisiyle.
- **MusicXML**: tek gitar partisi, iki dizek (standart nota + 6 telli TAB, her notanın tel/perdesi) ve akor sembolleri. Onset'ler tempoya göre 16'lık ızgaraya hizalanır; ilk ölçü, ilk notadan önceki ölçü başında başlar. Ölçü çizgisini aşan notalar bağlanır, boşluklar sus olur. Çıktı resmi MusicXML 4.0 şemasına göre geçerlidir; MuseScore ve Guitar Pro (Dosya → İçe aktar) açar.
- **ASCII TAB**: düz metin; yakın notalar birleşmez, akorlar hizalıdır.

Tempo, modelin onset tahminlerinden bulunur (GuitarSet testinde yarım/iki katı hatalar dahil %82 isabet; ses tabanlı `beat_track` ile %70). Swing/triplet hisli parçalarda yanılabilir; editördeki **BPM** alanından düzeltilip kaydedildiğinde ızgara, metronom, MIDI ve MusicXML bu tempoya göre yazılır. Tek bir sabit tempo varsayılıyor: metronomsuz, hızlanıp yavaşlayan kayıtlarda ızgara zamanla kayar.

## Gerçek kayıtlar ve şarkı modu

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
- **+ ek veri (V8+V7):** uygulamanın varsayılanı; EGDB ve Guitar-TECHS'in gerçek elektro gitar ve mikrofon kayıtları eklenmiş hali. Efektli eğitimin temiz kayıtlarda kaybettirdiği puanın bir kısmını geri alıyor ve her koşulda V4+V5'ten iyi.

Echo'da F1 düşük, çünkü yankılar da nota olarak yazılıyor. Test efektleri eğitimdeki GuitarSet efekt türleriyle aynı; yalnızca kayıtların kendisi görülmemiş.

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
| `PUT` | `/api/projects/{id}/notes` | Editörde düzenlenen notaları ve isteğe bağlı tempo, ölçü (`beats_per_measure`), ton (`key`) ve ölçü başı (`downbeat`) düzeltmelerini kaydet; tel/perdesi olmayan notalara optimizer pozisyon atar |
| `GET` | `/api/projects/{id}/midi` | MIDI dışa aktar |
| `GET` | `/api/projects/{id}/musicxml` | Nota + TAB içeren MusicXML (MuseScore, Guitar Pro) |
| `GET` | `/api/projects/{id}/tab` | ASCII TAB dışa aktar |
| `POST` | `/api/analysis` | Verilen notaların tonu, ölçü ızgarası ve akorları (kaydedilmemiş düzenlemeler için) |
| `GET` | `/api/models` | Yeni transkripsiyonların kullandığı modeller (`version`) |

Her proje, onu çözen modelin sürümünü (`model_version`: checkpoint klasörü ve dosya parmak izi) ve kullanıcının düzenleme kaydedip kaydetmediğini (`edited`) taşır. Model değişince proje listesi eski modelle çözülenleri işaretler; düzenlenmemiş olanlar tek tuşla güncel modelle yeniden çözümlenebilir. Düzenlenmiş projeler, düzenlemeler kaybolacağı için yalnızca editörden tek tek yenilenir.

## Testler

```bash
pytest                        # Python: API, analiz, MusicXML, veri setleri, model...
cd apps/web && npm test       # Web (Vitest)
```

Web uygulaması nota adlandırmayı ve ölçü numaralandırmayı TypeScript'te `music_core` ile aynı şekilde yapıyor. İkisinin ayrışmaması için Python'un ürettiği ortak bir fixture (`tests/fixtures/music_web.json`) her iki tarafta da test ediliyor. Python tarafı bilerek değiştirildiğinde fixture'ı yeniden üretmek için:

```bash
python tests/test_web_fixtures.py
```

GitHub Actions (`.github/workflows/ci.yml`) her push ve pull request'te ruff, pytest, TypeScript tip denetimi, Vitest ve Next.js üretim derlemesini çalıştırır. Veri seti ya da model checkpoint'i gerekmez.
