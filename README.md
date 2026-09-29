# Music Transcriber

Gitar kayıtlarını nota olaylarına, MIDI'ye ve gitar tab'ına dönüştüren uçtan uca bir otomatik müzik transkripsiyonu (AMT) projesi. Gitar yoksa söylenen, mırıldanan ya da ıslıkla çalınan bir melodinin de TAB'ı çıkarılabilir (bkz. [Ses → gitar](#ses--gitar-söyle-tabını-al)).

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

Bu tablo yalnızca GuitarSet ile eğitilmiş ilk çiftin sonuçları. Uygulamanın varsayılanı artık elektro gitar verisiyle de eğitilmiş ve C2'ye kadar inen V9 + V7 çifti (bkz. [Ek veri setleri](#ek-veri-setleri-elektro-gitar-ve-oda-mikrofonları) ve [Farklı akortlar ve capo](#farklı-akortlar-ve-capo)).

## Dizin yapısı

| Dizin | İçerik |
|---|---|
| `apps/web` | Next.js arayüzü: yükleme, proje listesi, editör, ayarlar |
| `apps/api` | FastAPI + SQLAlchemy + Celery (transkripsiyon işleri) |
| `ml/configs` | `base.yaml` (ortak), `guitar.yaml` (GuitarSet), `guitar_tab.yaml` (tab kafası), `guitar_mixed.yaml` / `guitar_tab_mixed.yaml` (ek veri setleriyle), `guitar_mixed_low.yaml` (C2'ye inen nota aralığı) |
| `ml/preprocessing` | ses yükleme, log-mel/CQT, JAMS ve tel başına MIDI anotasyonları, augmentation |
| `ml/datasets` | GuitarSet, EGDB ve Guitar-TECHS keşfi ve split'leri, PyTorch dataset |
| `ml/models` | `ConvStack`, `CNN`, `CRNN` (BiLSTM/GRU), çıkış kafaları |
| `ml/training` | eğitim döngüsü, loss, callback'ler |
| `ml/evaluation` | frame/nota/tab metrikleri, değerlendirme, görselleştirme |
| `ml/inference` | nota çözümleme (Onsets & Frames tarzı), `Predictor`, şarkı modu (Demucs) ve ses modu (`voice.py`) |
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

Modeller `ml/checkpoints` klasöründen salt-okunur bağlanır. Varsayılan çift `guitar_v9` (notalar) ve `guitar_v7` (tel/perde); başka klasörler `NOTES_MODEL` ve `TAB_MODEL` ortam değişkenleriyle (ör. proje kökündeki `.env`) seçilir. Checkpoint yoksa yüklenen kayıtlar açıklayıcı bir hata mesajıyla **Başarısız** durumuna düşer. CPU imajı yeterince hızlı: 22 saniyelik bir kaydı CPU'da 0.7 s'de çözüyor. GPU asıl şarkı modunda (Demucs) fark ediyor ve imajı birkaç GB büyütüyor. Demucs ağırlıkları ilk kullanımda indirilip `model-cache` volume'unda saklanır.

### Lokal geliştirme

API ayarları `apps/api/.env` dosyasından okunuyor. Örnek dosyayı kopyalayıp hangi modelin kullanılacağını (`MODEL_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v9/best.pt`, tab modeli için `MODEL_TAB_CHECKPOINT`, örneğin `ml/checkpoints/guitar_v7/best.pt`) orada seç. Göreli yollar proje köküne göre çözülüyor; veritabanı ve yüklenen sesler varsayılan olarak `apps/api` altında tutuluyor.

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

Kaydedilmiş hale geri alınınca "kaydedilmemiş değişiklik" uyarısı da kalkar. Boş alana tıklamak oynatma konumunu oraya taşır.

## Yükleme, kayıt ve ilerleme

- **Mikrofonla kayıt:** Yükleme sayfasında "Mikrofonla kaydet" tarayıcıda WAV kaydeder (en fazla 8 dakika, seviye göstergesiyle) ve dosya seçmişsin gibi yükler. Yankı giderme, gürültü bastırma ve otomatik kazanç kapalı: konuşma için yapılmış bu işlemler gitar notalarını bozuyor. Tarayıcılar mikrofona yalnızca `localhost` ya da HTTPS üzerinden izin verir.
- **İlerleme:** Çözümleme sürerken proje listesi ve editör aşamayı (yükleniyor, gitar ayrılıyor, notalar çözülüyor, son işlemler) ve yüzdeyi gösterir. Model, uzun kayıtları parça parça işlerken ilerlemeyi bildirir; API bunu en fazla yarım saniyede bir veritabanına yazar (`progress`, `stage`).

## Ses → gitar: söyle, TAB'ını al

Yüklerken "Ne kaydettiniz?" sorusuna **Ses (şarkı, mırıldanma, ıslık)** cevabı verilirse gitar modeli çalışmaz; tek sesli bir melodi notalara çevrilip seçilen akortta gitara yerleştirilir (`ml/inference/voice.py`). Sonrası gitar projeleriyle aynı: editör, TAB, MIDI/MusicXML ve **Gitar sesi indir** ile melodinin gitarla çalınmış hali (WAV). Eğitilmiş bir model yok, dolayısıyla eğitim de gerekmiyor:

1. **Perde takibi:** pYIN (librosa) sesin temel frekansını ve sesli olup olmadığını izler; YIN'in daha ince tahmini, ikisi uyuştuğunda onun yerine geçer.
2. **Kişisel akort:** Çoğu kişi tam A = 440 Hz'e göre söylemez. Kaydın ortalama sapması ölçülür ve yarım tonlar ona göre sayılır. 40 cent tiz söyleyen biri, her notası yarım ton yukarı yazılmadan transkribe edilir.
3. **Notalara bölme:** Kareleri tek tek en yakın yarım tona yuvarlamak vibratoyu ve uzun notalardaki kaymayı ayrı notalara böler. Bunun yerine, bir anın 200 ms öncesi ve sonrasının ortalama perdesi en az 0.7 yarım ton farklıysa oradan kesilir. Her nota, karelerinin medyan perdesini alır. Kısa parçalar (notalar arası kayış, notaya aşağıdan giriş) komşusuna katılır. Nota, perdesi yerine oturduğu yerde başlar. Aynı perdede tekrarlanan heceler ("da-da-da") ancak belirgin bir ünsüz izi varsa ayrılır: spektral değişim ile sesliliğin ve ses şiddetinin çukuru birlikte.
4. **Gitara yerleştirme:** Melodi gitarın aralığına sığmıyorsa (ıslık genelde bir iki oktav yukarıda) tam oktavlarla kaydırılır, editör bunu belirtir (`transpose`). Tel/perdeyi tab optimizer'ı seçer, tempo nota başlarından bulunur.

Arkada müzik varsa "Müzik eşliğinde söylenmiş" seçeneği önce Demucs ile vokali ayırır. Aynı seçenek gitar projelerinde gitarı ayırır.

**Ölçüm.** [VocalSet](https://zenodo.org/records/1442513) (20 profesyonel şarkıcı; gamlar, arpejler, uzun notalar ve üç kısa şarkı) ve her notanın başlangıç, bitiş ve perdesini işaretleyen [Annotated-VocalSet](https://zenodo.org/records/7061507) ile yapıldı (ikisi de CC BY 4.0). Ayarlar 6 şarkıcıda seçildi, aşağıdaki sonuçlar hiç kullanılmayan diğer 14 şarkıcıdan:

```bash
python scripts/benchmark_voice.py --vocalset ml/data/raw/vocalset/VocalSet11.zip --annotations ml/data/raw/vocalset/annotated_vocalset.zip
```

| Kayıtlar (14 şarkıcı) | Kayıt | Nota F1 (başlangıç ±50 ms) | Nota F1 (±100 ms) | Yalnız başlangıç F1 |
|---|---|---|---|---|
| Gamlar | 645 | 0.26 | 0.64 | 0.33 |
| Arpejler | 683 | 0.35 | 0.64 | 0.45 |
| Uzun notalar | 202 | 0.39 | 0.52 | 0.52 |
| Şarkılar | 77 | 0.38 | 0.61 | 0.46 |
| **Hepsi** | 1607 | **0.32** | **0.62** | 0.41 |

Etiketler notaların yazılı perdesini veriyor, ama kayıtların çoğu ondan yarım ton (bazen bir oktav) farklı söylenmiş. Bu yüzden nota F1 her kaydı kendi tonunda, ona en iyi uyan tam yarım ton kaydırmasıyla ölçüyor. ±50 ms ile ±100 ms arasındaki büyük fark çoğunlukla bir tanım farkından geliyor: etiketler legato bir notayı, kayış bittikten sonra perde oturduğunda başlatıyor. Bu yöntemin başlangıçları ise o kayışın içinde, ortalama 50 ms önce kalıyor.

Sınırlar:

- Net ve ayrık söylenmiş melodilerde iyi çalışıyor. Hızlı pasajlar ve legato gamlar zorluyor: kayışla notanın kendisi arasındaki sınır belirsiz.
- Aynı perdede sözle tekrarlanan notalar, ünsüz belirgin değilse tek nota olarak çıkıyor.
- Üst üste kaydedilmiş vokaller (armoni, dublaj) ve ağır efektli vokaller pYIN'e sesli görünmüyor. Bir MUSDB şarkısının vokal kanalında karelerin yalnızca %7'si sesli sayıldı.
- Tek sesli çalışıyor; aynı anda birden fazla nota (akor) çıkarmıyor.

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
- **+ ek veri (V8+V7):** EGDB ve Guitar-TECHS'in gerçek elektro gitar ve mikrofon kayıtları eklenmiş hali. Efektli eğitimin temiz kayıtlarda kaybettirdiği puanın bir kısmını geri alıyor ve her koşulda V4+V5'ten iyi. Uygulamanın varsayılan nota modeli V9, V8'in aralığı C2'ye genişletilmiş hali.

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
- **Başlıklar:** `nosniff`, `X-Frame-Options: DENY` ve API yanıtlarında `Content-Security-Policy: default-src 'none'` gönderiliyor. Web uygulamasının `Permissions-Policy` başlığı mikrofonu yalnızca uygulamanın kendi sayfalarına açıyor, kamera ve konumu kapatıyor.
- **Hata mesajları:** İç hata ayrıntıları yalnızca sunucu log'una yazılıyor, kullanıcıya genel bir mesaj gösteriliyor. Doğrulama hataları gönderilen değeri geri yansıtmıyor.
- **Dosya yolları:** Dosya adları temizleniyor. Silme ve sunma işlemleri yalnızca depolama klasörünün içindeki dosyalarla sınırlı.
- **İş kurtarma:** `JOB_TIMEOUT_MINUTES` süresinden uzun takılı kalan işler yeniden başlatılabiliyor. Süreç içi modda sunucu yeniden başlarken yarım kalan işler "başarısız" olarak işaretleniyor.
- **Konteynerler:** Konteynerler root olmayan bir kullanıcıyla çalışıyor ve model klasörü salt-okunur bağlanıyor.

## API

| Metot | Yol | Açıklama |
|---|---|---|
| `POST` | `/api/projects` | Ses yükle (`file`; isteğe bağlı `name`, `source` (`guitar` / `voice`), `separate_guitar`, `tuning`, `capo`), transkripsiyonu başlat |
| `GET` | `/api/projects` | Projeleri listele |
| `GET` | `/api/projects/{id}` | Proje durumu; çözümleme sürerken `progress` (0–1) ve `stage` |
| `DELETE` | `/api/projects/{id}` | Projeyi ve ses dosyasını sil |
| `POST` | `/api/projects/{id}/retranscribe` | Yeniden çözümle; `?source=`, `?separate_guitar=`, `?tuning=`, `?capo=` ile ayarlar değiştirilebilir |
| `GET` | `/api/projects/{id}/audio` | Orijinal ses |
| `GET` | `/api/projects/{id}/transcription` | Notalar, tempo, akort, ortalama güven, ses modunda oktav kaydırması (`transpose`) |
| `PUT` | `/api/projects/{id}/notes` | Editörde düzenlenen notaları ve isteğe bağlı tempo, ölçü (`beats_per_measure`), ton (`key`) ve ölçü başı (`downbeat`) düzeltmelerini kaydet; tel/perdesi olmayan notalara optimizer pozisyon atar |
| `GET` | `/api/projects/{id}/midi` | MIDI dışa aktar |
| `GET` | `/api/projects/{id}/musicxml` | Nota + TAB içeren MusicXML (MuseScore, Guitar Pro) |
| `GET` | `/api/projects/{id}/tab` | ASCII TAB dışa aktar |
| `POST` | `/api/analysis` | Verilen notaların tonu, ölçü ızgarası ve akorları (kaydedilmemiş düzenlemeler için) |
| `POST` | `/api/render/musicxml` | Verilen notaların MusicXML'i (editördeki nota görünümü için; `tuning`, `capo`, `title` ve analiz alanları) |
| `GET` | `/api/models` | Yeni transkripsiyonların kullandığı modeller (`version`) ve ses modu yöntemi (`voice_version`) |

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
