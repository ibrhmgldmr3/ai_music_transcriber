import Link from "next/link";

const STEPS = [
  {
    title: "1. Yükle",
    text: "WAV, MP3, FLAC… bir gitar kaydı yükleyin.",
  },
  {
    title: "2. Çözümle",
    text: "CNN + BiLSTM modeli her notanın pitch, onset ve offset değerini tahmin eder; bir optimizasyon algoritması her notayı en rahat çalınacak tel ve perdeye yerleştirir.",
  },
  {
    title: "3. Düzenle ve dışa aktar",
    text: "Sesi dinleyerek notaları düzeltin, sonucu MIDI veya TAB olarak indirin.",
  },
];

export default function HomePage() {
  return (
    <div className="stack">
      <section className="hero">
        <h1>Gitar kayıtlarını notaya, MIDI&apos;ye ve tab&apos;a dönüştürün</h1>
        <p className="muted">
          Otomatik müzik transkripsiyonu (AMT) ile kaydınızdaki notaları çıkarın, editörde dinleyip düzeltin.
        </p>
        <div className="row">
          <Link href="/upload" className="button primary">
            Kayıt yükle
          </Link>
          <Link href="/projects" className="button">
            Projelerim
          </Link>
        </div>
      </section>
      <section className="grid-3">
        {STEPS.map((step) => (
          <div key={step.title} className="card stack tight">
            <h3>{step.title}</h3>
            <p className="muted">{step.text}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
