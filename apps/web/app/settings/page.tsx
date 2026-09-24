"use client";

import { API_URL } from "@/lib/api";
import { useSettings } from "@/lib/settings";

export default function SettingsPage() {
  const { settings, update, reset } = useSettings();

  return (
    <div className="stack" style={{ maxWidth: 720 }}>
      <h1>Ayarlar</h1>

      <section className="card stack">
        <h3>Editör</h3>
        <label className="field">
          Yakınlaştırma: {settings.pixelsPerSecond} px/sn
          <input
            type="range"
            min={20}
            max={400}
            step={10}
            value={settings.pixelsPerSecond}
            onChange={(e) => update({ pixelsPerSecond: Number(e.target.value) })}
          />
        </label>
        <label className="row">
          <input type="checkbox" checked={settings.showTab} onChange={(e) => update({ showTab: e.target.checked })} />
          Gitar TAB&apos;ı göster
        </label>
        <label className="row">
          <input
            type="checkbox"
            checked={settings.showPianoRoll}
            onChange={(e) => update({ showPianoRoll: e.target.checked })}
          />
          Piyano rulosunu göster
        </label>
        <label className="row">
          <input
            type="checkbox"
            checked={settings.followPlayhead}
            onChange={(e) => update({ followPlayhead: e.target.checked })}
          />
          Oynatırken imleci takip et
        </label>
        <div>
          <button onClick={reset}>Varsayılanlara dön</button>
        </div>
      </section>

      <section className="card stack">
        <h3>Sunucu</h3>
        <p>
          API adresi: <code>{API_URL}</code>
        </p>
        <p className="muted small">
          <code>NEXT_PUBLIC_API_URL</code> ortam değişkeniyle değiştirilebilir. Ayarlar yalnızca bu tarayıcıda saklanır.
        </p>
      </section>
    </div>
  );
}
