import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Music Transcriber",
  description: "Gitar kayıtlarını notaya, MIDI'ye ve tab'a dönüştürün.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="tr">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            Music Transcriber
          </Link>
          <nav>
            <Link href="/upload">Yükle</Link>
            <Link href="/projects">Projeler</Link>
            <Link href="/settings">Ayarlar</Link>
          </nav>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
