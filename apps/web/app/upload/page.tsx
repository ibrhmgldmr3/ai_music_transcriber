import AudioUploader from "@/components/AudioUploader";

export default function UploadPage() {
  return (
    <div className="stack" style={{ maxWidth: 720 }}>
      <h1>Kayıt yükle</h1>
      <p className="muted">
        Tek bir gitarın çaldığı, mümkün olduğunca temiz bir kayıt en iyi sonucu verir. Gitarınız yoksa bir
        melodiyi söyleyip ya da mırıldanıp TAB&apos;ını çıkarabilirsiniz. Çözümleme arka planda çalışır; bitince
        editör otomatik olarak güncellenir.
      </p>
      <AudioUploader />
    </div>
  );
}
