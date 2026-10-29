# İGÜ TTO Bilgi Asistanı — Tamamen Yerel Sürüm

Bu sürüm, Chrome'da (veya başka bir tarayıcıda) bir **yerel bağlantı**
(`http://localhost:5000`) olarak açılır ve **Claude veya claude.ai ile
hiçbir bağlantısı yoktur**. Yerel bilgi dosyalarını kullanır ve gerektiğinde
yerel Ollama modeline başvurur:

1. `kb_data/faq.json` içindeki hazır soru-cevaplar
2. `kb_data/igu_tto_rag_documents.jsonl` içindeki bilgi tabanı
3. FAQ eşleşmesi veya yeterli bilgi bulunmazsa bilgisayarında çalışan **Ollama**

Sorularınız ve cevaplar hiçbir zaman bilgisayarınızın dışına çıkmaz.

---

## Kurulum (bir kerelik)

### 1. Python kütüphaneleri
```bash
pip install -r requirements.txt
```

### 2. Ollama kurulumu
[ollama.com](https://ollama.com) adresinden indirip kur. Sonra bir model indir:
```bash
ollama pull qwen2.5:3b
```

> ⚠️ **Önemli:** `qwen2.5:7b` gibi daha büyük modeller 4 GB VRAM'e (örn. GTX 1650)
> tam sığmaz; bu durumda model kısmen CPU üzerinde çalışır ve bir cevap üretmesi
> dakikalar sürebilir. `qwen2.5:3b` VRAM'e rahat sığar ve çok daha hızlıdır.
> Daha da hafif bir seçenek istersen: `ollama pull llama3.2:3b` (bu durumda
> `server.py` içindeki `OLLAMA_MODEL` değişkenini de güncelle).

Ollama'nın çalıştığından emin ol (uygulamayı açık bırak, ya da `ollama serve`).

---

## Çalıştırma

```bash
python server.py
```

VS Code'da terminal ana çalışma klasöründe açıldıysa PowerShell'de şu komutları kullan:

```powershell
cd .\to_local_web
& "$env:LOCALAPPDATA\Python\bin\python.exe" -u server.py
```

Terminalde şunu göreceksin:
```
Sunucu başlıyor: http://localhost:5000
```

Chrome'u aç ve şu adrese git:

**http://localhost:5000**

Artık soru sorabilirsin. Her şey bu bilgisayarda, yerel olarak çalışır.

---

## Nasıl çalışıyor?

1. Soru sorduğunda, tarayıcı soruyu `server.py`'a (kendi bilgisayarında çalışan
   küçük bir Flask sunucusu) gönderir.
2. Sunucu önce FAQ dosyasındaki 10 soruyu anlam temelli karşılaştırır ve güçlü
   bir eşleşme bulursa kayıtlı cevabı doğrudan döndürür.
3. FAQ eşleşmezse, TTO, üniversite ve kullanıcı tarafından eklenen kaynaklardan derlenen bilgi tabanındaki 91 belgeyi tarar.
4. Üstteki `Ollama Açık/Kapalı` düğmesi kapalıysa eşleşen RAG metinlerini kaynaklarıyla gösterir. Açık olduğu zaman ilgili bağlamı yerel Ollama'ya gönderir ve Türkçe yanıt oluşturur.
5. Bilgi tabanında yeterli içerik yoksa bunu açıkça bildirir.

Uygulama Claude'a veya harici bir servise bağlanmaz. Ollama çağrısı yalnızca
gerektiğinde `http://localhost:11434` adresindeki yerel modele gider.

---

## Bilgi tabanını güncellemek

`kb_data/igu_tto_rag_documents.jsonl` veya `kb_data/faq.json` dosyasını düzenle,
sonra sunucuyu yeniden başlat
(`Ctrl+C` ile durdur, `python server.py` ile tekrar çalıştır). Her başlangıçta
belgeler otomatik olarak yeniden taranır, ayrı bir "build" adımı gerekmez.
