"""
server.py
----------
Tamamen yerel çalışan İGÜ TTO bilgi asistanı.

Bu sunucu SADECE şu ikisiyle konuşur:
  1) Kendi diskindeki kb_data/faq.json ve kb_data/igu_tto_rag_documents.jsonl
  2) İlgili bilgi varsa kendi bilgisayarında çalışan Ollama (http://localhost:11434)

Claude veya claude.ai ile HİÇBİR bağlantısı yoktur. İnternet kapalıyken bile
(Ollama ve Python kütüphaneleri kurulduktan sonra) çalışır.

Çalıştırma:
    python server.py
Sonra tarayıcıda şu adresi aç:
    http://localhost:5000
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from sentence_transformers import SentenceTransformer

for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace")

KB_PATH = Path("kb_data/igu_tto_rag_documents.jsonl")
FAQ_PATH = Path("kb_data/faq.json")
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 4
FAQ_SIMILARITY_THRESHOLD = float(os.environ.get("FAQ_SIMILARITY_THRESHOLD", "0.58"))
MIN_CONTEXT_SIMILARITY_THRESHOLD = float(
    os.environ.get("MIN_CONTEXT_SIMILARITY_THRESHOLD", "0.40")
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
# Küçük ve hızlı model: 4 GB VRAM'e rahat sığar, qwen2.5:7b'den çok daha hızlı yanıt verir.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_INSTRUCTIONS = """Sen İGÜ TTO (İstanbul Gelişim Üniversitesi Teknoloji Transfer Ofisi) bilgi asistanısın.
Kurallar:
1. Sadece aşağıda verilen bağlam (context) içindeki bilgilere dayanarak cevap ver.
2. Güncel bilgi ile tarihsel/arşiv bilgiyi birbirine karıştırma; tarih belirtilmişse cevaba dahil et.
3. Bağlamda olmayan hiçbir şeyi (ücret, kesin süre, teknik detay, kuruluş yılı vb.) uydurma.
4. Bağlamda cevap yoksa bunu açıkça belirt ve kullanıcıyı şuraya yönlendir: tto@gelisim.edu.tr / 0212 422 70 00.
5. Cevabın SADECE Türkçe olsun, kısa ve net yaz.
6. Kaynak metin başka dilde olsa bile bilgiyi Türkçeye çevir; başka bir dilde kelime veya karakter kullanma.
7. Proje kaynağı yalnızca başlık veriyorsa başlığın dışında teknik ayrıntı üretme."""


def load_documents():
    docs = []
    with KB_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


print("Bilgi tabanı yükleniyor...")
DOCS = load_documents()
print(f"{len(DOCS)} belge yüklendi.")
with FAQ_PATH.open(encoding="utf-8") as f:
    FAQS = [item for item in json.load(f) if item.get("q") and item.get("a")]
print(f"{len(FAQS)} soru-cevap dosyadan yüklendi.")


def faq_topics(text):
    normalized = text.lower().translate(
        str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})
    )
    normalized = "".join(char if char.isalnum() else " " for char in normalized)
    tokens = normalized.split()

    topic_terms = {
        "definition": ("nedir", "demek", "gorev", "ne is", "is yapar"),
        "eligibility": (
            "kimler", "ogrenci", "akademisyen", "arastirma gorevl", "arastirmaci",
            "kamu kurum", "ozel sektor",
        ),
        "services": ("alan", "destek", "hizmet", "modul"),
        "funding": ("finans", "fon", "tubitak", "kosgeb", "hibe", "avrupa birligi"),
        "startup": ("sirket", "girisim", "is modeli", "mentorl", "yatirimci"),
        "application": ("basvur", "form"),
        "evaluation": ("sure", "degerlendir", "ne kadar", "zaman"),
        "protection": ("fikr", "koru", "bulus", "marka"),
        "cost": ("maliyet", "ucret", "fiyat", "bedel", "ucretsiz"),
    }

    def contains(term):
        if " " in term:
            return term in normalized
        return any(token.startswith(term) for token in tokens)

    topics = {
        topic
        for topic, terms in topic_terms.items()
        if any(contains(term) for term in terms)
    }
    if "definition" in topics and not any(
        contains(term) for term in ("tto", "ofis", "teknoloji transfer")
    ):
        topics.remove("definition")
    if contains("patent") and contains("faydali") and contains("fark"):
        topics.add("patent_difference")
    return topics


FAQ_TOPICS = [faq_topics(item["q"]) for item in FAQS]
SPECIFIC_FAQ_TOPICS = {
    "eligibility", "funding", "startup", "evaluation", "protection", "cost",
    "patent_difference",
}

print(f"Ambedding modeli yükleniyor ({EMBED_MODEL_NAME})... (ilk çalıştırmada biraz sürebilir)")
# Keep the retrieval model on CPU so Ollama can use the GPU for generation.
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

CORPUS_TEXTS = [
    f"{d.get('title','')} {d.get('text','')} {d.get('metadata',{}).get('category','')}"
    for d in DOCS
]
print("Belgeler için vektörler hesaplanıyor...")
DOC_EMBEDDINGS = EMBED_MODEL.encode(CORPUS_TEXTS, normalize_embeddings=True)
FAQ_QUESTIONS = [item["q"] for item in FAQS]
FAQ_EMBEDDINGS = EMBED_MODEL.encode(FAQ_QUESTIONS, normalize_embeddings=True)
print("Hazır.")

app = Flask(__name__, static_folder="static")


def retrieve(question, top_k=TOP_K, q_vec=None):
    if q_vec is None:
        q_vec = EMBED_MODEL.encode([question], normalize_embeddings=True)[0]
    scores = DOC_EMBEDDINGS @ q_vec
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        d = DOCS[i]
        results.append(
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "text": d.get("text"),
                "score": float(scores[i]),
                "source_url": d.get("metadata", {}).get("source_url"),
            }
        )
    return results


def rag_only_answer(question, chunks):
    """Return short source excerpts without asking a language model."""
    stop_words = {
        "nedir", "ne", "nasıl", "nasil", "hangi", "kim", "kimler", "için",
        "icin", "ile", "ve", "bir", "bu", "mi", "mı", "mu", "mü", "de",
        "da", "var", "olan", "olarak", "hakkında", "hakkinda", "bilgi",
    }
    query_terms = {
        token for token in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", question.lower())
        if len(token) > 2 and token not in stop_words
    }
    selected = []
    seen = set()

    for chunk in chunks[:3]:
        paragraphs = [
            re.sub(r"\s+", " ", p).strip()
            for p in re.split(r"\n\s*\n", chunk.get("text", ""))
        ]
        paragraphs = [
            re.sub(r"^(?:#+\s*|[-*]\s*|\d+[.)]\s*)", "", p).strip()
            for p in paragraphs
            if len(p.strip()) >= 35
        ]
        if not paragraphs:
            continue

        def paragraph_score(paragraph):
            words = set(re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", paragraph.lower()))
            return sum(
                1 for term in query_terms
                if any(word.startswith(term) or term.startswith(word) for word in words)
            )

        best_paragraph = max(paragraphs, key=paragraph_score)
        if best_paragraph in seen:
            continue
        seen.add(best_paragraph)
        if len(best_paragraph) > 520:
            best_paragraph = best_paragraph[:517].rsplit(" ", 1)[0] + "..."
        selected.append((chunk.get("title") or "İlgili kaynak", best_paragraph))
        if len(selected) == 3:
            break

    if not selected:
        return "Bilgi tabanında bu soruyla ilgili gösterilebilecek bir bölüm bulunamadı."

    return "Bilgi tabanında bulunan ilgili bölümler:\n\n" + "\n\n".join(
        f"{title}: {excerpt}" for title, excerpt in selected
    )


def retrieve_faq(question, q_vec=None):
    if q_vec is None:
        q_vec = EMBED_MODEL.encode([question], normalize_embeddings=True)[0]

    question_topics = faq_topics(question)
    specific_topics = question_topics & SPECIFIC_FAQ_TOPICS
    if specific_topics:
        candidates = [
            i for i, topics in enumerate(FAQ_TOPICS)
            if topics & specific_topics
        ]
    else:
        candidates = [
            i for i, topics in enumerate(FAQ_TOPICS)
            if topics & question_topics
        ]
    if not candidates:
        return None

    candidate_scores = FAQ_EMBEDDINGS[candidates] @ q_vec
    local_best = int(np.argmax(candidate_scores))
    best_idx = candidates[local_best]
    score = float(candidate_scores[local_best])
    threshold = FAQ_SIMILARITY_THRESHOLD
    if specific_topics == {"protection"}:
        # Turkish suffix changes make paraphrases in this topic score lower.
        threshold = min(threshold, 0.25)
    print(
        f"FAQ candidate similarity={score:.3f}; threshold={threshold:.2f}; "
        f"topics={sorted(question_topics)}"
    )
    if score < threshold:
        return None

    faq = FAQS[best_idx]
    return {
        "answer": faq["a"],
        "score": score,
        "source": {
            "id": faq.get("source", "faq"),
            "title": f"FAQ: {faq['q']}",
        },
    }


def text_response(answer, sources=None):
    sources = sources or []
    if "application/x-ndjson" not in request.headers.get("Accept", ""):
        return jsonify({"answer": answer, "sources": sources})

    def generate():
        for start in range(0, len(answer), 32):
            yield json.dumps(
                {"token": answer[start : start + 32]}, ensure_ascii=False
            ) + "\n"
        yield json.dumps({"sources": sources}, ensure_ascii=False) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def ask_ollama(context, question):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"BAĞLAM:\n{context}\n\nSORU: {question}\n\n"
                    "Yalnızca yukarıdaki bağlamdan cevap ver. Cevabı sadece Türkçe ve "
                    "en fazla iki kısa cümleyle yaz. Bağlam yetersizse bunu açıkça söyle; "
                    "başka dilde kelime veya karakter kullanma."
                ),
            },
        ],
        "stream": True,
        "keep_alive": "30m",
        "options": {"num_predict": 130, "temperature": 0.0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=(10, 180), stream=True)
    resp.raise_for_status()
    return resp


@app.route("/")
def index():
    response = send_from_directory("static", "index.html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Soru boş olamaz."}), 400

    query_embedding = EMBED_MODEL.encode([question], normalize_embeddings=True)[0]
    faq_match = retrieve_faq(question, query_embedding)
    if faq_match:
        print(f"FAQ match (similarity={faq_match['score']:.3f}): {faq_match['source']['title']}")
        return text_response(faq_match["answer"], [faq_match["source"]])

    chunks = retrieve(question, q_vec=query_embedding)
    best_score = chunks[0]["score"] if chunks else 0.0
    ollama_enabled = data.get("ollama_enabled", True) is True
    print(
        f"KB best similarity={best_score:.3f}; "
        f"Ollama enabled={ollama_enabled}; "
        f"context available={best_score >= MIN_CONTEXT_SIMILARITY_THRESHOLD}"
    )
    if best_score < MIN_CONTEXT_SIMILARITY_THRESHOLD:
        return text_response(
            "Bilgi tabanında bu soruyu yanıtlayacak yeterli bilgi bulunamadı. "
            "İGÜ TTO ile tto@gelisim.edu.tr veya 0212 422 70 00 üzerinden iletişime geçebilirsiniz."
        )

    context = "\n\n---\n\n".join(f"[{c['id']}] {c['title']}\n{c['text']}" for c in chunks)
    sources = [
        {"id": c["id"], "title": c["title"], "url": c.get("source_url")}
        for c in chunks
    ]

    if not ollama_enabled:
        print("RAG-only answer; Ollama was not called.")
        return text_response(rag_only_answer(question, chunks), sources)

    try:
        ollama_response = ask_ollama(context, question)
    except requests.exceptions.ConnectionError:
        return jsonify(
            {
                "error": (
                    "Ollama'ya bağlanılamadı. Ollama'nın açık olduğundan ve "
                    f"'{OLLAMA_MODEL}' modelinin indirilmiş olduğundan emin ol "
                    f"('ollama pull {OLLAMA_MODEL}')."
                )
            }
        ), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Hata: {e}"}), 500

    if "application/x-ndjson" not in request.headers.get("Accept", ""):
        try:
            answer_parts = []
            for line in ollama_response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                token = event.get("message", {}).get("content", "")
                if token:
                    answer_parts.append(token)
            return jsonify({"answer": "".join(answer_parts), "sources": sources})
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Yanıt alınamadı: {e}"}), 500
        finally:
            ollama_response.close()

    def generate():
        try:
            for line in ollama_response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                token = event.get("message", {}).get("content", "")
                if token:
                    yield json.dumps({"token": token}, ensure_ascii=False) + "\n"
            yield json.dumps({"sources": sources}, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"error": f"Response stream interrupted: {e}"}, ensure_ascii=False) + "\n"
        finally:
            ollama_response.close()

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print("\nSunucu başlıyor: http://localhost:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
