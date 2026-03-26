from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import re
import httpx

load_dotenv()

app = FastAPI(title="Mawda Portfolio RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openrouter/auto"

KB = []
if KB_PATH.exists():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
else:
    print(f"WARNING: knowledge_base.json not found at {KB_PATH}")

print("MODEL:", MODEL_NAME)


class ChatRequest(BaseModel):
    question: str


def is_arabic(text: str) -> bool:
    """Detect if the input contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


def fix_mixed_text(text: str) -> str:
    """Small cleanup for mixed Arabic/English spacing."""
    text = str(text or "")
    text = re.sub(r"و([A-Za-z])", r"و \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\u0600-\u06FF\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def retrieve_chunks(question: str, top_k: int = 3):
    q = normalize_text(question)
    q_words = [w for w in q.split() if len(w) > 1]

    if not q_words:
        return []

    scored_results = []

    for item in KB:
        title = item.get("title", "")
        content = item.get("content", "")
        full_text = normalize_text(f"{title} {content}")
        normalized_title = normalize_text(title)

        score = 0
        for word in q_words:
            if word in full_text:
                score += 2
            if word in normalized_title:
                score += 3

        if score > 0:
            scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored_results[:top_k]]


def build_context(matches, lang: str = "en") -> str:
    """Build context in the same language style as the user's question."""
    if lang == "ar":
        return "\n\n".join(
            f"العنوان: {item.get('title', '')}\nالمحتوى: {item.get('content', '')}"
            for item in matches
        )

    return "\n\n".join(
        f"Title: {item.get('title', '')}\nContent: {item.get('content', '')}"
        for item in matches
    )


def clean_arabic_response(text: str) -> str:
    text = str(text or "").strip()

    english_fallback_patterns = [
        r"(?i)^the context does not provide.*",
        r"(?i)^based on the context.*",
        r"(?i)^this information is not mentioned.*",
        r"(?i)^according to the context.*",
    ]

    for pattern in english_fallback_patterns:
        if re.search(pattern, text):
            return "هذه المعلومة غير مذكورة في البورتفوليو."

    replacements = {
        "This information is not mentioned in the portfolio.": "هذه المعلومة غير مذكورة في البورتفوليو.",
        "Mawda works with": "تعمل مودة باستخدام",
        "Mawda has skills in": "تمتلك مودة مهارات في",
        "Mawda's portfolio includes": "يتضمن بورتفوليو مودة",
        "based on the context": "",
        "according to the context": "",
    }

    for src, target in replacements.items():
        text = text.replace(src, target)

    text = fix_mixed_text(text)
    text = re.sub(r"\s+", " ", text).strip()

    # If the reply is still mostly English, return a safe Arabic fallback
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    english_words = len(re.findall(r"[A-Za-z]{3,}", text))
    if arabic_chars < 3 and english_words > 4:
        return "هذه المعلومة غير مذكورة في البورتفوليو."

    return text


def clean_english_response(text: str) -> str:
    text = str(text or "").strip()

    if "هذه المعلومة غير مذكورة في البورتفوليو" in text:
        return "This information is not mentioned in the portfolio."

    # Remove Arabic if any leaked into the English answer
    text = re.sub(r"[\u0600-\u06FF]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


async def generate_rag_answer(question: str, context: str, lang: str) -> str:
    if lang == "ar":
        system_prompt = """
أنت مساعد ذكي خاص ببورتفوليو مودة القرافي.

الهدف:
تقديم إجابات دقيقة، احترافية، وسريعة عن خبرات ومهارات ومشاريع مودة.

قواعد صارمة:
- إذا كان سؤال المستخدم بالعربية، يجب أن تكون الإجابة بالعربية فقط.
- استخدم الإنجليزية فقط في أسماء الأدوات والتقنيات مثل: Python, SQL, Power BI, Tableau, FastAPI, Streamlit.
- لا تبدأ الجواب بأي جملة إنجليزية.
- لا تنسخ النص الإنجليزي من السياق كما هو.
- أعد صياغة المعلومات داخل جمل عربية طبيعية وواضحة.
- لا تضف أي معلومات غير موجودة في السياق.
- إذا لم توجد الإجابة في السياق، قل فقط:
هذه المعلومة غير مذكورة في البورتفوليو.

أسلوب الإجابة:
- اجعل الإجابة مختصرة جدًا: جملة إلى جملتين كحد أقصى.
- استخدم أسلوبًا بشريًا طبيعيًا، وليس ترجمة حرفية.
- ابدأ الإجابة مباشرة دون مقدمات مثل: بالتأكيد، حسب المعلومات، بناءً على السياق.
- ركّز على المعلومة المهمة فقط.
"""
    else:
        system_prompt = """
You are a portfolio assistant for Mawda Alguraafi.

Goal:
Provide accurate, professional, and fast answers about Mawda's experience, skills, and projects.

Strict rules:
- Answer only in English.
- Do not use Arabic at all.
- Do not mix languages.
- Do not add any information not found in the context.
- If the answer is not found, say exactly:
This information is not mentioned in the portfolio.

Style:
- Keep answers short: 1-2 sentences maximum.
- Sound natural and human, not robotic.
- Start directly with the answer.
- Do not use fillers such as: Sure, Based on the context, According to the context.
- Focus only on the key information.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portfolio2-rme6.onrender.com",
        "X-Title": "Mawda Portfolio",
    }

    if lang == "ar":
        user_prompt = f"السؤال:\n{question}\n\nالسياق:\n{context}"
    else:
        user_prompt = f"Question:\n{question}\n\nContext:\n{context}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 140,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        if response.status_code >= 400:
            try:
                error_json = response.json()
            except Exception:
                error_json = {"raw_text": response.text}
            raise Exception(f"OpenRouter error {response.status_code}: {error_json}")

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}


@app.get("/test-openrouter")
async def test_openrouter():
    if not OPENROUTER_API_KEY:
        return {"status": "error", "message": "OPENROUTER_API_KEY is missing"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portfolio2-rme6.onrender.com",
        "X-Title": "Mawda Portfolio",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Say hello in one short sentence."}
        ],
        "temperature": 0,
        "max_tokens": 20,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        try:
            body = response.json()
        except Exception:
            body = response.text

        return {
            "status_code": response.status_code,
            "body": body,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    lang = "ar" if is_arabic(question) else "en"

    if not question:
        return {
            "answer": "الرجاء إدخال سؤال." if lang == "ar" else "Please enter a question."
        }

    if not OPENROUTER_API_KEY:
        return {
            "answer": "مفتاح OpenRouter API غير موجود." if lang == "ar" else "OpenRouter API key is missing."
        }

    if not KB:
        return {
            "answer": "قاعدة المعرفة غير متوفرة حاليًا." if lang == "ar" else "Knowledge base is currently unavailable."
        }

    matches = retrieve_chunks(question, top_k=3)

    if not matches:
        return {
            "answer": "هذه المعلومة غير مذكورة في البورتفوليو."
            if lang == "ar"
            else "This information is not mentioned in the portfolio."
        }

    context = build_context(matches, lang)

    try:
        answer = await generate_rag_answer(question, context, lang)

        if lang == "ar":
            answer = clean_arabic_response(answer)
        else:
            answer = clean_english_response(answer)

        return {"answer": answer}

    except Exception as e:
        print("Unexpected error:", str(e))
        return {
            "answer": "تعذر الوصول إلى المساعد حاليًا."
            if lang == "ar"
            else "The assistant is currently unavailable."
        }
