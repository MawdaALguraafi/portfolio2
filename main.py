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
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\u0600-\u06FF\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def fix_mixed_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    text = re.sub(r"\s+و\s+", " و", text)
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
        r"(?i)^i cannot find.*",
    ]

    for pattern in english_fallback_patterns:
        if re.search(pattern, text):
            return "هذه المعلومة غير مذكورة في البورتفوليو."

    replacements = {
        "This information is not mentioned in the portfolio.": "هذه المعلومة غير مذكورة في البورتفوليو.",
        "Mawda works with": "تعمل مودة باستخدام",
        "Mawda has skills in": "تمتلك مودة مهارات في",
        "Mawda's portfolio includes": "يتضمن بورتفوليو مودة",
        "According to the context,": "",
        "Based on the context,": "",
        "according to the context": "",
        "based on the context": "",
    }

    for src, target in replacements.items():
        text = text.replace(src, target)

    text = fix_mixed_text(text)
    text = re.sub(r"\s+", " ", text).strip()

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    english_words = len(re.findall(r"[A-Za-z]{3,}", text))

    if arabic_chars < 3 and english_words > 4:
        return "هذه المعلومة غير مذكورة في البورتفوليو."

    return text


def clean_english_response(text: str) -> str:
    text = str(text or "").strip()

    if "هذه المعلومة غير مذكورة في البورتفوليو" in text:
        return "This information is not mentioned in the portfolio."

    text = re.sub(r"[\u0600-\u06FF]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_custom_answer(question: str, lang: str):
    q = normalize_text(question)

    if lang == "ar":
        if (
            "عرف" in q
            or "عرفيني" in q
            or "عن نفسك" in q
            or "نبذه" in q
            or "نبذة" in q
            or "من هي موده" in q
            or "من هي مودة" in q
        ):
            return (
                "مودة القرافي متخصصة في تحليل البيانات والذكاء الاصطناعي، وتمتلك خبرة عملية في بناء "
                "لوحات المعلومات والتقارير التحليلية وتطوير حلول تقنية تدعم اتخاذ القرار. "
                "تتميز بقدرتها على تحويل البيانات إلى رؤى واضحة ومشاريع عملية باستخدام أدوات مثل "
                "Python وSQL وPower BI وTableau وFastAPI."
            )

        if (
            "مهارات" in q
            or "وش تعرف" in q
            or "قدرات" in q
            or "تقنيات" in q
            or "ادوات" in q
            or "أدوات" in q
            or "skills" in q
        ):
            return (
                "تمتلك مودة مهارات قوية في Python وSQL وExcel وPower BI وTableau وFastAPI وStreamlit، "
                "إلى جانب تنظيف البيانات وتحليلها استكشافيًا وتدريب النماذج وتقييمها. "
                "كما تتميز بقدرتها على بناء لوحات معلومات وتقارير تحليلية تساعد على فهم الأداء "
                "وتحويل البيانات إلى نتائج عملية واضحة."
            )

        if (
            "خبرات" in q
            or "الخبرات" in q
            or "خبره" in q
            or "خبرة" in q
            or "experience" in q
        ):
            return (
                "لدى مودة خبرة عملية في تحليل البيانات وبناء لوحات المعلومات والتقارير التحليلية، "
                "إضافة إلى تطوير مشاريع تقنية مرتبطة بالذكاء الاصطناعي ومعالجة البيانات. "
                "كما عملت على أدوات وتقنيات تدعم التحليل واتخاذ القرار، مع اهتمام واضح بالتطبيق العملي "
                "وجودة المخرجات."
            )

        if (
            "مشاريع" in q
            or "مشروع" in q
            or "اعمالها" in q
            or "أعمالها" in q
            or "projects" in q
        ):
            return (
                "يتضمن بورتفوليو مودة مجموعة من المشاريع التي تعكس قدرتها على الدمج بين التحليل والتقنية، "
                "مثل مشاريع تحليل البيانات، ولوحات المعلومات، والحلول المعتمدة على الذكاء الاصطناعي. "
                "وتركز مشاريعها على تقديم قيمة عملية واضحة، وليس فقط تنفيذ الفكرة من الجانب التقني."
            )

        if (
            "ليش اوظف" in q
            or "ليش نوظف" in q
            or "وش يميز" in q
            or "ما الذي يميز" in q
            or "لماذا موده" in q
            or "لماذا مودة" in q
            or "why hire" in q
        ):
            return (
                "ما يميز مودة هو جمعها بين المهارات التحليلية والتقنية والقدرة على تطبيقها عمليًا في مشاريع "
                "واقعية. فهي لا تكتفي بفهم البيانات، بل تحولها إلى تقارير ولوحات معلومات وحلول ذكية "
                "تدعم القرار وتظهر أثرًا واضحًا، وهذا يجعلها مرشحة قوية لأي جهة تبحث عن شخص يجمع بين "
                "التحليل والدقة والتطبيق العملي."
            )

    else:
        if (
            "tell me about yourself" in q
            or "introduce yourself" in q
            or "about mawda" in q
            or "who is mawda" in q
        ):
            return (
                "Mawda Alguraafi is a data-focused professional with strong interests in data analysis and artificial intelligence. "
                "She has hands-on experience in building dashboards, analytical reports, and practical technical solutions using "
                "tools such as Python, SQL, Power BI, Tableau, and FastAPI."
            )

        if "skills" in q or "technical skills" in q:
            return (
                "Mawda has strong skills in Python, SQL, Excel, Power BI, Tableau, FastAPI, and Streamlit, along with experience "
                "in data cleaning, exploratory data analysis, model training, and evaluation. She is especially strong at turning "
                "data into clear insights, dashboards, and practical decision-support solutions."
            )

        if "experience" in q or "background" in q:
            return (
                "Mawda has practical experience in data analysis, dashboard development, analytical reporting, and AI-related projects. "
                "Her work reflects a strong balance between technical execution, analytical thinking, and real-world application."
            )

        if "projects" in q or "portfolio" in q:
            return (
                "Mawda's portfolio includes projects in data analysis, dashboards, and AI-driven solutions that demonstrate both "
                "technical capability and practical business value. Her projects are focused on delivering clear, usable outcomes "
                "rather than just technical implementation."
            )

        if "why hire" in q or "what makes her stand out" in q:
            return (
                "What makes Mawda stand out is her ability to combine analytical thinking, technical skills, and practical execution. "
                "She does not just work with data technically; she turns it into clear reports, dashboards, and useful solutions, "
                "which makes her a strong candidate for teams looking for real impact."
            )

    return None


async def generate_rag_answer(question: str, context: str, lang: str) -> str:
    if lang == "ar":
        system_prompt = """
أنت مساعد ذكي خاص ببورتفوليو مودة القرافي.

هدفك:
تقديم إجابات احترافية، جذابة، وواضحة تعكس مودة كمرشحة قوية في مجالات تحليل البيانات، الذكاء الاصطناعي، ولوحات المعلومات، مع إبراز قيمتها المهنية بشكل مقنع ومختصر.

قواعد صارمة:
- إذا كان سؤال المستخدم بالعربية، يجب أن تكون الإجابة بالعربية فقط.
- استخدم الإنجليزية فقط في أسماء الأدوات والتقنيات مثل Python وSQL وPower BI وTableau وFastAPI وStreamlit.
- لا تبدأ الجواب بأي جملة إنجليزية.
- لا تنسخ النص من السياق حرفيًا.
- أعد صياغة المعلومات بأسلوب مهني جذاب وواضح.
- لا تضف أي معلومات غير موجودة في السياق.
- إذا لم توجد الإجابة في السياق، قل فقط:
هذه المعلومة غير مذكورة في البورتفوليو.

أسلوب الإجابة:
- اجعل الإجابة طبيعية ومقنعة ومرتبة.
- عند الحديث عن مودة، أبرز نقاط القوة المهنية والمهارات والأثر العملي.
- إذا كان السؤال عن المهارات أو الخبرات أو المشاريع، اجعل الإجابة تبدو قوية ومناسبة لشخص قد يوظفها.
- لا تجعل الإجابة جافة أو مختصرة جدًا إلا إذا كان السؤال مباشرًا جدًا.
- اجعل طول الإجابة من جملتين إلى 4 جمل عند الحاجة.
- استخدم أسلوبًا احترافيًا يبرز قيمة مودة دون مبالغة.
"""
    else:
        system_prompt = """
You are an intelligent portfolio assistant for Mawda Alguraafi.

Your goal:
Provide professional, polished, and appealing answers that present Mawda as a strong candidate in data analysis, AI, dashboards, and technical problem-solving.

Strict rules:
- If the user asks in English, answer only in English.
- Do not use Arabic at all.
- Do not mix languages.
- Do not copy the context word for word.
- Rewrite the information in a polished, professional, and human way.
- Do not add any information not found in the context.
- If the answer is not found, say exactly:
This information is not mentioned in the portfolio.

Style:
- Make the answer clear, professional, and appealing.
- Highlight strengths, practical skills, and impact when relevant.
- If the user asks about skills, experience, or projects, make the answer sound strong and hiring-friendly.
- Keep the answer natural, not robotic.
- Use 2 to 4 sentences when needed.
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
        "temperature": 0.3,
        "max_tokens": 260,
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
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

    custom_answer = get_custom_answer(question, lang)
    if custom_answer:
        return {"answer": custom_answer}

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

        if lang == "ar":
            return {"answer": "تعذر الوصول إلى المساعد حاليًا. الرجاء المحاولة مرة أخرى."}
        return {"answer": "The assistant is currently unavailable. Please try again."}
