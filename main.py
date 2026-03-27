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
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

KB = []
if KB_PATH.exists():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
else:
    print(f"WARNING: knowledge_base.json not found at {KB_PATH}")

print("MODEL:", MODEL_NAME)
print("KB ITEMS:", len(KB))


class ChatRequest(BaseModel):
    question: str


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }

    for src, target in replacements.items():
        text = text.replace(src, target)

    text = re.sub(r"[^\w\u0600-\u06FF\s\.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_mixed_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.replace(" :", ":")
    return text.strip()


def extract_language_content(content: str, lang: str) -> str:
    content = str(content or "").strip()
    if not content:
        return ""

    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(parts) == 1:
        return parts[0]

    arabic_parts = [p for p in parts if is_arabic(p)]
    english_parts = [p for p in parts if not is_arabic(p)]

    if lang == "ar":
        return "\n\n".join(arabic_parts) if arabic_parts else parts[-1]
    return "\n\n".join(english_parts) if english_parts else parts[0]


def extract_language_title(title: str, lang: str) -> str:
    title = str(title or "").strip()
    if not title:
        return ""

    if " / " in title:
        parts = [p.strip() for p in title.split(" / ") if p.strip()]
        if len(parts) >= 2:
            if lang == "ar":
                for part in parts:
                    if is_arabic(part):
                        return part
                return parts[-1]
            else:
                for part in parts:
                    if not is_arabic(part):
                        return part
                return parts[0]

    return title


def get_custom_answer(question: str, lang: str):
    q = normalize_text(question)

    arabic_greetings = {
        "هاي", "هلا", "هلاا", "مرحبا", "اهلا", "أهلا",
        "السلام", "السلام عليكم", "صباح الخير", "مساء الخير"
    }

    english_greetings = {
        "hi", "hello", "hey", "good morning", "good evening"
    }

    if lang == "ar" and q in {normalize_text(x) for x in arabic_greetings}:
        return "مرحبًا! اسأليني عن مودة، مثل المشاريع، الخبرات، التعليم، المهارات، أو المعدل."

    if lang == "en" and q in {normalize_text(x) for x in english_greetings}:
        return "Hi! Ask me about Mawda’s projects, experience, education, skills, or GPA."

    return None


def expand_query_words(question: str):
    q = normalize_text(question)
    words = [w for w in q.split() if len(w) > 1]

    synonym_map = {
        "موده": ["mawda", "alguraafi", "mawda alguraafi"],
        "مودة": ["mawda", "alguraafi", "mawda alguraafi"],
        "القرافي": ["alguraafi", "mawda alguraafi"],

        "المعدل": ["gpa", "cgpa", "4.88", "first class honors", "honors"],
        "معدل": ["gpa", "cgpa", "4.88", "honors"],
        "جامعه": ["university", "taibah", "taibah university"],
        "جامعة": ["university", "taibah", "taibah university"],
        "طيبه": ["taibah", "taibah university"],
        "طيبة": ["taibah", "taibah university"],
        "الشرف": ["first class honors", "honors"],

        "خبره": ["experience", "work", "responsibilities"],
        "خبرة": ["experience", "work", "responsibilities"],
        "الخبره": ["experience", "work", "responsibilities"],
        "الخبرة": ["experience", "work", "responsibilities"],
        "تشتغل": ["work", "job", "experience", "data analyst", "elm"],
        "تعمل": ["work", "job", "experience", "data analyst", "elm"],
        "وظيفه": ["job", "work", "data analyst"],
        "وظيفة": ["job", "work", "data analyst"],
        "مسؤوليات": ["responsibilities", "tasks"],
        "مهام": ["tasks", "responsibilities"],

        "تدريب": ["internship", "intern", "training", "responsibilities"],
        "التدريب": ["internship", "intern", "training", "responsibilities"],
        "متدربه": ["internship", "intern", "criminal evidence"],
        "متدربة": ["internship", "intern", "criminal evidence"],
        "سوت": ["did", "responsibilities", "tasks", "conducted", "organized", "prepared"],
        "سوى": ["did", "responsibilities", "tasks", "conducted", "organized", "prepared"],
        "وش": ["what", "details"],
        "ماذا": ["what", "details"],
        "تحقيق": ["investigation", "forensic"],
        "جنائي": ["forensic", "criminal evidence"],
        "ادله": ["evidence", "digital evidence"],
        "أدلة": ["evidence", "digital evidence"],

        "تعليم": ["education", "bachelor", "bootcamp", "university"],
        "دراسه": ["education", "bachelor", "bootcamp", "university"],
        "دراسة": ["education", "bachelor", "bootcamp", "university"],
        "مؤهلات": ["education", "bachelor", "bootcamp"],
        "بكالوريوس": ["bachelor", "computer science", "taibah university"],
        "معسكر": ["bootcamp", "tuwaiq academy", "data science", "ai bootcamp"],
        "طويق": ["tuwaiq academy", "bootcamp"],

        "مهارات": ["skills", "technical skills", "tools", "technologies"],
        "المهارات": ["skills", "technical skills", "tools", "technologies"],
        "تقنيه": ["technical", "technologies", "tools"],
        "تقنية": ["technical", "technologies", "tools"],
        "التقنيه": ["technical", "technologies", "tools"],
        "التقنية": ["technical", "technologies", "tools"],
        "تولز": ["tools", "technologies", "used include", "uses"],
        "ادوات": ["tools", "technologies", "uses", "used include"],
        "أدوات": ["tools", "technologies", "uses", "used include"],
        "تستخدم": ["uses", "tools", "technologies"],
        "استخدمت": ["used", "tools", "technologies"],
        "برمجه": ["programming", "python", "java", "sql"],
        "برمجة": ["programming", "python", "java", "sql"],
        "بيانات": ["data analysis", "eda", "tableau", "power bi", "streamlit", "fastapi"],
        "تعلم": ["machine learning", "scikit-learn", "model training", "model evaluation"],
        "تصميم": ["figma", "canva"],
        "تقارير": ["reporting", "webi", "idt"],

        "مشروع": ["project", "projects", "portfolio"],
        "مشاريع": ["project", "projects", "portfolio"],
        "اعمال": ["projects", "portfolio"],
        "أعمال": ["projects", "portfolio"],

        "نباهه": ["nabahah", "laboratory safety", "lab safety", "yolov8", "supabase", "fastapi"],
        "نباهة": ["nabahah", "laboratory safety", "lab safety", "yolov8", "supabase", "fastapi"],
        "interviewsense": ["interview evaluation", "assemblyai", "deepface", "fer", "flask"],
        "اديداس": ["adidas", "sales dashboard", "eda", "streamlit", "plotly"],
        "أديداس": ["adidas", "sales dashboard", "eda", "streamlit", "plotly"],
        "اوبر": ["uber", "booking", "power bi", "excel"],
        "أوبر": ["uber", "booking", "power bi", "excel"],
        "king": ["king county", "housing", "price prediction", "scikit-learn"],
        "county": ["king county", "housing", "price prediction"],
        "السعوديه": ["saudi arabia", "business activities", "power bi"],
        "السعودية": ["saudi arabia", "business activities", "power bi"],
        "تجاره": ["e-commerce", "customer behavior", "excel", "power bi"],
        "تجارة": ["e-commerce", "customer behavior", "excel", "power bi"],
        "ايرادات": ["revenue", "tracking dashboard", "product and region", "power bi"],
        "إيرادات": ["revenue", "tracking dashboard", "product and region", "power bi"],

        "elm": ["data analyst", "tableau", "power bi", "webi", "idt", "sql server"],
        "علم": ["elm", "data analyst", "tableau", "power bi", "webi", "idt", "sql server"],

        "xry": ["xry", "xamn", "md-next", "md-red"],
        "xamn": ["xry", "xamn", "md-next", "md-red"],
        "webi": ["sap web intelligence", "idt", "reporting tools"],
        "idt": ["sap information design tool", "webi", "reporting tools"],
    }

    expanded = set(words)

    for word in words:
        if word in synonym_map:
            expanded.update(normalize_text(x) for x in synonym_map[word])

    joined_q = " ".join(words)

    if "تدريب" in joined_q or "internship" in joined_q:
        expanded.update([
            "internship", "responsibilities", "criminal evidence",
            "xry", "xamn", "md-next", "md-red",
            "digital evidence", "technical reports", "forensic"
        ])

    if "خبر" in joined_q or "experience" in joined_q:
        expanded.update([
            "experience", "responsibilities", "tableau", "power bi",
            "webi", "idt", "sql server", "data analyst"
        ])

    if "مهار" in joined_q or "skills" in joined_q:
        expanded.update([
            "skills", "technical skills", "python", "sql", "java",
            "power bi", "tableau", "streamlit", "fastapi",
            "scikit-learn", "excel", "word", "html", "css",
            "figma", "canva", "webi", "idt"
        ])

    if "معدل" in joined_q or "gpa" in joined_q:
        expanded.update([
            "gpa", "4.88", "first class honors", "taibah university"
        ])

    if "مشاريع" in joined_q or "projects" in joined_q:
        expanded.update(["project", "projects", "portfolio"])

    return list(expanded)


def detect_broad_category(question: str):
    q = normalize_text(question)

    project_terms = [
        "كل المشاريع", "المشاريع", "projects", "all projects",
        "project list", "اعرض المشاريع", "اذكر المشاريع",
        "ما هي المشاريع", "وش مشاريعها", "وش المشاريع"
    ]

    experience_terms = [
        "كل الخبرات", "الخبرات", "الخبره", "الخبرة",
        "experience", "experiences", "اعرض الخبرات",
        "وش خبراتها", "ما هي الخبرات"
    ]

    education_terms = [
        "كل التعليم", "التعليم", "education", "academic background",
        "الدراسه", "الدراسة", "المؤهلات", "المؤهل", "اعرض التعليم"
    ]

    skills_terms = [
        "المهارات", "كل المهارات", "المهارات التقنية", "المهارات التقنيه",
        "technical skills", "skills", "وش مهاراتها", "اذكر المهارات"
    ]

    if any(normalize_text(term) in q for term in project_terms):
        return {"mode": "category", "categories": {"project", "projects"}}

    if any(normalize_text(term) in q for term in experience_terms):
        return {"mode": "category", "categories": {"experience"}}

    if any(normalize_text(term) in q for term in education_terms):
        return {"mode": "category", "categories": {"education"}}

    if any(normalize_text(term) in q for term in skills_terms):
        return {"mode": "category", "categories": {"skills"}}

    return None


def retrieve_by_category(categories):
    wanted = {normalize_text(c) for c in categories}
    results = []

    for item in KB:
        category = normalize_text(item.get("category", ""))
        if category in wanted:
            results.append(item)

    return results


def retrieve_chunks(question: str, top_k: int = 8):
    category_intent = detect_broad_category(question)
    if category_intent:
        matches = retrieve_by_category(category_intent["categories"])
        return matches[:top_k] if top_k else matches

    q = normalize_text(question)
    q_words = expand_query_words(question)

    if not q_words:
        return []

    scored_results = []

    for item in KB:
        item_id = normalize_text(item.get("id", ""))
        title = item.get("title", "")
        content = item.get("content", "")
        category = normalize_text(item.get("category", ""))

        normalized_title = normalize_text(title)
        normalized_content = normalize_text(content)
        full_text = f"{item_id} {normalized_title} {normalized_content}"

        score = 0

        if q and q in full_text:
            score += 20

        for word in q_words:
            if not word:
                continue

            if word == item_id:
                score += 15
            if word in normalized_title:
                score += 10
            if word in normalized_content:
                score += 5

        for token in q.split():
            if token in normalized_title:
                score += 6
            elif token in normalized_content:
                score += 3

        if any(x in q for x in ["تدريب", "internship", "متدرب"]):
            if "internship" in item_id or "intern" in normalized_title or "training" in normalized_content:
                score += 12
            if "responsibilities" in item_id:
                score += 8

        if any(x in q for x in ["خبره", "خبرة", "experience", "work"]):
            if category == "experience":
                score += 10
            if "responsibilities" in item_id:
                score += 6

        if any(x in q for x in ["مهارات", "skills", "تقنيه", "تقنية", "تولز", "ادوات", "أدوات"]):
            if category == "skills":
                score += 12

        if any(x in q for x in ["تستخدم", "استخدمت", "tools", "uses", "technologies"]):
            if any(tool_word in normalized_content for tool_word in [
                "python", "sql", "java", "power bi", "tableau", "streamlit",
                "fastapi", "scikit", "excel", "word", "html", "css",
                "figma", "canva", "webi", "idt", "xry", "xamn", "md-next", "md-red"
            ]):
                score += 10

        if any(x in q for x in ["معدل", "gpa", "honors"]):
            if category == "education":
                score += 10

        if any(x in q for x in ["مشروع", "مشاريع", "project", "projects"]):
            if category in {"project", "projects"}:
                score += 8

        if score > 0:
            scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    seen_ids = set()
    results = []

    for score, item in scored_results:
        item_id = item.get("id", "")
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        results.append(item)
        if len(results) >= top_k:
            break

    return results


def build_context(matches, lang: str = "en") -> str:
    if not matches:
        return ""

    sections = []
    for item in matches:
        title = extract_language_title(item.get("title", ""), lang)
        content = extract_language_content(item.get("content", ""), lang)
        category = item.get("category", "")

        if lang == "ar":
            sections.append(
                f"الفئة: {category}\nالعنوان: {title}\nالمحتوى: {content}"
            )
        else:
            sections.append(
                f"Category: {category}\nTitle: {title}\nContent: {content}"
            )

    return "\n\n---\n\n".join(sections)


def clean_arabic_response(text: str) -> str:
    text = str(text or "").strip()

    english_fallback_patterns = [
        r"(?i)^the context does not provide.*",
        r"(?i)^based on the context.*",
        r"(?i)^this information is not mentioned.*",
        r"(?i)^according to the context.*",
        r"(?i)^i cannot find.*",
        r"(?i)^i could not find.*",
    ]

    for pattern in english_fallback_patterns:
        if re.search(pattern, text):
            return "هذه المعلومة غير مذكورة في البورتفوليو."

    replacements = {
        "This information is not mentioned in the portfolio.": "هذه المعلومة غير مذكورة في البورتفوليو.",
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


async def generate_rag_answer(question: str, context: str, lang: str) -> str:
    if lang == "ar":
        system_prompt = """
أنت مساعد ذكي خاص ببورتفوليو مودة القرافي.

مهمتك:
الإجابة فقط اعتمادًا على المعلومات الموجودة في السياق المسترجع من knowledge base.

قواعد صارمة:
- إذا كان السؤال بالعربية فأجب بالعربية فقط.
- لا تبدأ الإجابة بالإنجليزية.
- استخدم الإنجليزية فقط في أسماء الأدوات والتقنيات مثل Python وSQL وPower BI وTableau وFastAPI.
- لا تضف أي معلومة غير موجودة في السياق.
- لا تخمّن ولا تستنتج معلومات غير مذكورة.
- إذا كان السؤال عن مشروع أو خبرة أو تعليم أو مهارة، فاذكر ما ورد في السياق فقط.
- إذا كان السؤال عن التدريب، فاذكر ما قامت به مودة أثناء التدريب والأدوات التي استخدمتها إذا كانت موجودة في السياق.
- إذا كان السؤال عن الخبرة، فاذكر المسؤوليات والأدوات أو التقنيات المستخدمة إذا كانت موجودة في السياق.
- إذا كان السؤال عن المهارات التقنية، فاجمع كل المهارات التقنية المذكورة في السياق بشكل مرتب وواضح.
- إذا كانت الإجابة موزعة على أكثر من عنصر في السياق، فادمجها في إجابة واحدة طبيعية دون تكرار.
- إذا طلب المستخدم "كل المشاريع" أو "كل الخبرات" أو "كل التعليم" أو "كل المهارات"، اعرض جميع العناصر الموجودة في السياق بشكل مرتب وواضح.
- إذا سأل المستخدم عن ترتيب مثل "أول مشروع" ولم يكن الترتيب مذكورًا في السياق، قل بوضوح إن ترتيب المشاريع غير مذكور في البورتفوليو.
- إذا لم توجد الإجابة في السياق، قل فقط:
هذه المعلومة غير مذكورة في البورتفوليو.

أسلوب الإجابة:
- اجعل الإجابة طبيعية ومرتبة ومهنية.
- يفضّل أن تكون مختصرة وواضحة.
- لا تستخدم مقدمات طويلة.
"""
    else:
        system_prompt = """
You are an intelligent portfolio assistant for Mawda Alguraafi.

Your task:
Answer only using the information found in the retrieved knowledge base context.

Strict rules:
- If the user asks in English, answer only in English.
- Do not use Arabic.
- Do not add any information that is not explicitly in the context.
- Do not guess or infer missing facts.
- If the user asks for projects, experience, education, or skills, provide only what appears in the context.
- If the user asks about the internship, mention what Mawda did during the internship and the tools she used if they appear in the context.
- If the user asks about experience, mention the responsibilities and the tools or technologies used if they appear in the context.
- If the user asks about technical skills, combine all relevant technical skills found in the context into one clear answer.
- If the answer is spread across multiple context items, merge them into one natural response without repetition.
- If the user asks for all projects, all experience, all education, or all skills, list all relevant items from the context clearly.
- If the user asks for an order such as "first project" and no order is given in the context, clearly say that the project order is not specified in the portfolio.
- If the answer is not found, say exactly:
This information is not mentioned in the portfolio.

Style:
- Keep the answer professional, natural, and clear.
- Be concise.
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
        "temperature": 0.2,
        "max_tokens": 500,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
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

    broad_category = detect_broad_category(question)
    if broad_category:
        matches = retrieve_by_category(broad_category["categories"])
    else:
        matches = retrieve_chunks(question, top_k=8)

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

        if not answer.strip():
            answer = (
                "هذه المعلومة غير مذكورة في البورتفوليو."
                if lang == "ar"
                else "This information is not mentioned in the portfolio."
            )

        return {"answer": answer}

    except Exception as e:
        print("Unexpected error:", str(e))
        if lang == "ar":
            return {"answer": "تعذر الوصول إلى المساعد حاليًا. الرجاء المحاولة مرة أخرى."}
        return {"answer": "The assistant is currently unavailable. Please try again."}
