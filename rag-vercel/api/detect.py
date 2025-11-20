# api/detect.py
from flask import Flask, request, jsonify
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------- DeepSeek Config -----------------
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"  # <-- update if your model is in another region
API_KEY = "4_JENf9g9NVi7_332loZt65qIydiAJCPNHhbx0irqaHtJPkfqcUCpp8tp85SlqOU8QX1lYp4AsvLtKqgx0OXRQ"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def deepseek_chat(prompt, system_prompt=None, max_tokens=512, temperature=0.3):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-v3.1",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": "Request failed", "details": str(e)}

    data = response.json()
    try:
        return {"answer": data["choices"][0]["message"]["content"].strip()}
    except (KeyError, IndexError):
        return {"error": "Invalid response format", "details": data}

# ----------------- Symptoms -----------------
SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"},
    {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"},
    {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"},
    {"key": "dizziness", "text": "دوخة"},
    {"key": "Voice quality changes", "text": "تغيرات في جودة الصوت"},
    {"key": "Hoarseness", "text": "بحة الصوت"},
    {"key": "Taste changes", "text": "تغير الطعم"},
    {"key": "Decreased appetite", "text": "انخفاض الشهية"},
    {"key": "Vomiting", "text": "تقيؤ"},
    {"key": "Heartburn", "text": "حرقة صدر"},
    {"key": "Gas", "text": "الغازات"},
    {"key": "Bloating", "text": "الانتفاخ"},
    {"key": "Hiccups", "text": "زغطة"},
    {"key": "Constipation", "text": "امساك"},
    {"key": "Diarrhea", "text": "اسهال"},
    {"key": "Fecal incontinence", "text": "سلس برازي"},
    {"key": "Shortness of breath", "text": "ضيق تنفس"},
]

# ----------------- Embeddings -----------------
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts, convert_to_numpy=True)

def detect_symptoms_embedding(user_text, top_k=3):
    user_embedding = model.encode([user_text], convert_to_numpy=True)[0]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_sim(user_embedding, emb) for emb in symptom_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {"key": SYMPTOMS[i]["key"], "text": SYMPTOMS[i]["text"], "similarity": float(similarities[i])}
        for i in top_indices
    ]

# ----------------- Flask App -----------------
app = Flask(__name__)

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    user_text = data["text"]
    detected_symptoms = detect_symptoms_embedding(user_text)
    deepl_answer = deepseek_chat(user_text)

    return jsonify({
        "input": user_text,
        "detected_symptoms": detected_symptoms,
        "deepseek_answer": deepl_answer
    })

if __name__ == "__main__":
    app.run(debug=True)
