# api/detect.py
from flask import Flask, request, jsonify
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------- DeepSeek Config -----------------
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
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

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        return {"error": f"Request failed with status {response.status_code}", "details": response.text}

    data = response.json()
    return {"answer": data["choices"][0]["message"]["content"].strip()}

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
symptom_embeddings = model.encode(symptom_texts)

def detect_symptoms_embedding(user_text, top_k=3):
    user_embedding = model.encode([user_text])[0]
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    similarities = [cosine_sim(user_embedding, emb) for emb in symptom_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    detected = [{"key": SYMPTOMS[i]["key"], "text": SYMPTOMS[i]["text"], "similarity": float(similarities[i])} for i in top_indices]
    return detected

# ----------------- Flask App -----------------
app = Flask(__name__)

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    user_text = data["text"]
    detected_symptoms = detect_symptoms_embedding(user_text)

    # Example: optionally call DeepSeek for explanation
    deepl_answer = deepseek_chat(user_text)

    return jsonify({
        "input": user_text,
        "detected_symptoms": detected_symptoms,
        "deepseek_answer": deepl_answer
    })

# For local testing
if __name__ == "__main__":
    app.run(debug=True)
