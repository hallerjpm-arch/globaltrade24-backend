from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Definimos el modelo directamente en el cliente o en la llamada
client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=os.getenv("HF_TOKEN")
)

@app.route("/")
def home():
    return "GlobalTrade24 Backend Running"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}

    question = data.get("question", "")
    language = data.get("language", "de")

    language_name = {
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pl": "Polish"
    }.get(language, "German")

    prompt = f"""You are the customer service assistant of GlobalTrade24 GmbH.

Rules:
- Answer only in {language_name}
- Be professional
- Be concise
- Help with:
  - orders
  - shipping
  - delivery times
  - customs
  - products
  - returns
- If information is unknown, recommend contacting a human employee.

Customer:
{question}
"""

    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )

        answer = completion.choices[0].message.content

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "answer": f"AI Error: {str(e)}"
        }), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
