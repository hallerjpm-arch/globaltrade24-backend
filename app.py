from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("question","")
    language = data.get("language","de")

    language_name = {
        "de":"German",
        "fr":"French",
        "es":"Spanish",
        "pl":"Polish"
    }.get(language,"German")

    prompt = f"""
You are the customer service assistant of GlobalTrade24 GmbH.

Rules:
- Answer only in {language_name}
- Be professional
- Be concise
- Help with orders, returns, shipping and products

Customer:
{question}
"""

    try:

        completion = client.chat.completions.create(
            model="google/gemma-2-2b-it",
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

        return jsonify({
            "answer": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
