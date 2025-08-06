from flask import Flask, request, jsonify
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

# Replace with your own Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyANxhjAdBZkhIbLlImw3DKTt0yUINt-6gQ"

# LangChain Gemini model wrapper
llm = GooglePalm(temperature=0.3)

# Prompt template
template = PromptTemplate.from_template(
    "A patient describes the following symptoms: {symptoms}. Classify them into one of the following hospital wards: General, Emergency, or Mental Health. Just return the category name."
)

@app.route("/classify", methods=["POST"])
def classify_symptoms():
    data = request.json
    symptoms = data.get("symptoms", "")
    prompt = template.format(symptoms=symptoms)

    try:
        result = llm(prompt)
        return jsonify({"ward": result.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
