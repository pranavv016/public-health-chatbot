import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Disease Awareness Chatbot",
    description="Rule-based symptom matching system",
    version="1.0"
)

# Allow the frontend (GitHub Pages / Netlify / localhost) to call this API.
# Replace "*" with your actual frontend URL once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomRequest(BaseModel):
    symptoms: list[str]


@app.get("/")
def home():
    return {
        "message": "Disease Awareness Chatbot API is running"
    }


@app.post("/check-symptoms")
def check_symptoms(request: SymptomRequest):

    try:
        # Get all symptoms
        symptoms_response = supabase.table("Symptoms").select("*").execute()
        symptoms_data = symptoms_response.data

        # Get disease-symptom relationships
        relations_response = (
            supabase
            .table("Disease_symptoms")
            .select("*")
            .execute()
        )
        relations = relations_response.data

        # Get diseases
        diseases_response = supabase.table("Disease").select("*").execute()
        diseases = diseases_response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    # Convert user symptoms to lowercase
    user_symptoms = {
        symptom.strip().lower()
        for symptom in request.symptoms
    }

    # Match symptom names with IDs
    symptom_ids = set()

    for symptom in symptoms_data:
        if symptom["Name"].lower() in user_symptoms:
            symptom_ids.add(symptom["id"])

    # Calculate disease scores
    disease_scores = {}

    for relation in relations:

        disease_id = relation["disease_id"]
        symptom_id = relation["symptoms_id"]
        importance = relation["importance"] or 1

        if symptom_id in symptom_ids:

            if disease_id not in disease_scores:
                disease_scores[disease_id] = 0

            disease_scores[disease_id] += importance

    # Create results
    results = []

    for disease in diseases:

        disease_id = disease["id"]

        if disease_id in disease_scores:

            results.append({
                "disease": disease["Name"],
                "score": disease_scores[disease_id],
                "description": disease.get("Description"),
                "awareness": disease.get("Awareness"),
                "warning_signs": disease.get("Warning_signs"),
                "prevention": disease.get("Prevention")
            })

    # Highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "message": "These are possible conditions based on the entered symptoms. This is not a confirmed diagnosis.",
        "results": results
    }
