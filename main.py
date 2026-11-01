import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Use Colab's dataset folder during development.
# On Render, use the data folder included with this project.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    Path("/content/wellcare_data")
    if Path("/content/wellcare_data").exists()
    else BASE_DIR / "data"
)
DATA_DIR = Path(os.getenv("WELLCARE_DATA_DIR", str(DEFAULT_DATA_DIR)))

app = FastAPI(
    title="WellCare AI API",
    description="Educational health-awareness assistant using public datasets",
    version="2.0.0",
)

# Allow the GitHub Pages frontend and optional local development origins.
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://pranavv016.github.io,http://localhost:8000,http://127.0.0.1:5500"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SymptomRequest(BaseModel):
    # Kept compatible with the current index.html, which sends a list.
    symptoms: list[str] = Field(default_factory=list)
    # Optional full text can be sent by a future frontend version.
    text: Optional[str] = None


def find_csv(filename: str) -> Path:
    """Find a dataset CSV even if KaggleHub created an extra nested folder."""
    matches = list(DATA_DIR.rglob(filename)) if DATA_DIR.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {DATA_DIR}. "
            "Check that both Kaggle datasets have been downloaded."
        )
    return matches[0]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;!?\t\n")


def normalize_disease(name: str) -> str:
    name = clean_text(name)
    aliases = {
        "dimorphic hemmorhoids(piles)": "dimorphic hemorrhoids",
        "dimorphic hemorrhoids": "dimorphic hemorrhoids",
        "gerd": "gastroesophageal reflux disease",
        "gastroesophageal reflux disease": "gastroesophageal reflux disease",
        "peptic ulcer diseae": "peptic ulcer disease",
        "peptic ulcer disease": "peptic ulcer disease",
    }
    return aliases.get(name, name)


# -----------------------------------------------------------------------------
# Load and prepare the five CSV files once when the API starts.
# -----------------------------------------------------------------------------
try:
    disease_data = pd.read_csv(find_csv("dataset.csv"))
    disease_description = pd.read_csv(find_csv("symptom_Description.csv"))
    symptom_severity = pd.read_csv(find_csv("Symptom-severity.csv"))
    symptom_precaution = pd.read_csv(find_csv("symptom_precaution.csv"))
    symptom2disease = pd.read_csv(find_csv("Symptom2Disease.csv"))

    disease_data["Disease_clean"] = disease_data["Disease"].apply(normalize_disease)

    symptom_columns = [c for c in disease_data.columns if c.lower().startswith("symptom_")]
    disease_symptoms: dict[str, set[str]] = {}
    for _, row in disease_data.iterrows():
        disease = row["Disease_clean"]
        values = {clean_text(row[col]) for col in symptom_columns}
        values.discard("")
        disease_symptoms.setdefault(disease, set()).update(values)

    severity_weights = {}
    for _, row in symptom_severity.iterrows():
        symptom = clean_text(row.get("Symptom", ""))
        if symptom:
            try:
                severity_weights[symptom] = float(row.get("weight", 1) or 1)
            except (TypeError, ValueError):
                severity_weights[symptom] = 1.0

    descriptions = {}
    for _, row in disease_description.iterrows():
        descriptions[normalize_disease(row.get("Disease", ""))] = str(row.get("Description", "") or "")

    precautions = {}
    precaution_columns = [c for c in symptom_precaution.columns if c.lower().startswith("precaution_")]
    for _, row in symptom_precaution.iterrows():
        disease = normalize_disease(row.get("Disease", ""))
        items = [str(row[col]).strip() for col in precaution_columns
                 if pd.notna(row.get(col)) and str(row[col]).strip()]
        precautions[disease] = items

    # Prepare Dataset 2 for nearest-text-example retrieval.
    if "label" not in symptom2disease.columns or "text" not in symptom2disease.columns:
        raise ValueError("Symptom2Disease.csv must contain 'label' and 'text' columns.")

    symptom2disease = symptom2disease.dropna(subset=["label", "text"]).copy()
    symptom2disease["label_clean"] = symptom2disease["label"].apply(normalize_disease)
    symptom2disease["text_clean"] = symptom2disease["text"].astype(str).str.strip()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    text_matrix = tfidf.fit_transform(symptom2disease["text_clean"])

    # All known symptom names are used to find symptoms inside natural sentences.
    known_symptoms = sorted(
        {s for symptoms in disease_symptoms.values() for s in symptoms},
        key=len,
        reverse=True,
    )

except Exception as exc:
    raise RuntimeError(
        f"WellCare AI could not load its datasets: {exc}. "
        f"Expected CSV files somewhere inside {DATA_DIR}."
    ) from exc


# -----------------------------------------------------------------------------
# Matching helpers
# -----------------------------------------------------------------------------
def extract_symptoms(user_text: str) -> set[str]:
    """Find dataset symptom phrases in the user's sentence."""
    text = clean_text(user_text)
    found = set()
    for symptom in known_symptoms:
        # Match complete phrases, not partial words.
        pattern = r"(?<![a-z0-9])" + re.escape(symptom) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.add(symptom)
    return found


def get_text_matches(user_text: str, limit: int = 3) -> list[dict]:
    """Return closest Dataset 2 text examples; scores are similarity, not risk."""
    query = tfidf.transform([user_text])
    similarities = cosine_similarity(query, text_matrix).ravel()
    if similarities.size == 0:
        return []

    # Keep the best example for each normalized disease label.
    best_by_disease = {}
    for index, similarity in enumerate(similarities):
        label = symptom2disease.iloc[index]["label_clean"]
        if label not in best_by_disease or similarity > best_by_disease[label]["similarity"]:
            best_by_disease[label] = {
                "disease": label,
                "similarity": float(similarity),
                "matched_example": symptom2disease.iloc[index]["text_clean"],
            }

    ranked = sorted(best_by_disease.values(), key=lambda x: x["similarity"], reverse=True)
    return [item for item in ranked if item["similarity"] > 0][:limit]


def disease_details(disease: str) -> dict:
    return {
        "description": descriptions.get(disease, ""),
        "precautions": precautions.get(disease, []),
    }


# -----------------------------------------------------------------------------
# API routes
# -----------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "WellCare AI API is running",
        "status": "ok",
        "dataset_diseases": len(disease_symptoms),
        "known_symptoms": len(known_symptoms),
        "text_examples": len(symptom2disease),
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "datasets_loaded": True}


@app.post("/check-symptoms")
def check_symptoms(request: SymptomRequest):
    # Support both the current frontend's list and an optional full-text field.
    parts = [str(item) for item in request.symptoms if str(item).strip()]
    if request.text and request.text.strip():
        parts.append(request.text.strip())
    user_text = " ".join(parts).strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Please enter a symptom or describe your concern.")

    matched_symptoms = extract_symptoms(user_text)
    structured_results = []

    for disease, symptoms in disease_symptoms.items():
        matched = sorted(symptoms.intersection(matched_symptoms))
        if not matched:
            continue

        total_weight = sum(severity_weights.get(s, 1.0) for s in symptoms) or 1.0
        matched_weight = sum(severity_weights.get(s, 1.0) for s in matched)
        coverage = round((matched_weight / total_weight) * 100, 1)
        details = disease_details(disease)

        structured_results.append({
            "disease": disease.title(),
            "score": coverage,  # Dataset coverage only; not a probability.
            "match_type": "structured_symptom_match",
            "matched_symptoms": matched,
            "matched_symptom_count": len(matched),
            "description": details["description"],
            "precautions": details["precautions"],
        })

    structured_results.sort(
        key=lambda item: (item["matched_symptom_count"], item["score"]),
        reverse=True,
    )

    text_matches = get_text_matches(user_text)

    # If no structured symptoms were recognized, show text similarity matches.
    # Otherwise, attach the nearest text similarity as a separate field when available.
    text_score_by_disease = {item["disease"]: item for item in text_matches}
    for result in structured_results:
        key = normalize_disease(result["disease"])
        if key in text_score_by_disease:
            result["text_similarity"] = round(
                text_score_by_disease[key]["similarity"] * 100, 1
            )

    if structured_results:
        results = structured_results[:10]
        method = "structured symptom matching"
    else:
        results = []
        for item in text_matches:
            details = disease_details(item["disease"])
            results.append({
                "disease": item["disease"].title(),
                "score": round(item["similarity"] * 100, 1),
                "match_type": "text_similarity_only",
                "matched_symptoms": [],
                "description": details["description"],
                "precautions": details["precautions"],
                "matched_example": item["matched_example"],
            })
        method = "text similarity only"

    return {
        "message": (
            "These are educational matches from the supplied datasets, not a medical diagnosis. "
            "Scores describe dataset matching or text similarity; they are not disease probabilities. "
            "A qualified healthcare professional must assess health concerns."
        ),
        "method": method,
        "recognized_symptoms": sorted(matched_symptoms),
        "results": results,
    }
