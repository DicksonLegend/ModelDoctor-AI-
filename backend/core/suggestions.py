"""
Suggestion Engine — Rule-based core with Gemini AI enhancement.
Maps diagnosed problems → actionable improvement suggestions.
"""

from __future__ import annotations

import json
import httpx
from app.config import GEMINI_API_KEY, GEMINI_MODEL


# ── Rule-Based Suggestions ────────────────────────────────────────────

RULES = {
    "Overfitting": {
        "action": "Reduce model complexity",
        "explanation": (
            "Try reducing max_depth for tree-based models, increasing "
            "min_samples_split, or using regularization. Cross-validation "
            "can also help assess generalization."
        ),
    },
    "Mild Overfitting": {
        "action": "Apply light regularization",
        "explanation": (
            "Add slight regularization, reduce max_depth by 1-2 levels, "
            "or increase min_samples_leaf. The gap is moderate so small "
            "adjustments should help."
        ),
    },
    "Underfitting": {
        "action": "Increase model complexity",
        "explanation": (
            "Try increasing n_estimators for Random Forest, adding polynomial "
            "features, or using a more complex model. Also check if important "
            "features are included in the dataset."
        ),
    },
    "Low Accuracy": {
        "action": "Improve feature engineering",
        "explanation": (
            "Consider creating new features, removing noisy ones, or trying "
            "different preprocessing pipelines. Feature selection methods "
            "like SelectKBest can help identify the most informative features."
        ),
    },
    "Severe Class Imbalance": {
        "action": "Apply SMOTE and class weights",
        "explanation": (
            "Use SMOTE (Synthetic Minority Oversampling) to balance your dataset. "
            "Also set class_weight='balanced' in your model. "
            "Consider evaluation with F1-score instead of accuracy."
        ),
    },
    "Class Imbalance": {
        "action": "Apply resampling or class weights",
        "explanation": (
            "Set class_weight='balanced' in your model or apply SMOTE. "
            "This will help the model pay equal attention to all classes."
        ),
    },
    "High Missing Values": {
        "action": "Impute or remove high-null columns",
        "explanation": (
            "For columns with >50% missing, consider dropping them. "
            "For others, use median imputation for numeric and mode for "
            "categorical features."
        ),
    },
    "Missing Values Present": {
        "action": "Apply imputation",
        "explanation": (
            "Fill missing values using median (numeric) or mode (categorical). "
            "Scikit-learn's SimpleImputer makes this easy."
        ),
    },
    "Low Variance Features": {
        "action": "Remove low-variance features",
        "explanation": (
            "Features with near-zero variance don't help the model. "
            "Remove them to reduce noise and speed up training."
        ),
    },
    "Possible Data Leakage": {
        "action": "Investigate data pipeline",
        "explanation": (
            "Check if test data is leaking into training. Common causes: "
            "target variable encoded in features, or temporal data not "
            "split correctly."
        ),
    },
    "Feature Scaling Needed": {
        "action": "Apply StandardScaler",
        "explanation": (
            "Normalize features to the same scale using StandardScaler or "
            "MinMaxScaler. This is especially important for distance-based "
            "models and can also help tree-based models converge faster."
        ),
    },
    "Feature Mismatch": {
        "action": "Retrain model on current data",
        "explanation": (
            "The uploaded model was trained on a different feature set. "
            "A fresh model has been auto-trained on your current dataset. "
            "Use the Retrain feature to try additional configurations."
        ),
    },
}


def get_rule_suggestions(diagnosis: list) -> list:
    """Generate suggestions from rule engine based on diagnosis results."""
    suggestions = []
    for item in diagnosis:
        problem = item["problem"]
        if problem in RULES:
            suggestions.append({
                "issue": problem,
                "action": RULES[problem]["action"],
                "explanation": RULES[problem]["explanation"],
            })
    return suggestions


async def enhance_with_ai(diagnosis: list, suggestions: list) -> list:
    """
    Enhance suggestions using Google Gemini API for human-readable explanations.
    Falls back to rule-based suggestions if API is unavailable.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return suggestions

    if not suggestions:
        return suggestions

    try:
        prompt = _build_prompt(diagnosis, suggestions)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 1024,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                return suggestions

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return suggestions

            ai_text = parts[0].get("text", "")
            if not ai_text:
                return suggestions

            # Try to parse as JSON first
            enhanced = _parse_gemini_json(ai_text, suggestions)
            if enhanced:
                return enhanced

            # Fallback: extract explanations from plain text
            enhanced = _parse_gemini_text(ai_text, suggestions)
            return enhanced

    except Exception as e:
        print(f"Gemini API error: {e}")
        return suggestions


def _build_prompt(diagnosis: list, suggestions: list) -> str:
    """Build a structured prompt for Gemini."""
    prompt = (
        "You are an expert ML engineer. A user's model has been diagnosed with issues. "
        "For each suggestion below, rewrite ONLY the explanation to be more specific, "
        "helpful, and student-friendly. Keep the issue and action the same.\n\n"
        "Return your response as a JSON array with objects containing: "
        '"issue", "action", "explanation"\n\n'
        "Diagnosis:\n"
    )
    for d in diagnosis:
        prompt += f"- {d['problem']} ({d['severity']}): {d['reason']}\n"

    prompt += "\nSuggestions to enhance:\n"
    prompt += json.dumps(suggestions, indent=2)

    return prompt


def _parse_gemini_json(ai_text: str, fallback: list) -> list:
    """Try to parse Gemini response as JSON array."""
    try:
        # Clean up markdown code fences if present
        text = ai_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            result = []
            for i, item in enumerate(parsed):
                if isinstance(item, dict) and "explanation" in item:
                    result.append({
                        "issue": item.get("issue", fallback[i]["issue"] if i < len(fallback) else "Unknown"),
                        "action": item.get("action", fallback[i]["action"] if i < len(fallback) else ""),
                        "explanation": item["explanation"],
                    })
            if result:
                return result
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return []


def _parse_gemini_text(ai_text: str, fallback: list) -> list:
    """Fallback: extract enhanced explanations from plain text response."""
    enhanced = [s.copy() for s in fallback]

    # Try to find explanation-like text for each suggestion
    lines = ai_text.split("\n")
    explanation_texts = []
    current_exp = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("explanation:"):
            if current_exp:
                explanation_texts.append(" ".join(current_exp))
            current_exp = [stripped.split(":", 1)[1].strip()]
        elif current_exp and stripped and not stripped.startswith(("-", "*", "Issue:", "Action:")):
            current_exp.append(stripped)

    if current_exp:
        explanation_texts.append(" ".join(current_exp))

    for i, exp in enumerate(explanation_texts):
        if i < len(enhanced) and len(exp) > 20:
            enhanced[i]["explanation"] = exp

    return enhanced
