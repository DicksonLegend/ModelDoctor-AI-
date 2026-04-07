"""
Suggestion Engine — Rule-based core with optional Gemini AI enhancement.
Maps diagnosed problems → actionable improvement suggestions.
"""

from __future__ import annotations

import httpx
from app.config import GEMINI_API_KEY, GEMINI_MODEL


# ── Rule-Based Suggestions ────────────────────────────────────────────

RULES: dict = {
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
            "Add slight regularization, reduce max_depth by 1–2 levels, "
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
}


def get_rule_suggestions(diagnosis: list) -> list:
    """
    Generate suggestions from rule engine based on diagnosis results.
    """
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

    try:
        prompt = _build_prompt(diagnosis, suggestions)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": (
                                        "You are an ML expert. Given model diagnosis and suggestions, "
                                        "rewrite each suggestion's explanation to be clear, concise, "
                                        "and helpful for a data science student. Keep the same structure.\n\n"
                                        + prompt
                                    )
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.5,
                        "maxOutputTokens": 800,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract text from Gemini response
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            enhanced = _parse_ai_response(content, suggestions)
            return enhanced

    except Exception:
        # Fallback to rule-based suggestions on any failure
        return suggestions


def _build_prompt(diagnosis: list, suggestions: list) -> str:
    lines = ["Model Diagnosis Results:"]
    for d in diagnosis:
        lines.append(f"- {d['problem']} ({d['severity']}): {d['reason']}")

    lines.append("\nCurrent Suggestions:")
    for s in suggestions:
        lines.append(f"- Issue: {s['issue']}")
        lines.append(f"  Action: {s['action']}")
        lines.append(f"  Explanation: {s['explanation']}")

    lines.append(
        "\nPlease enhance each explanation to be more helpful and specific. "
        "Return in the same format: Issue | Action | Explanation, one per line."
    )
    return "\n".join(lines)


def _parse_ai_response(content: str, fallback: list) -> list:
    """Best-effort parse of AI text. Falls back if parsing fails."""
    try:
        enhanced = []
        for suggestion in fallback:
            enhanced.append({
                "issue": suggestion["issue"],
                "action": suggestion["action"],
                "explanation": suggestion["explanation"],
            })

        # Try to extract enhanced explanations from Gemini response
        ai_lines = [l.strip() for l in content.split("\n") if l.strip()]
        explanation_lines = [
            l for l in ai_lines
            if l.lower().startswith("explanation:") or l.lower().startswith("- explanation:")
        ]
        for i, exp_line in enumerate(explanation_lines):
            if i < len(enhanced):
                cleaned = exp_line.split(":", 1)[-1].strip().lstrip("- ").strip()
                if len(cleaned) > 20:
                    enhanced[i]["explanation"] = cleaned

        return enhanced
    except Exception:
        return fallback
