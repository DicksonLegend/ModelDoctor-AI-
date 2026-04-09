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
    "Low Explained Variance": {
        "action": "Improve regression feature set",
        "explanation": (
            "Add stronger predictive features, remove noisy columns, and test "
            "non-linear regressors. Low explained variance means the model is "
            "not capturing enough target behavior."
        ),
    },
    "High Prediction Error": {
        "action": "Reduce error with robust regression tuning",
        "explanation": (
            "Tune tree depth/estimators or regularization strength, and check "
            "for outliers in the target. Also compare MAE and RMSE to understand "
            "average versus large-error behavior."
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


async def build_plain_language_summary(
    metrics: dict,
    diagnosis: list,
    suggestions: list,
    health: dict,
    class_distribution: dict | None = None,
) -> str:
    """
    Build a detailed, non-technical explanation for end users.
    The summary never changes metric values; it only explains them.
    """
    fallback = _build_plain_language_fallback(
        metrics, diagnosis, suggestions, health, class_distribution
    )

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return fallback

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )

        prompt = (
            "You are writing an easy-to-understand ML report for a non-technical user. "
            "Use ONLY the numbers given below exactly as-is. Do not create new metrics, "
            "do not change values, and do not contradict the diagnosis. "
            "Write 3 short sections with clear headings: "
            "1) What this score means, 2) What is going wrong, 3) What to do next. "
            "Keep language simple and practical. Avoid jargon.\n\n"
            "Health score:\n"
            f"- score: {health.get('score')}\n"
            f"- status: {health.get('status')}\n"
            f"- breakdown: {json.dumps(health.get('breakdown', {}))}\n\n"
            "Metrics:\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "Diagnosis:\n"
            f"{json.dumps(diagnosis, indent=2)}\n\n"
            "Suggestions:\n"
            f"{json.dumps(suggestions, indent=2)}\n\n"
            "Class distribution:\n"
            f"{json.dumps(class_distribution or {}, indent=2)}\n"
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 900,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return fallback

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return fallback

        text = (parts[0].get("text") or "").strip()
        return text if text else fallback
    except Exception as e:
        print(f"Gemini summary error: {e}")
        return fallback


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


def _build_plain_language_fallback(
    metrics: dict,
    diagnosis: list,
    suggestions: list,
    health: dict,
    class_distribution: dict | None,
) -> str:
    """Deterministic plain-language explanation when AI is unavailable."""
    task_type = metrics.get("task_type", "classification")
    acc = float(metrics.get("accuracy", 0.0)) * 100
    f1 = float(metrics.get("f1_score", 0.0)) * 100
    r2 = metrics.get("r2_score")
    rmse = metrics.get("rmse")
    train_acc = metrics.get("train_accuracy")
    gap = None
    if train_acc is not None:
        gap = abs(float(train_acc) - float(metrics.get("accuracy", 0.0))) * 100

    top_issue = diagnosis[0]["problem"] if diagnosis else "No major issues detected"
    top_reason = diagnosis[0]["reason"] if diagnosis else "The model is in acceptable condition."

    next_step = (
        f"Start with: {suggestions[0]['action']}"
        if suggestions else
        "Start by collecting more representative training data."
    )

    class_note = ""
    if class_distribution:
        try:
            counts = sorted(class_distribution.values())
            if counts and counts[0] > 0:
                ratio = round(counts[-1] / counts[0], 2)
                class_note = (
                    f"Class balance is uneven (largest class is about {ratio}x the smallest), "
                    "which can make the model unfair across categories."
                )
        except Exception:
            class_note = ""

    summary = [
        "### What this score means",
        (
            f"Your model health score is {health.get('score', 'N/A')} ({health.get('status', 'N/A')}). "
            f"On this test, accuracy is {acc:.1f}% and F1 score is {f1:.1f}%."
            if task_type != "regression"
            else f"Your model health score is {health.get('score', 'N/A')} ({health.get('status', 'N/A')}). "
                 f"On this test, R2 score is {r2 if r2 is not None else 'N/A'} and RMSE is {rmse if rmse is not None else 'N/A'}."
        ),
        "",
        "### What is going wrong",
        top_reason,
    ]

    if gap is not None:
        summary.append(
            f"The gap between train and test performance is {gap:.1f}%, which indicates how well the model generalizes."
        )
    if class_note:
        summary.append(class_note)

    summary.extend([
        "",
        "### What to do next",
        f"Main issue detected: {top_issue}. {next_step}",
        "After retraining, compare health score and F1 score together, not accuracy alone.",
        "All metric values above are computed directly by the evaluation code, not generated by AI.",
    ])

    return "\n".join(summary)


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
