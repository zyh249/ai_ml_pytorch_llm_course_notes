"""文本分类服务模板：输入校验、健康检查、模型单例与版本字段。"""
from __future__ import annotations

from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_VERSION = "news-classifier-demo-v1"


def predict_text(text: str) -> tuple[str, float]:
    # 替换为真实模型调用；模型应在进程启动时加载一次，而不是每个请求加载。
    if any(word in text for word in ("股票", "A股", "基金")):
        return "stocks", 0.91
    if any(word in text for word in ("大学", "考研", "招生")):
        return "education", 0.88
    return "reject", 0.51


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})


@app.post("/v1/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "text must be a non-empty string"}), 400
    if len(text) > 500:
        return jsonify({"error": "text is too long"}), 413
    label, score = predict_text(text.strip())
    return jsonify({
        "text": text.strip(),
        "label": label,
        "score": score,
        "model_version": MODEL_VERSION,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=False)
