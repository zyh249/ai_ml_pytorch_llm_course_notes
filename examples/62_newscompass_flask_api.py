"""生产友好的文本分类 Flask API 骨架：模型只加载一次，并校验输入。"""
from __future__ import annotations

from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)
model = joblib.load("newscompass_rf.joblib")  # 进程启动时加载一次


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="text 必须是非空字符串"), 400
    if len(text) > 2000:
        return jsonify(error="text 过长"), 413
    label = model.predict([text])[0]
    return jsonify(text=text, pred_class=str(label))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=False)
