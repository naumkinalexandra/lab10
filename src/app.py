from pathlib import Path
import io
import logging
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.responses import FileResponse
from sklearn.metrics import roc_auc_score
from src.common import ARTIFACTS_DIR, MODEL_PATH, TARGET_COLUMN, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Mortgage Approval ML Service", version="1.0.0",
              description="Учебный ML-сервис. Не использовать для реальных кредитных решений.")
model: Any = load_model()


def feature_columns(loaded):
    if hasattr(loaded, "feature_names_in_"):
        return list(loaded.feature_names_in_)
    pre = getattr(loaded, "named_steps", {}).get("preprocessor")
    return list(pre.feature_names_in_) if pre is not None and hasattr(pre, "feature_names_in_") else []


def ensure_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Сначала обучите модель или вызовите /upload-model.")


def probabilities(loaded, frame):
    if not hasattr(loaded, "predict_proba"):
        return None
    probs = loaded.predict_proba(frame)
    classes = list(getattr(loaded, "classes_", []))
    if not classes and hasattr(loaded, "named_steps"):
        classes = list(loaded.named_steps["classifier"].classes_)
    return probs[:, classes.index(1)] if 1 in classes else probs[:, -1]


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/schema")
def schema():
    ensure_model()
    return {"features": feature_columns(model), "target": TARGET_COLUMN}


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global model
    if not file.filename or Path(file.filename).suffix.lower() not in {".pkl", ".pickle"}:
        raise HTTPException(status_code=400, detail="Загрузите файл .pkl или .pickle.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пуст.")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ARTIFACTS_DIR / "uploaded_model.tmp"
    try:
        tmp.write_bytes(content)
        candidate = joblib.load(tmp)
        if not callable(getattr(candidate, "predict", None)):
            raise ValueError("У объекта нет метода predict().")
        joblib.dump(candidate, MODEL_PATH)
        model = candidate
        logger.info("Модель успешно загружена")
        return {"status": "success", "message": "Модель загружена и сохранена."}
    except Exception as exc:
        logger.exception("Ошибка загрузки модели")
        raise HTTPException(status_code=400, detail=f"Не удалось загрузить модель: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/predict")
async def predict(payload: Any = Body(...)):
    ensure_model()
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
        records = payload
    else:
        raise HTTPException(status_code=422, detail="Передайте JSON-объект или непустой массив объектов.")
    frame = pd.DataFrame(records)
    expected = feature_columns(model)
    if expected:
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Не хватает признаков: {missing}")
        frame = frame[expected]
    try:
        preds = model.predict(frame)
        probs = probabilities(model, frame)
        output = []
        for i, record in enumerate(records):
            row = dict(record)
            row["predicted_loan_status"] = "одобрено" if int(preds[i]) == 1 else "отказ"
            if probs is not None:
                row["approval_probability"] = round(float(probs[i]), 6)
            output.append(row)
        return {"predictions": output}
    except Exception as exc:
        logger.exception("Ошибка предсказания")
        raise HTTPException(status_code=422, detail=f"Не удалось обработать признаки: {exc}") from exc


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    ensure_model()
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Загрузите CSV-файл.")
    try:
        frame = pd.read_csv(io.BytesIO(await file.read()))
        if frame.empty:
            raise HTTPException(status_code=400, detail="CSV-файл пуст.")
        truth = None
        if TARGET_COLUMN in frame.columns:
            from src.train import normalize_target
            truth = normalize_target(frame[TARGET_COLUMN])
            frame = frame.drop(columns=[TARGET_COLUMN])
        expected = feature_columns(model)
        if expected:
            missing = [c for c in expected if c not in frame.columns]
            if missing:
                raise HTTPException(status_code=422, detail=f"В CSV отсутствуют признаки: {missing}")
            frame = frame[expected]
        preds = model.predict(frame)
        probs = probabilities(model, frame)
        output = frame.copy()
        output["predicted_loan_status"] = ["одобрено" if int(v) == 1 else "отказ" for v in preds]
        if probs is not None:
            output["approval_probability"] = probs
        auc = float(roc_auc_score(truth, probs)) if truth is not None and probs is not None and truth.nunique() == 2 else None
        return {"rows": output.to_dict(orient="records"), "row_count": len(output), "roc_auc": auc,
                "note": "ROC-AUC доступен, если CSV содержит loan_status и оба класса."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ошибка обработки CSV")
        raise HTTPException(status_code=400, detail=f"Не удалось обработать CSV: {exc}") from exc
