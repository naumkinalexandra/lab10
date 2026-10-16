# Лабораторная работа №10 — ML-сервис одобрения ипотеки

Учебный проект для 2-го курса: pandas, scikit-learn, FastAPI, HTML/JS, pytest.

> Это демонстрационный прототип, не финансовый инструмент. Не используйте его для реальных кредитных решений.

## Датасет
Скачайте CSV по ссылке преподавателя и положите в `data/loan_data.csv`. Исходный файл не включён в архив. Ожидается целевой столбец `loan_status`; если значения цели отличаются от `Approved/Rejected`, `Y/N`, `1/0`, настройте `normalize_target()` в `src/train.py`.

## Установка и запуск
Рекомендуется Python 3.11.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data data/loan_data.csv
uvicorn src.app:app --reload
```
Откройте `http://127.0.0.1:8000/` или Swagger `http://127.0.0.1:8000/docs`.

## API
- `POST /upload-model`: загрузка доверенной модели `.pkl` через multipart поле `file`. **Pickle может выполнять произвольный код — не загружайте недоверенные файлы и не открывайте endpoint публично.**
- `POST /predict`: JSON-объект или массив объектов. Признаки должны совпадать с обучающими; схема доступна через `GET /schema`. Ответ содержит входные поля, `predicted_loan_status` и вероятность, если модель поддерживает `predict_proba`. При отсутствии модели — HTTP 503.
- `POST /predict-from-csv`: CSV через multipart поле `file`. Возвращает полный набор строк с прогнозами; если есть `loan_status` и оба класса, дополнительно считает ROC-AUC.
- Дополнительно: `GET /health`, `GET /schema`.

## ML-этапы
Удаление дубликатов и строк без целевой метки; импутация числовых пропусков медианой, категориальных — модой; масштабирование числовых признаков; One-Hot Encoding категорий; отбор признаков `SelectFromModel`; разделение train/test; сравнение логистической регрессии и случайного леса по ROC-AUC; сериализация лучшего pipeline в `artifacts/model.pkl`, метрики — `artifacts/metrics.json`.

## Проверки
```bash
pytest -q
ruff check .
```
Docker:
```bash
docker build -t mortgage-ml .
docker run --rm -p 8000:8000 mortgage-ml
```

## Структура
- `src/train.py` — обучение, предобработка, сравнение моделей.
- `src/app.py` — FastAPI.
- `frontend/index.html` — интерфейс.
- `tests/` — тесты.
- `prompts/` — промпты для вайбкодинга.
- `.sourcecraft/ci.yaml` — пример CI; синтаксис нужно сверить с настройками Sourcecraft в вашем курсе.

## Роли в команде
ML Engineer — анализ датасета, обучение, ROC-AUC; Backend Developer — API, валидация, обработка ошибок; Frontend Developer — интерфейс и интеграция. Для команды из двух человек frontend можно объединить с backend.

## Что добавить в отчёт
1. Размер датасета и баланс классов.
2. Обоснование предобработки.
3. Таблицу ROC-AUC двух моделей и объяснение выбора.
4. Описание API и скриншоты.
5. Ограничения модели и вывод.
Не придумывайте метрики: используйте реальные значения `artifacts/metrics.json` после обучения.

DAST/DATS: реализованы разделение модулей, понятные имена, базовая типизация, обработка ошибок, тесты и линтинг. Уточните у преподавателя дополнительные трактовки DAST/DATS.
