# 🚦 TrafficLens AI
### AI-powered CCTV traffic intelligence for Bengaluru

Built for **Gridlock Hackathon 2.0** by Flipkart — a real-time traffic analysis system that processes CCTV footage to classify congestion levels, detect helmet violations, and visualise patterns on a live Bengaluru city map.

---

## What it does

TrafficLens AI takes a traffic video as input and runs a full dual-model AI pipeline on it:

- **Congestion classification** — counts vehicles per frame using YOLOv8 and classifies each frame as Low / Medium / High congestion with temporal smoothing
- **Helmet violation detection** — detects motorcycle riders without helmets using a custom-trained YOLOv8 model (mAP50: 0.89)
- **Live dashboard** — visualises results on an interactive Bengaluru junction map, vehicle count timeline, violation log, and analytics charts

---

## Model performance

| Metric | Score |
|--------|-------|
| mAP50 (overall) | **0.892** |
| mAP50-95 | 0.649 |
| Precision | **0.905** |
| Recall | 0.827 |
| mAP50 — With Helmet | 0.943 |
| mAP50 — Without Helmet | 0.841 |

Trained on 2,042 images using YOLOv8s on Google Colab T4 GPU, 50 epochs.

---

## Architecture

```
Video upload (Streamlit)
        ↓
FastAPI backend — queues job instantly
        ↓
Redis message broker
        ↓
Celery worker — runs full ML pipeline
    ├── YOLOv8 vehicle detector (COCO pretrained)
    │       ↓ vehicle count per frame
    ├── Congestion classifier (Low / Medium / High)
    │       ↓ temporal smoothing over 5-frame buffer
    └── Helmet violation detector (custom trained)
            ↓ "Without Helmet" detections → violation events
        ↓
PostgreSQL — stores detections + violations
        ↓
Streamlit dashboard — charts, map, violation log
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| ML models | YOLOv8 (Ultralytics) |
| Video processing | OpenCV |
| Backend API | FastAPI + Uvicorn |
| Task queue | Celery + Redis |
| Database | PostgreSQL + SQLAlchemy |
| Frontend | Streamlit + Plotly |
| Containerisation | Docker Compose |

---

## Project structure

```
trafficlens-ai/
├── api/                  ← FastAPI backend (6 endpoints)
│   ├── main.py
│   └── schemas.py
├── database/             ← SQLAlchemy models + CRUD
│   ├── models.py
│   ├── database.py
│   └── crud.py
├── worker/               ← Celery ML pipeline
│   └── tasks.py
├── dashboard/            ← Streamlit frontend (5 pages)
│   ├── app.py
│   ├── pages/
│   └── utils/
├── models/               ← trained YOLO weights (.pt)
├── docker-compose.yml
└── requirements.txt
```

---

## Running locally

**Prerequisites:** Python 3.10+, Docker Desktop

**1. Clone and install:**
```bash
git clone https://github.com/ishika345/trafficlens-ai.git
cd trafficlens-ai
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Set up environment:**
```bash
# Create .env file with:
DATABASE_URL=postgresql://postgres:password@localhost:5432/trafficlens
REDIS_URL=redis://localhost:6379/0
VEHICLE_MODEL_PATH=yolov8n.pt
HELMET_MODEL_PATH=models/helmet_yolov8.pt
FRAME_SKIP=5
```

**3. Start services (4 terminals):**
```bash
# Terminal 1 — PostgreSQL + Redis
docker-compose up -d

# Terminal 2 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 3 — Celery worker
celery -A worker.tasks worker --loglevel=info --pool=solo

# Terminal 4 — Streamlit dashboard
streamlit run dashboard\app.py
```

**4. Open** `http://localhost:8501` — upload a traffic video and watch the pipeline run.

API docs available at `http://localhost:8000/docs`

---

## Dashboard pages

| Page | What it shows |
|------|--------------|
| **Analyse** | Upload video, live progress bar, quick summary |
| **Congestion** | Vehicle count timeline, level distribution pie chart, peak moments |
| **Violations** | Filterable log of helmet violations with confidence scores |
| **Analytics** | Congestion heatmap, confidence distribution, Bengaluru junction map |

---

## Scalability

The FastAPI + Celery + Redis architecture is horizontally scalable — adding 100 more camera streams means running more Celery workers, not rewriting any code. Each worker processes one video independently, writes to the shared PostgreSQL database, and results appear on the dashboard in real time.

---

## Dataset

Helmet detection model trained on the [Helmet Detection dataset](https://universe.roboflow.com/project-pdvie/helmet-srsz5) from Roboflow Universe — 2,042 images, 2 classes (`With Helmet`, `Without Helmet`), CC BY 4.0 licence.

---

*Built for Gridlock Hackathon 2.0 — Flipkart's call to solve Bengaluru's traffic challenges with AI.*
