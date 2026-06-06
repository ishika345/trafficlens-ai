from fastapi import FastAPI

app = FastAPI(title="TrafficLens AI")

@app.get("/")
def root():
    return {"status": "ok", "message": "TrafficLens AI is running"}