import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Optional

DEV_MODE = bool(os.getenv("DEV_MODE"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "bahai-crossrefs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/e5-large-v2")

app = FastAPI(title="Bahá’í Cross‑Reference API")
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(MODEL_NAME)

# ---- Auth helpers --------------------------------------------------------
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as grequests
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    _req = grequests.Request()
except ImportError:
    GOOGLE_CLIENT_ID = None
    _req = None

def verify_token(bearer: Optional[str]):
    if DEV_MODE:
        return {"sub": "dev-user"}
    if not bearer or not bearer.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = bearer.split()[1]
    info = id_token.verify_oauth2_token(token, _req, GOOGLE_CLIENT_ID)
    if not info.get("email_verified"):
        raise HTTPException(status_code=403, detail="Unverified email")
    return info  # dict with email, name, sub

# ---- Models --------------------------------------------------------------
class Query(BaseModel):
    query: str
    top_k: int = 5
    concept_filter: Optional[str] = None

class ExplainReq(BaseModel):
    a_id: str
    b_id: str

# ---- Routes --------------------------------------------------------------
@app.post("/search")
async def search(q: Query, authorization: str = Header(default=None)):
    verify_token(authorization)
    vec = model.encode(f"query: {q.query}").tolist()

    flt = None
    if q.concept_filter:
        flt = {"must": [{"key": "concepts", "match": {"value": q.concept_filter}}]}
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=q.top_k,
        query_filter=flt
    )
    return [
        {
            "id": h.id,
            "score": h.score,
            "text": h.payload["text"],
            "source": h.payload["source"],
            "concepts": h.payload.get("concepts", [])
        }
        for h in hits
    ]

# ---- Explanation endpoint (optional) -------------------------------------
try:
    import openai, textwrap as _tw
    openai.api_key = os.getenv("OPENAI_API_KEY")
    PROMPT = _tw.dedent("""        You are a scholar of comparative religion. In two sentences, explain
    the conceptual similarity between the following passages, quoting a key
    phrase from each. Focus on shared themes, not wording quirks.
    Passage A: {a}
    Passage B: {b}
    """)
    def _llm_explain(a, b):
        response = openai.ChatCompletion.create(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125"),
            messages=[{"role": "user", "content": PROMPT.format(a=a[:500], b=b[:500])}],
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
except Exception:
    def _llm_explain(a, b):
        return "Explanation service unavailable."

@app.post("/explain")
async def explain(req: ExplainReq, authorization: str = Header(default=None)):
    verify_token(authorization)
    a_doc = client.retrieve(collection_name=COLLECTION, ids=[req.a_id])[0]
    b_doc = client.retrieve(collection_name=COLLECTION, ids=[req.b_id])[0]
    explanation = _llm_explain(a_doc.payload["text"], b_doc.payload["text"])
    return {"explanation": explanation}
