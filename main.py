# main.py (FastAPI/Uvicorn対応版)

# --------------------------------------------------------------------------
# 1. ライブラリのインポート
# --------------------------------------------------------------------------
import os
import json
import requests
import chromadb
import traceback
import csv
from datetime import datetime
import urllib3
import pypdf
import docx
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# 2. FastAPIアプリケーションの初期設定
# --------------------------------------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ★★★★★ 修正点: ローカルPCで動作するようにパス設定を統一 ★★★★★
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
FEEDBACK_FILE_PATH = os.path.join(BASE_DIR, "feedback.csv")


# グローバル変数としてモデルとDBクライアントを保持
embedding_models = {}
db_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client
    print("--- Application Startup ---")
    print(f"Project base directory: {BASE_DIR}")
    print(f"Database path: {DB_PATH}")
    print(f"Feedback log path: {FEEDBACK_FILE_PATH}")
    
    try:
        if not os.path.exists(FEEDBACK_FILE_PATH):
            print(f"Log file not found at {FEEDBACK_FILE_PATH}. Creating a new one.")
            with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['log_id', 'timestamp', 'rating', 'query', 'response'])
            print("Log file created successfully.")
    except Exception as e:
        print(f"FATAL: Error creating log file on startup: {e}")
    
    print("Initializing ChromaDB client...")
    try:
        # ★★★★★ 修正点: データをファイルに保存する永続化クライアントに戻す ★★★★★
        db_client = chromadb.PersistentClient(path=DB_PATH)
        print(f"ChromaDB persistent client initialized successfully.")
    except Exception as e:
        print(f"FATAL: Error initializing ChromaDB: {e}")
    
    print("Loading default embedding model...")
    get_embedding_model()
    print("Default embedding model loaded.")
    print("--------------------------")
    
    yield
    
    print("--- Application shutting down. ---")
    embedding_models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ヘルパー関数 ---
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    if model_name not in embedding_models:
        print(f"Loading embedding model: {model_name}")
        try:
            embedding_models[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            return None
    return embedding_models[model_name]

# --- Pydanticモデル定義 ---
class ScrapeRequest(BaseModel):
    url: str
    collection_name: str
    embedding_model: str = "all-MiniLM-L6-v2"

class FeedbackRequest(BaseModel):
    log_id: str
    rating: str

class CollectionRequest(BaseModel):
    name: str

class ChatRequest(BaseModel):
    query: str
    model: str = "llama3:8b"
    collection: str
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3

# --------------------------------------------------------------------------
# 3. APIエンドポイント (ルート) の定義
# --------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_client():
    file_path = os.path.join(BASE_DIR, "client.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="client.html not found.")
    return FileResponse(file_path)

@app.get("/admin", response_class=FileResponse, include_in_schema=False)
async def serve_admin():
    file_path = os.path.join(BASE_DIR, "admin.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="admin.html not found.")
    return FileResponse(file_path)

@app.get("/log", response_class=FileResponse, include_in_schema=False)
async def serve_log_page():
    file_path = os.path.join(BASE_DIR, "log.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="log.html not found.")
    return FileResponse(file_path)

@app.get("/logs", include_in_schema=False)
async def get_logs():
    logs_data = []
    try:
        if not os.path.exists(FEEDBACK_FILE_PATH):
            return [] 
        with open(FEEDBACK_FILE_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs_data.append(row)
        return logs_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/chromadb/status")
async def chromadb_status():
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
    try:
        db_client.heartbeat()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ollama/status")
async def ollama_status():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        response.raise_for_status()
        return {"connected": True, "models": response.json().get('models', [])}
    except Exception as e:
        return {"connected": False, "detail": f"Failed to connect to local Ollama server: {e}"}

@app.post("/scrape")
async def scrape_and_add_website(req: ScrapeRequest):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(req.url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'lxml')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator='\n', strip=True)
        if not text: raise HTTPException(status_code=400, detail="No text content found.")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        collection = db_client.get_or_create_collection(name=req.collection_name)
        model = get_embedding_model(req.embedding_model)
        if not model: raise HTTPException(status_code=500, detail=f"Embedding model not available")
        embeddings = model.encode(chunks).tolist()
        ids = [f"{req.url}:{i}" for i in range(len(chunks))]
        metadatas = [{"source_url": req.url} for _ in range(len(chunks))]
        collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
        return {"message": f"Successfully added content from {req.url}", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...), embedding_model: str = Form(...)):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        text, filename = "", file.filename
        contents = await file.read()
        if filename.lower().endswith('.pdf'):
            import io
            reader = pypdf.PdfReader(io.BytesIO(contents))
            text = "".join(page.extract_text() or "" for page in reader.pages)
        elif filename.lower().endswith('.docx'):
            import io
            doc = docx.Document(io.BytesIO(contents))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            try: text = contents.decode('utf-8')
            except UnicodeDecodeError: text = contents.decode('shift_jis')
        if not text.strip(): raise HTTPException(status_code=400, detail="No text extracted.")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        collection = db_client.get_or_create_collection(name=collection_name)
        model = get_embedding_model(embedding_model)
        if not model: raise HTTPException(status_code=500, detail="Embedding model not available")
        embeddings = model.encode(chunks).tolist()
        ids = [f"{filename}:{i}" for i in range(len(chunks))]
        metadatas = [{"filename": filename} for _ in range(len(chunks))]
        collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
        return {"message": "File processed successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def handle_feedback(req: FeedbackRequest):
    try:
        if not os.path.exists(FEEDBACK_FILE_PATH):
            with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['log_id', 'timestamp', 'rating', 'query', 'response'])
        
        rows = []
        updated = False
        
        with open(FEEDBACK_FILE_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            log_id_index = header.index('log_id')
            rating_index = header.index('rating')
            for row in reader:
                if row and row[log_id_index] == req.log_id:
                    row[rating_index] = req.rating
                    updated = True
                rows.append(row)

        if not updated: raise HTTPException(status_code=404, detail=f"Log ID {req.log_id} not found.")

        with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return {"status": "success", "message": "Feedback updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    return [{"name": c.name} for c in db_client.list_collections()]

@app.post("/collections")
async def create_collection(req: CollectionRequest):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        db_client.get_or_create_collection(name=req.name)
        return JSONResponse(content={"message": f"Collection '{req.name}' created."}, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        db_client.delete_collection(name=collection_name)
        return {"message": f"Collection '{collection_name}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/documents")
async def get_documents_in_collection(collection_name: str):
    if not db_client: raise HTTPException(status_code=500, detail="ChromaDB not available")
    try:
        collection = db_client.get_collection(name=collection_name)
        data = collection.get(include=['metadatas'])
        sources = set(md.get('filename') or md.get('source_url', 'unknown') for md in data['metadatas'])
        return {"documents": [{"id": src} for src in sorted(list(sources))], "count": collection.count()}
    except ValueError:
        return {"documents": [], "count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    if not db_client:
        raise HTTPException(status_code=500, detail="ChromaDB not available")

    log_id = str(uuid.uuid4())

    async def generate_response():
        full_response = ""
        try:
            yield f"data: {json.dumps({'log_id': log_id})}\n\n"
            embedding_model = get_embedding_model(req.embedding_model)
            if not embedding_model:
                yield f"data: {json.dumps({'error': 'Embedding model not available'})}\n\n"
                return
            query_embedding = embedding_model.encode([req.query]).tolist()
            try:
                collection = db_client.get_collection(name=req.collection)
                results = collection.query(query_embeddings=query_embedding, n_results=req.top_k)
                context = "\n---\n".join(results['documents'][0])
                yield f"data: {json.dumps({'search_results': True})}\n\n"
            except Exception:
                context = "利用可能な参考情報はありません。"
            
            prompt = f"""
# 命令書
あなたは札幌学院大学の学生をサポートする、非常に優秀で親切なAIアシスタントです。
# 制約条件
- 必ず提供された「参考情報」の内容を最優先にして、正確な回答を生成してください。
- 特に、ユーザーの質問に「生協」や「アパート」といった言葉が含まれている場合は、回答に加えて、必ず以下の大学生協ルームガイドのウェブサイトへ誘導してください: https://www.hokkaido-univcoop.jp/sgu/
- 学生が理解しやすいように、専門用語を避け、丁寧で親しみやすい言葉遣いを心がけてください。
- 重要な情報は、箇条書き（・箇条書き）や太字を使って整理し、見やすくしてください。
- 参考情報にウェブサイトのURLが含まれている場合は、回答の末尾に「詳細は以下のリンクもご確認ください」のように案内し、そのURLを提示してください。
- 参考情報に答えが直接見つからない場合でも、関連性が高いと思われる情報があれば、それを提示してください。
- 参考情報に全く関連する情報がない場合は、「申し訳ありませんが、その質問に関する情報は見つかりませんでした。大学の公式サイトをご確認いただくか、学生支援課にお問い合わせください。」と回答してください。
- 架空の情報や、参考情報に記載されていない推測を回答に含めてはいけません。
### 参考情報:
{context}
### ユーザーの質問:
{req.query}
### 回答:
"""
            payload = {"model": req.model, "messages": [{'role': 'user', 'content': prompt}], "stream": True}
            
            # ★★★★★ 修正点: Ollamaの接続先をローカルPCに固定 ★★★★★
            OLLAMA_HOST = "http://127.0.0.1:11434"
            OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
            
            with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded_line = json.loads(line.decode('utf-8'))
                        if 'content' in decoded_line.get('message', {}):
                            content = decoded_line['message']['content']
                            full_response += content
                            content_data = {"content": content}
                            yield f"data: {json.dumps(content_data)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            try:
                if not os.path.exists(FEEDBACK_FILE_PATH):
                    with open(FEEDBACK_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['log_id', 'timestamp', 'rating', 'query', 'response'])
                
                with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([log_id, timestamp, 'N/A', req.query, full_response])
                print(f"Successfully logged conversation.")
            except Exception as e:
                print(f"Failed to write log for conversation {log_id}: {e}")

    return StreamingResponse(generate_response(), media_type='text/event-stream')
