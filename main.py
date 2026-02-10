import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine_async import rag_engine_async
import app.api.auth as auth
import app.api.admin as admin
import app.api.chat as chat  # Unified chat module with hybrid classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up FastAPI...")
    await db_engine_async.connect()
    count = await rag_engine_async.index_mongodb_collections()
    print(f"Indexed {count} documents.")
    yield


app = FastAPI(title="UCSI Chatbot", version="3.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router, prefix="/api", tags=["Auth"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

# Static Files
# Mount specific folders to root to support relative paths in HTML (css/... and js/...)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/css", StaticFiles(directory="static/site/css"), name="css")
app.mount("/js", StaticFiles(directory="static/site/js"), name="js")

# Mount site assets if any other folders exist in static/site (images, etc)
# app.mount("/site", StaticFiles(directory="static/site"), name="site")

# Root
@app.get("/")
async def read_root():
    return FileResponse("static/site/code_hompage.html")


@app.get("/admin")
async def read_admin():
    admin_html = Path("static/admin/admin.html")
    if not admin_html.exists():
        return FileResponse("static/site/code_hompage.html")
    return FileResponse(str(admin_html))

if __name__ == "__main__":
    import uvicorn
    # Updated to main:app
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
