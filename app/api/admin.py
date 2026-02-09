from pathlib import Path

from fastapi import APIRouter, Body, File, UploadFile
from fastapi.responses import JSONResponse

from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine_async import rag_engine_async

router = APIRouter()


@router.get("/stats")
async def get_stats():
    student_stats = await db_engine_async.get_student_stats()

    total_feedbacks = 0
    positive_feedbacks = 0
    unanswered_logs = []
    recent_feedbacks = []

    db = db_engine_async.db
    if db is not None:
        try:
            total_feedbacks = await db.Feedback.count_documents({})
            positive_feedbacks = await db.Feedback.count_documents(
                {"$or": [{"rating": "positive"}, {"score": {"$gte": 4}}]}
            )
        except Exception:
            pass

        try:
            unanswered_logs = await db.unanswered.find({}, {"_id": 0}).sort("timestamp", -1).limit(10).to_list(length=10)
        except Exception:
            unanswered_logs = []

        try:
            recent_feedbacks = await db.Feedback.find({}, {"_id": 0}).sort("timestamp", -1).limit(10).to_list(length=10)
        except Exception:
            recent_feedbacks = []

    satisfaction_rate = 0
    if total_feedbacks > 0:
        satisfaction_rate = round((positive_feedbacks / total_feedbacks) * 100, 2)

    return {
        "status": "ok",
        "db_connected": db is not None,
        "total_students": student_stats.get("total_students", 0),
        "gender": student_stats.get("gender", {}),
        "top_nationalities": student_stats.get("top_nationalities", {}),
        # Keep legacy admin panel keys.
        "satisfaction_rate": satisfaction_rate,
        "total_feedbacks": total_feedbacks,
        "unanswered_count": len(unanswered_logs),
        "unanswered_logs": unanswered_logs,
        "recent_feedbacks": recent_feedbacks,
    }


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "No file provided"},
        )

    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".csv"}:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Only PDF/TXT/CSV files are supported"},
        )

    content = await file.read()
    kb_dir = Path("data/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)
    save_path = kb_dir / safe_name
    with open(save_path, "wb") as f:
        f.write(content)

    success = await rag_engine_async.ingest_file(str(save_path))
    if not success:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "File saved but indexing failed"},
        )

    return {
        "success": True,
        "filename": safe_name,
        "status": "Uploaded and Indexed",
        "message": f"Successfully ingested {safe_name}",
    }


@router.post("/reindex")
async def reindex_mongodb():
    count = await rag_engine_async.index_mongodb_collections()
    return {
        "success": True,
        "status": "reindexed",
        "indexed_documents": count,
    }


@router.get("/files")
async def list_files():
    kb_dir = Path("data/knowledge_base")
    if not kb_dir.exists():
        return {"files": []}

    files = []
    for p in kb_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in {".pdf", ".txt", ".csv", ".docx"}:
            files.append(p.name)

    files.sort(key=str.lower)
    return {"files": files}


@router.delete("/files")
async def delete_file(body: dict = Body(default={})):
    filename = str((body or {}).get("filename") or "").strip()
    if not filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Filename required"},
        )

    safe_name = Path(filename).name
    file_path = Path("data/knowledge_base") / safe_name
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "File not found"},
        )

    try:
        file_path.unlink()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Delete failed: {e}"},
        )

    return {"success": True, "deleted": safe_name}
