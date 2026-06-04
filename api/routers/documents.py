import os
import re
import shutil
import tempfile
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from api.schemas import DocumentUploadResponse, DocumentInfo, DocumentListResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "books")

# Windows 非法文件名字符
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str) -> str:
    """清除文件名中 Windows/POSIX 非法字符，并合并空白"""
    cleaned = _INVALID_FS_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
    return cleaned or "untitled"


def _count_chunks_by_title(index_manager, book_title: str) -> int:
    """统计指定书籍在 ChromaDB 中的实际向量块数"""
    try:
        result = index_manager.chroma_collection.get(
            where={"book_title": book_title},
            include=[],
        )
        return len(result.get("ids", []))
    except Exception:
        return 0


def _file_path_for_title(index_manager, book_title: str):
    """查指定书籍的原始 PDF 路径；查不到返回 None"""
    try:
        result = index_manager.chroma_collection.get(
            where={"book_title": book_title},
            include=["metadatas"],
            limit=1,
        )
    except Exception:
        return None
    for meta in result.get("metadatas") or []:
        fp = meta.get("file_path")
        if fp:
            return fp
    return None


def _resolve_within_uploads(file_path: str):
    """把（可能是相对的）file_path 解析为绝对路径，并校验仍在 UPLOAD_DIR 内（防目录穿越）。
    通过返回绝对路径，否则返回 None。"""
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    abs_path = os.path.abspath(file_path)
    base = os.path.abspath(UPLOAD_DIR)
    try:
        if os.path.commonpath([abs_path, base]) != base:
            return None
    except ValueError:
        # 不同盘符（Windows）会抛 ValueError
        return None
    return abs_path


def create_documents_router(index_manager):
    """工厂函数：注入 index_manager 依赖"""
    router = APIRouter(prefix="/api", tags=["documents"])

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    @router.post("/documents/upload")
    async def upload_book(
        file: UploadFile = File(...),
        book_title: str = Form(...),
    ):
        """上传技术书籍 PDF 并入库"""
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

        # 时间戳避免文件名冲突；sanitize 掉 Windows 非法字符（: < > | ? * 等）
        raw_name, ext = os.path.splitext(file.filename)
        name = _sanitize_filename(raw_name)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(UPLOAD_DIR, f"{name}_{ts}{ext.lower()}")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                shutil.copyfileobj(file.file, f)
            # 清理同名旧文件（跳过被锁定的）
            for old_path in [os.path.join(UPLOAD_DIR, p) for p in os.listdir(UPLOAD_DIR)
                             if p.startswith(name) and p.lower().endswith(".pdf")]:
                try:
                    os.remove(old_path)
                except (PermissionError, OSError):
                    pass
            shutil.move(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # 入库
        try:
            index_manager.add_book(
                pdf_path=os.path.abspath(save_path),
                book_title=book_title,
            )
            # 统计当前书籍的真实块数（按 book_title 过滤）
            book_chunk_count = _count_chunks_by_title(index_manager, book_title)
            return DocumentUploadResponse(
                filename=file.filename,
                book_title=book_title,
                status="indexed",
                chunk_count=book_chunk_count,
                message=f"《{book_title}》入库成功，共 {book_chunk_count} 个向量块",
            )
        except Exception as e:
            return DocumentUploadResponse(
                filename=file.filename,
                book_title=book_title,
                status="failed",
                message=str(e),
            )

    @router.get("/documents")
    async def list_documents():
        """列出已入库的文档"""
        all_data = index_manager.chroma_collection.get(include=["metadatas"])

        books = {}
        for meta in all_data["metadatas"]:
            title = meta.get("book_title", "未知")
            if title not in books:
                books[title] = {
                    "book_title": title,
                    "file_path": meta.get("file_path", ""),
                    "chunk_count": 0,
                }
            books[title]["chunk_count"] += 1
            # 记录最大页码
            page = meta.get("page", meta.get("page_start", 0))
            books[title]["page_count"] = max(books[title].get("page_count", 0), page)

        return DocumentListResponse(
            books=[
                DocumentInfo(
                    book_title=b["book_title"],
                    file_path=b["file_path"],
                    page_count=b.get("page_count", 0),
                    chunk_count=b["chunk_count"],
                    indexed_at="",
                )
                for b in books.values()
            ],
            total_vectors=index_manager.chroma_collection.count(),
        )

    @router.get("/documents/{book_title}/file")
    async def get_document_file(book_title: str):
        """提供原始 PDF 文件（内联显示），供前端引用卡片跳转。
        浏览器原生 PDF 阅读器支持 URL 片段 #page=N 定位到指定页。"""
        file_path = _file_path_for_title(index_manager, book_title)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"未找到书籍: {book_title}")

        abs_path = _resolve_within_uploads(file_path)
        if abs_path is None:
            raise HTTPException(status_code=403, detail="非法文件路径")
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="原始文件已不存在")

        filename = os.path.basename(abs_path)
        return FileResponse(
            abs_path,
            media_type="application/pdf",
            headers={
                # inline 让浏览器内嵌显示而非下载；filename* 用 RFC 5987 编码支持中文
                "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}",
            },
        )

    @router.delete("/documents/{book_title}")
    async def delete_document(book_title: str):
        """按书名删除文档索引"""
        all_data = index_manager.chroma_collection.get(include=["metadatas"])

        ids_to_delete = []
        for i, meta in enumerate(all_data["metadatas"]):
            if meta.get("book_title") == book_title:
                ids_to_delete.append(all_data["ids"][i])

        if ids_to_delete:
            index_manager.chroma_collection.delete(ids=ids_to_delete)
            return {"status": "deleted", "book_title": book_title, "count": len(ids_to_delete)}

        raise HTTPException(status_code=404, detail=f"未找到书籍: {book_title}")

    return router
