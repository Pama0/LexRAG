import re

import fitz
from llama_index.core.schema import Document


class BookPDFParser:

    CHAPTER_PATTERNS = [
        re.compile(r'^第[一二三四五六七八九十百千\d]+章\s*'),
        re.compile(r'^第[一二三四五六七八九十百千\d]+节\s*'),
        re.compile(r'^(?:Chapter|CHAPTER)\s*\d+'),
    ]

    MONOSPACE_FONTS = {
        'courier', 'consolas', 'monaco', 'monospace',
        'source code', 'fira code', 'jetbrains',
        'liberation mono', 'menlo',
    }

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        heading_font_size_threshold: float = 12.0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_font_size_threshold = heading_font_size_threshold

    def parse(self, pdf_path: str, book_title: str) -> list[Document]:
        doc = fitz.open(pdf_path)
        try:
            blocks = self._extract_blocks(doc)
        finally:
            doc.close()

        sections = self._detect_structure(blocks)
        documents = self._build_documents(sections, book_title, pdf_path)
        return documents

    def _extract_blocks(self, doc: fitz.Document) -> list:
        all_blocks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in page_blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    span_info = self._extract_span_info(line)
                    if span_info["text"]:
                        span_info["page"] = page_num + 1
                        span_info["block_bbox"] = block["bbox"]
                        all_blocks.append(span_info)
        return all_blocks

    def _extract_span_info(self, line: dict) -> dict:
        texts = []
        font_names = set()
        font_sizes = []
        flags_list = []

        for span in line["spans"]:
            texts.append(span["text"])
            font_names.add(span["font"].lower())
            font_sizes.append(span["size"])
            flags_list.append(span.get("flags", 0))

        avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
        is_bold = any(f & 2 for f in flags_list)

        return {
            "text": "".join(texts),
            "font_size": round(avg_size, 1),
            "is_bold": is_bold,
            "fonts": font_names,
            "page": 0,
            "block_bbox": None,
        }

    def _detect_structure(self, blocks: list) -> list:
        sections = []
        current_section = None

        for block in blocks:
            text = block["text"].strip()
            if not text:
                continue

            heading = self._match_heading(text, block)
            if heading:
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "heading": heading,
                    "page_start": block["page"],
                    "page_end": block["page"],
                    "blocks": [],
                }
                continue

            content_type = self._classify_block(text, block)
            block["content_type"] = content_type

            if current_section is None:
                current_section = {
                    "heading": "",
                    "page_start": block["page"],
                    "page_end": block["page"],
                    "blocks": [],
                }

            current_section["blocks"].append(block)
            current_section["page_end"] = block["page"]

        if current_section:
            sections.append(current_section)

        return sections

    def _match_heading(self, text: str, block: dict) -> str | None:
        is_large = block["font_size"] >= self.heading_font_size_threshold
        is_bold = block.get("is_bold", False)

        if not (is_large or is_bold):
            return None

        for pat in self.CHAPTER_PATTERNS:
            if pat.match(text):
                return text

        if is_large and re.match(r'^\d+(\.\d+)*\s', text):
            return text

        if is_large and len(text) < 60 and not text.endswith(('.', '。', ';', '；')):
            return text

        return None

    def _classify_block(self, text: str, block: dict) -> str:
        if self._is_monospace(block.get("fonts", set())):
            return "code"

        code_indicators = [
            r'^\s*(?:def |class |import |from |SELECT |CREATE |INSERT |UPDATE |DELETE )',
            r'^\s*(?:if |for |while |try |catch |function |const |let |var |export )',
            r'^\s*(?:public |private |protected |static |void |int |String |List<)',
            r'^\s*#include|^\s*package |^\s*use |^\s*impl ',
            r'[{};\[\]]$',
        ]
        for pat in code_indicators:
            if re.search(pat, text):
                return "code"

        return "text"

    def _is_monospace(self, fonts: set) -> bool:
        for f in fonts:
            f_lower = f.lower()
            if any(mf in f_lower for mf in self.MONOSPACE_FONTS):
                return True
        return False

    def _build_documents(
        self, sections: list, book_title: str, pdf_path: str
    ) -> list[Document]:
        documents = []

        for section in sections:
            chunks = self._chunk_section(section)
            for chunk in chunks:
                doc = Document(
                    text=chunk["text"],
                    metadata={
                        "book_title": book_title,
                        "chapter": section["heading"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "content_type": chunk.get("content_type", "text"),
                        "file_path": pdf_path,
                        "chunk_type": "book_chunk",
                    },
                )
                documents.append(doc)

        return documents

    def _chunk_section(self, section: dict) -> list:
        chunks = []
        buffer = []
        buffer_len = 0
        page_start = section["page_start"]
        page_end = page_start

        for block in section["blocks"]:
            text = block["text"].strip()
            if not text:
                continue

            if buffer_len + len(text) > self.chunk_size and buffer:
                chunks.append({
                    "text": "\n\n".join(buffer),
                    "page_start": page_start,
                    "page_end": page_end,
                    "content_type": self._dominant_type(buffer, section["blocks"]),
                })
                overlap_text = buffer[-1] if buffer else ""
                buffer = [overlap_text] if len(overlap_text) < self.chunk_overlap else []
                buffer_len = len(overlap_text) if buffer else 0
                page_start = page_end

            buffer.append(text)
            buffer_len += len(text)
            page_end = block["page"]

        if buffer:
            chunks.append({
                "text": "\n\n".join(buffer),
                "page_start": page_start,
                "page_end": page_end,
                "content_type": self._dominant_type(buffer, section["blocks"]),
            })

        return chunks

    def _dominant_type(self, buffer: list, blocks: list) -> str:
        code_count = sum(1 for b in blocks if b.get("content_type") == "code")
        return "code" if code_count > len(blocks) * 0.5 else "text"
