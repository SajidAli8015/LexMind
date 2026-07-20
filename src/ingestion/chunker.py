"""
Legal-Aware Chunker for LexMind
Groups parsed document elements into meaningful chunks
at article and section boundaries for optimal retrieval.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

from src.ingestion.parser import ParsedDocument, DocumentElement


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class Chunk:
    """
    Represents one searchable unit of legal content.

    Each chunk corresponds to one article or section of a
    legal document, containing the heading and all its
    sub-clauses and paragraphs as a single text block.

    Example:
        Chunk(
            text="ARTICLE 47 TERMINATION\\n"
                 "Either party may terminate...\\n"
                 "47.1 Notice must be given in writing.\\n"
                 "47.2 Termination is immediate if...",
            article_ref="Article 47",
            chunk_type="article",
            chunk_index=5
        )
    """
    chunk_id: str               # unique ID: "{doc_id}_chunk_{index}"
    doc_id: str                 # parent document identifier
    text: str                   # complete chunk text
    chunk_type: str             # article / part / paragraph / sub_chunk
    article_ref: str = ""       # e.g. "Article 47" for metadata filtering
    page_number: int = 0
    chunk_index: int = 0        # position in document (1-based)
    total_chunks: int = 0       # total chunks in this document
    char_count: int = 0         # character count of text
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Auto-calculate char_count after init."""
        if not self.char_count:
            self.char_count = len(self.text)


@dataclass
class ChunkingResult:
    """
    Result of chunking one document.
    Contains all chunks plus summary statistics.
    """
    doc_id: str
    file_name: str
    chunks: List[Chunk] = field(default_factory=list)
    total_chunks: int = 0
    total_chars: int = 0
    articles_found: int = 0

    def __post_init__(self):
        self.total_chunks = len(self.chunks)
        self.total_chars = sum(c.char_count for c in self.chunks)
        self.articles_found = sum(
            1 for c in self.chunks if c.chunk_type == "article"
        )


# ─── Chunker Class ───────────────────────────────────────────

class LegalChunker:
    """
    Splits a ParsedDocument into legal-aware chunks.

    Chunking strategy:
    - Level 1 headings (PART/CHAPTER) → their own chunk
    - Level 2 headings (ARTICLE/SECTION) → start a new chunk,
      collect all content until the next level 2 heading
    - If a chunk exceeds max_chunk_size → split into sub-chunks
      with overlap to preserve context at boundaries
    - Remaining content after last heading → paragraph chunk
    """

    def __init__(
        self,
        max_chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
    ):
        """
        Args:
            max_chunk_size: Maximum characters per chunk before splitting
            chunk_overlap:  Characters of overlap between sub-chunks
            min_chunk_size: Minimum characters — smaller chunks are skipped
        """
        from src.config import settings
        self.max_chunk_size = (
            max_chunk_size if max_chunk_size is not None
            else settings.CHUNK_SIZE
        )
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None
            else settings.CHUNK_OVERLAP
        )
        self.min_chunk_size = (
            min_chunk_size if min_chunk_size is not None
            else settings.CHUNK_MIN_SIZE
        )
        logger.info(
            f"LegalChunker initialized — "
            f"max_size={self.max_chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def chunk(self, doc: ParsedDocument) -> ChunkingResult:
        """
        Main method: chunk a ParsedDocument into legal chunks.

        Args:
            doc: ParsedDocument from parser.py

        Returns:
            ChunkingResult with all chunks and statistics
        """
        doc_id = self._generate_doc_id(doc.file_name)
        logger.info(
            f"Chunking {doc.file_name} "
            f"({doc.element_count()} elements)"
        )

        # Group elements into raw groups by article boundary
        raw_groups = self._group_elements(doc.elements)

        # Convert groups into Chunk objects
        chunks = []
        for group_idx, group in enumerate(raw_groups):
            group_chunks = self._group_to_chunks(
                group=group,
                doc_id=doc_id,
                file_name=doc.file_name,
                base_index=len(chunks)
            )
            chunks.extend(group_chunks)

        # Set total_chunks on every chunk now that we know the total
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
            chunk.metadata["total_chunks"] = len(chunks)

        result = ChunkingResult(
            doc_id=doc_id,
            file_name=doc.file_name,
            chunks=chunks
        )

        logger.info(
            f"Chunked {doc.file_name}: "
            f"{result.total_chunks} chunks, "
            f"{result.articles_found} articles, "
            f"{result.total_chars} total chars"
        )
        return result

    def _group_elements(
        self, elements: List[DocumentElement]
    ) -> List[List[DocumentElement]]:
        """
        Group elements by article/section boundary.

        A new group starts whenever we encounter a level 1 or
        level 2 heading. All following elements (level 0 paragraphs
        and level 3 sub-clauses) belong to that group until the
        next level 1 or 2 heading appears.

        Example:
            Input elements:
              [HEADING L1] PART I
              [HEADING L2] ARTICLE 1       ← new group starts
              [PARAGRAPH]  Definitions...  ← belongs to ARTICLE 1
              [HEADING L3] 1.1 ...         ← belongs to ARTICLE 1
              [HEADING L2] ARTICLE 2       ← new group starts
              [PARAGRAPH]  Obligations...  ← belongs to ARTICLE 2

            Output groups:
              Group 1: [PART I]
              Group 2: [ARTICLE 1, Definitions, 1.1]
              Group 3: [ARTICLE 2, Obligations]
        """
        groups = []
        current_group = []

        for element in elements:
            is_group_boundary = (
                element.element_type == "heading"
                and element.level in (1, 2)
            )

            if is_group_boundary:
                # Save current group if it has content
                if current_group:
                    groups.append(current_group)
                # Start new group with this heading
                current_group = [element]
            else:
                # Add to current group
                if current_group:
                    current_group.append(element)
                else:
                    # Content before any heading — start a group
                    current_group = [element]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _group_to_chunks(
        self,
        group: List[DocumentElement],
        doc_id: str,
        file_name: str,
        base_index: int,
    ) -> List[Chunk]:
        """
        Convert one element group into one or more Chunk objects.

        If the group text fits within max_chunk_size → one chunk.
        If it exceeds max_chunk_size → split into sub-chunks with overlap.
        """
        # Build the full text for this group
        text_parts = [e.text for e in group if e.text.strip()]
        full_text = "\n".join(text_parts)

        if not full_text.strip():
            return []

        if len(full_text) < self.min_chunk_size:
            return []

        # Determine chunk metadata from first element (the heading)
        first_elem = group[0]
        article_ref = self._extract_article_ref(first_elem.text)
        chunk_type = self._determine_chunk_type(first_elem)
        page_number = first_elem.page_number

        # Base metadata applied to all chunks from this group
        base_metadata = {
            "doc_id": doc_id,
            "file_name": file_name,
            "article_ref": article_ref,
            "chunk_type": chunk_type,
            "page_number": page_number,
        }

        # If text fits in one chunk — return single chunk
        if len(full_text) <= self.max_chunk_size:
            chunk_index = base_index + 1
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            return [Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=full_text,
                chunk_type=chunk_type,
                article_ref=article_ref,
                page_number=page_number,
                chunk_index=chunk_index,
                metadata={**base_metadata, "chunk_index": chunk_index}
            )]

        # Text too long — split into sub-chunks with overlap
        return self._split_with_overlap(
            text=full_text,
            doc_id=doc_id,
            base_metadata=base_metadata,
            chunk_type="sub_chunk",
            article_ref=article_ref,
            page_number=page_number,
            base_index=base_index,
        )

    def _split_with_overlap(
        self,
        text: str,
        doc_id: str,
        base_metadata: dict,
        chunk_type: str,
        article_ref: str,
        page_number: int,
        base_index: int,
    ) -> List[Chunk]:
        """
        Split a long text into overlapping sub-chunks.

        Overlap example with max_size=20, overlap=5:
            Full text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            Sub-chunk 1: "ABCDEFGHIJKLMNOPQRST"    (chars 0-19)
            Sub-chunk 2: "PQRSTUVWXYZ"             (starts at 15, overlap=5)

        For legal text we try to split at sentence boundaries
        rather than arbitrary character positions.
        """
        sub_chunks = []
        start = 0
        sub_index = 0

        while start < len(text):
            end = start + self.max_chunk_size

            if end >= len(text):
                # Last piece — take everything remaining
                sub_text = text[start:]
            else:
                # Try to find a good split point (sentence boundary)
                split_point = self._find_split_point(text, start, end)
                sub_text = text[start:split_point]
                end = split_point

            sub_text = sub_text.strip()
            if len(sub_text) >= self.min_chunk_size:
                chunk_index = base_index + sub_index + 1
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                sub_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=sub_text,
                    chunk_type=chunk_type,
                    article_ref=article_ref,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_index,
                        "sub_chunk_index": sub_index + 1,
                        "is_sub_chunk": True,
                    }
                ))
                sub_index += 1

            if end >= len(text):
                break

            # Move start forward but keep overlap
            start = end - self.chunk_overlap

        return sub_chunks

    def _find_split_point(
        self, text: str, start: int, preferred_end: int
    ) -> int:
        """
        Find the best split point near preferred_end.

        Prefers splitting at:
        1. End of a sentence (. or ؟ for Arabic)
        2. End of a line
        3. End of a word
        4. preferred_end exactly if nothing better found
        """
        search_start = max(start, preferred_end - 200)
        search_text = text[search_start:preferred_end]

        # Try sentence boundary
        for pattern in [r'\. ', r'؟ ', r'\n', r' ']:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                last_match = matches[-1]
                return search_start + last_match.end()

        return preferred_end

    def _extract_article_ref(self, heading_text: str) -> str:
        """
        Extract a clean article reference from heading text.

        Examples:
            "ARTICLE 47 TERMINATION RIGHTS" → "Article 47"
            "SECTION 3 PAYMENT"             → "Section 3"
            "PART I GENERAL PROVISIONS"     → "Part I"
            "3.2 Sub-clause content"        → "Clause 3.2"
            "Some random heading"           → "Some random heading"
        """
        text = heading_text.strip()

        # Match ARTICLE/SECTION/CLAUSE + number
        match = re.match(
            r'^(ARTICLE|SECTION|CLAUSE|SCHEDULE|PART|CHAPTER)\s+'
            r'([IVXivx\d]+)',
            text,
            re.IGNORECASE
        )
        if match:
            keyword = match.group(1).capitalize()
            number = match.group(2).upper()
            return f"{keyword} {number}"

        # Match numbered sub-clause like 47.1
        match = re.match(r'^(\d+\.\d+(\.\d+)*)', text)
        if match:
            return f"Clause {match.group(1)}"

        # Return first 50 chars of heading as reference
        return text[:50] if len(text) > 50 else text

    def _determine_chunk_type(self, element: DocumentElement) -> str:
        """
        Determine the chunk type from the first element.

        Returns:
            "part"      for level 1 headings (PART/CHAPTER)
            "article"   for level 2 headings (ARTICLE/SECTION)
            "paragraph" for everything else
        """
        if element.element_type == "heading":
            if element.level == 1:
                return "part"
            elif element.level == 2:
                return "article"
        return "paragraph"

    def _generate_doc_id(self, file_name: str) -> str:
        """
        Generate a stable document ID from the file name.
        Uses MD5 hash to ensure unique but consistent IDs.

        Example:
            "contract_abc.pdf" → "contract_abc_a1b2c3d4"
        """
        name_without_ext = file_name.rsplit(".", 1)[0]
        hash_suffix = hashlib.md5(
            file_name.encode()
        ).hexdigest()[:8]
        # Clean name: only alphanumeric and underscores
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name_without_ext)
        return f"{clean_name}_{hash_suffix}"


# ─── Convenience Function ────────────────────────────────────

from pathlib import Path


def chunk_document(
    doc: ParsedDocument,
    max_chunk_size: int = None,
    chunk_overlap: int = None,
) -> ChunkingResult:
    """
    Convenience function to chunk a document in one line.

    Usage:
        from src.ingestion.chunker import chunk_document
        result = chunk_document(parsed_doc)
        print(result.total_chunks)
    """
    chunker = LegalChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk(doc)


# ─── Quick Test ──────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os
    from src.ingestion.parser import parse_document

    sample = """PART I - GENERAL PROVISIONS

ARTICLE 1 DEFINITIONS
In this Agreement the following terms shall have meanings:

1.1 Agreement means this contract dated January 2024.
1.2 Party means either ABC Corporation or XYZ Limited.

ARTICLE 2 OBLIGATIONS
ABC Corporation shall perform the following:

2.1 Deliver services within 30 days of the Effective Date.
2.2 Maintain confidentiality of all shared information.

ARTICLE 3 PAYMENT TERMS
XYZ Limited shall pay PKR 5000000 within 30 days.

3.1 Late payments attract 2 percent monthly interest.
3.2 Disputes must be raised within 14 days of invoice.
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample)
        tmp = f.name

    # Parse then chunk
    parsed = parse_document(tmp)
    result = chunk_document(parsed)

    print(f"Document : {result.file_name}")
    print(f"Doc ID   : {result.doc_id}")
    print(f"Chunks   : {result.total_chunks}")
    print(f"Articles : {result.articles_found}")
    print(f"Total chars: {result.total_chars}")
    print()

    for chunk in result.chunks:
        print(f"Chunk {chunk.chunk_index}: [{chunk.chunk_type.upper()}] "
              f"article_ref='{chunk.article_ref}' "
              f"chars={chunk.char_count}")
        print(f"  Preview: {chunk.text[:80]}...")
        print()

    os.unlink(tmp)

    passed = result.total_chunks >= 3 and result.articles_found >= 3
    print("Chunker test PASSED" if passed else "Chunker test FAILED")
