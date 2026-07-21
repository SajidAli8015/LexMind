"""
Document Title Extractor for LexMind

Attempts to extract a meaningful document title using
a three-step cascade:

Step 1 — Parse: read first few elements, look for a clear heading
Step 2 — LLM:   if unclear, ask the LLM to identify the document
Step 3 — Fallback: use cleaned filename as last resort

This runs during ingestion so every chunk gets a doc_title
stored in its ChromaDB metadata.
"""

import re
from typing import Optional
from loguru import logger


def extract_title_from_elements(elements: list) -> Optional[str]:
    """
    Try to find a clear title from the first few parsed elements.
    Stops at the FIRST match — does not concatenate multiple lines.
    """
    if not elements:
        return None

    LEGAL_KEYWORDS = [
        'law', 'act', 'regulation', 'agreement', 'contract',
        'policy', 'code', 'statute', 'ordinance', 'decree',
        'bylaw', 'charter', 'constitution', 'treaty', 'convention',
        'directive', 'order', 'rules', 'system', 'نظام', 'labor',
        'labour', 'employment', 'companies', 'commercial', 'civil',
    ]

    SKIP_PATTERNS = [
        r'^\d{1,4}$',                          # page numbers
        r'^page\s+\d+',                         # "page 1"
        r'in the name of',                      # Bismillah translation
        r'most gracious',                       # Bismillah translation
        r'most merciful',                       # Bismillah translation
        r'بسم الله',                            # Bismillah Arabic
        r'الرحمن الرحيم',                       # Bismillah Arabic
        r'^\d{4}\s*h$',                        # year like "1426 H"
    ]

    for element in elements[:8]:
        text = element.text.strip()

        # Skip very short or very long lines
        if len(text) < 4 or len(text) > 120:
            continue

        # Skip lines matching skip patterns
        skip = False
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue

        text_lower = text.lower()

        # Level 1 or 2 heading with legal keyword — strong signal
        if element.element_type == 'heading' and element.level in (1, 2):
            if any(kw in text_lower for kw in LEGAL_KEYWORDS):
                logger.info(f"Title from heading: '{text}'")
                return _clean_title(text)

        # All-caps short line with legal keyword
        if text.isupper() and 4 < len(text) < 80:
            if any(kw in text_lower for kw in LEGAL_KEYWORDS):
                logger.info(f"Title from caps: '{text}'")
                return _clean_title(text)

        # Any line with legal keyword that is short enough to be a title
        if len(text) < 80 and any(kw in text_lower for kw in LEGAL_KEYWORDS):
            logger.info(f"Title from keyword: '{text}'")
            return _clean_title(text)

    return None


def extract_title_with_llm(
    elements: list,
    llm=None,
) -> Optional[str]:
    """
    Step 2: Ask the LLM to identify the document from its first content.

    Only called if Step 1 failed. Sends the first 800 characters
    of the document to the LLM with a short prompt.

    Args:
        elements: List of DocumentElement objects
        llm:      LangChain chat model (from llm_client.get_llm())

    Returns:
        LLM-identified title, or None if LLM fails
    """
    if not elements or llm is None:
        return None

    # Build a text sample from the first elements
    sample_lines = []
    total_chars = 0
    for element in elements[:10]:
        text = element.text.strip()
        if text and total_chars < 800:
            sample_lines.append(text)
            total_chars += len(text)

    if not sample_lines:
        return None

    sample_text = "\n".join(sample_lines)

    prompt = f"""Read the following excerpt from the beginning of a legal document.
Identify what this document is called — its official title or common name.

Reply with ONLY the document title — nothing else.
Keep it short (maximum 8 words).
If you cannot determine the title, reply with: UNKNOWN

Document excerpt:
{sample_text}

Document title:"""

    try:
        response = llm.invoke(prompt)
        title = response.content.strip()

        if not title or title.upper() == "UNKNOWN" or len(title) > 100:
            logger.warning(f"LLM could not identify document title")
            return None

        # Sanity check — reject if LLM returned something odd
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]

        logger.info(f"Title identified by LLM: '{title}'")
        return _clean_title(title)

    except Exception as e:
        logger.warning(f"LLM title extraction failed: {e}")
        return None


def title_from_filename(file_name: str) -> str:
    """
    Step 3 (fallback): Generate a readable title from the filename.

    Examples:
        "saudi_labor_law_2024.pdf"  → "Saudi Labor Law 2024"
        "nda_agreement_v2.docx"     → "Nda Agreement V2"
        "Copy of نظام العمل.pdf"    → "نظام العمل"  (removes "Copy of")

    Args:
        file_name: The uploaded filename

    Returns:
        A cleaned, readable title string
    """
    # Remove extension
    name = file_name.rsplit('.', 1)[0]

    # Remove common prefixes like "Copy of", "Final -", etc.
    name = re.sub(r'^(copy\s+of\s+|final\s*[-_]?\s*|draft\s*[-_]?\s*)',
                  '', name, flags=re.IGNORECASE).strip()

    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')

    # Remove extra whitespace
    name = ' '.join(name.split())

    # Title case if all ASCII (don't title-case Arabic)
    if name.isascii():
        name = name.title()

    logger.info(f"Title from filename: '{name}'")
    return name if name else file_name


def detect_document_title(
    elements: list,
    file_name: str,
    llm=None,
    user_provided: Optional[str] = None,
) -> str:
    """
    Main entry point. Runs the three-step cascade.

    Priority order:
    1. User-provided title (if user typed one at upload time)
    2. Auto-extracted from document structure (Step 1)
    3. LLM identification (Step 2) — only if llm is provided
    4. Cleaned filename (Step 3 — always works)

    Args:
        elements:       ParsedDocument elements
        file_name:      Original uploaded filename
        llm:            Optional LangChain chat model
        user_provided:  Title typed by user at upload time

    Returns:
        Best available title string — never None or empty
    """
    # Priority 1: user provided a title
    if user_provided and user_provided.strip():
        title = user_provided.strip()
        logger.info(f"Using user-provided title: '{title}'")
        return title

    # Priority 2: extract from document structure
    title = extract_title_from_elements(elements)
    if title:
        return title

    # Priority 3: ask the LLM
    if llm is not None:
        title = extract_title_with_llm(elements, llm)
        if title:
            return title

    # Priority 4: clean filename fallback
    return title_from_filename(file_name)


def _clean_title(title: str) -> str:
    """
    Clean and normalise a detected title string.
    - Strip whitespace
    - Remove trailing punctuation (except closing brackets)
    - Collapse multiple spaces
    - Limit to 150 characters
    """
    title = title.strip()
    title = ' '.join(title.split())
    title = title.rstrip('.,;:')
    title = title[:150]
    return title
