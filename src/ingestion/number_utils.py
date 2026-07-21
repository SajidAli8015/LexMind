"""
Word-to-number converter for legal article citations.
Converts written numbers like "Seventy-Five" to digits like "75"
so citation highlighting works consistently across all documents.
"""

import re

# Basic number words mapping
ONES = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19,
}

TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}

MULTIPLIERS = {
    'hundred': 100,
    'thousand': 1000,
}


def words_to_number(text: str) -> int:
    """
    Convert a written number string to an integer.

    Examples:
        "seventy-five"        → 75
        "one hundred"         → 100
        "one hundred and two" → 102
        "thirty"              → 30
        "twelve"              → 12

    Args:
        text: Written number string (case-insensitive)

    Returns:
        Integer value, or -1 if conversion fails
    """
    text = text.lower().strip()
    text = text.replace('-', ' ').replace(' and ', ' ')
    words = text.split()

    if not words:
        return -1

    total = 0
    current = 0

    for word in words:
        if word in ONES:
            current += ONES[word]
        elif word in TENS:
            current += TENS[word]
        elif word == 'hundred':
            current = current * 100 if current > 0 else 100
        elif word == 'thousand':
            current = current * 1000 if current > 0 else 1000
            total += current
            current = 0
        else:
            return -1

    total += current
    return total if total > 0 else -1


def normalise_article_citations(text: str) -> str:
    """
    Convert written article numbers to numeric format in text.

    Finds patterns like:
        "Article Seventy-Five"     → "Article 75"
        "Article One Hundred"      → "Article 100"
        "Section Thirty"           → "Section 30"
        "Clause Twelve"            → "Clause 12"

    Does NOT change already-numeric citations:
        "Article 75"               → "Article 75" (unchanged)

    Args:
        text: Answer text from the Reasoning Agent

    Returns:
        Text with written article numbers replaced by digits
    """
    # Pattern: Article/Section/Clause followed by written number words
    # Matches: "Article Seventy-Five" or "Article One Hundred and Two"
    pattern = (
        r'\b(Article|Section|Clause|Schedule|Part)\s+'
        r'([A-Z][a-z]+(?:[\s\-][A-Z][a-z]+)*)'
        r'(?=\s*[\[\.,\s]|$)'
    )

    def replace_match(match):
        prefix = match.group(1)    # "Article"
        number_text = match.group(2)  # "Seventy-Five"

        # Skip if it's already a digit
        if number_text[0].isdigit():
            return match.group(0)

        # Skip short words that are clearly not numbers
        # (e.g. "Article Ten" is a number but "Article One" too)
        num = words_to_number(number_text)
        if num == -1:
            return match.group(0)  # Could not convert — leave as-is

        return f"{prefix} {num}"

    return re.sub(pattern, replace_match, text)


def normalise_citations_in_brackets(text: str) -> str:
    """
    Specifically normalise written numbers inside citation brackets.

    Targets patterns like:
        [Article Seventy-Five — Labor Law]
        [Section Twenty — Companies Law]

    Args:
        text: Answer text containing bracketed citations

    Returns:
        Text with normalised numbers inside brackets
    """
    # Match content inside square brackets that starts with
    # Article/Section/Clause
    bracket_pattern = (
        r'\[((?:Article|Section|Clause|Schedule|Part)'
        r'\s+[A-Za-z\s\-]+(?:\s*[—–-]\s*[^\]]+)?)\]'
    )

    def replace_bracket(match):
        content = match.group(1)
        normalised = normalise_article_citations(content)
        return f"[{normalised}]"

    return re.sub(bracket_pattern, replace_bracket, text)
