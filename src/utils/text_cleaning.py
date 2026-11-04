"""
Text cleaning and preprocessing utilities for YouTube & Reddit sentiment analysis.
Provides robust cleaning functions for URLs, user mentions, special characters,
HTML tags, emojis, and whitespace normalization.
"""

import html
import re
from typing import Optional

# Common emojis to sentiment words/tokens mapping
EMOJI_SENTIMENT_MAP = {
    # Positive emojis
    "❤️": " love ",
    "💖": " love ",
    "💕": " love ",
    "😍": " love ",
    "🥰": " love ",
    "😊": " happy ",
    "😀": " happy ",
    "😃": " happy ",
    "😄": " happy ",
    "😁": " happy ",
    "🎉": " great ",
    "👍": " good ",
    "🔥": " awesome ",
    "✨": " amazing ",
    "💯": " perfect ",
    "👏": " bravo ",
    "🙌": " praise ",
    "🤩": " excited ",
    "😎": " cool ",
    "💪": " strong ",

    # Negative emojis
    "😢": " sad ",
    "😭": " crying ",
    "😞": " disappointed ",
    "😔": " sad ",
    "😡": " angry ",
    "😠": " mad ",
    "🤬": " furious ",
    "👎": " bad ",
    "💔": " heartbroken ",
    "🤮": " disgusting ",
    "🤢": " gross ",
    "💩": " crap ",
    "🙄": " annoyed ",
    "🤦": " facepalm ",
    "🤦‍♂️": " facepalm ",
    "🤦‍♀️": " facepalm ",
    "💀": " dead ",
    "☠️": " bad ",

    # Neutral / thought emojis
    "🤔": " thinking ",
    "😐": " neutral ",
    "😶": " speechless ",
}

# Pre-compiled regex patterns for maximum efficiency
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"(@[\w]+|u/[\w-]+|r/[\w-]+)")
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s\.\?!,']|_")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")


def clean_text(text: Optional[str], preserve_emojis: bool = True) -> str:
    """
    Cleans a text comment for sentiment analysis:
    - Decodes HTML entities (&amp; -> &, etc.)
    - Removes URLs
    - Removes user/subreddit mentions (@user, u/user, r/subreddit)
    - Maps or normalizes emojis to semantic text tokens
    - Strips unwanted HTML tags and special characters
    - Normalizes repeated characters (e.g. 'sooooo goooood' -> 'soo good')
    - Collapses multiple whitespace characters into a single space

    Args:
        text: Raw input string or None
        preserve_emojis: If True, maps recognized emojis to sentiment terms

    Returns:
        Cleaned text string
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Strip HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 3. Remove URLs
    text = URL_PATTERN.sub(" ", text)

    # 4. Remove user and subreddit mentions (@user, u/username, r/subreddit)
    text = MENTION_PATTERN.sub(" ", text)

    # 5. Emoji conversion/handling
    if preserve_emojis:
        for emoji_char, replacement in EMOJI_SENTIMENT_MAP.items():
            if emoji_char in text:
                text = text.replace(emoji_char, replacement)

    # 6. Reduce elongated repeated letters (e.g., 'looooove' -> 'loove')
    text = REPEAT_CHAR_PATTERN.sub(r"\1\1", text)

    # 7. Remove strange special characters but preserve standard punctuation
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)

    # 8. Collapse whitespace and strip
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text.lower()


def label_to_sentiment_name(label: int) -> str:
    """Converts numeric label (-1, 0, 1) to human-readable string."""
    mapping = {
        -1: "negative",
        0: "neutral",
        1: "positive",
    }
    return mapping.get(int(label), "unknown")


def sentiment_name_to_label(name: str) -> int:
    """Converts sentiment name string to numeric label (-1, 0, 1)."""
    mapping = {
        "negative": -1,
        "neutral": 0,
        "positive": 1,
    }
    return mapping.get(name.lower().strip(), 0)
