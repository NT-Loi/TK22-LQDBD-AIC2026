import os
import re
import logging
from functools import lru_cache
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Vietnamese accented characters (both lower and upper case)
VI_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
)

# Common English words frequently appearing in video descriptions & queries
COMMON_EN_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "and", "or", "is", "are",
    "was", "were", "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now", "person", "man",
    "woman", "people", "group", "car", "cars", "holding", "standing", "walking", "running",
    "wearing", "sitting", "looking", "talking", "speaking", "playing", "driving", "riding",
    "view", "close-up", "closeup", "shot", "scene", "background", "outdoor", "outdoors",
    "indoor", "indoors", "red", "blue", "green", "white", "black", "yellow", "two", "three",
    "one", "four", "five", "several", "front", "behind", "side", "chef", "cutting", "vegetables",
    "dam", "drone", "colliding", "highway", "street", "city", "room", "water", "sky", "sea",
    "beach", "crowd", "stage", "camera", "hand", "hands", "face", "hair", "dog", "cat", "animal",
    "eating", "drinking", "cooking", "jumping", "dancing", "exercising", "toes", "feet", "head"
}

# Common Vietnamese words without accents (to prevent false positive English detection)
COMMON_VI_UNACCENTED = {
    "nguoi", "trong", "dang", "nhom", "mot", "nhung", "co", "tren", "duoi", "va", "cua",
    "cho", "canh", "quay", "chiec", "con", "tai", "vao", "den", "khi", "khong", "thi",
    "nhieu", "it", "ao", "quan", "xe", "nha", "duong", "bien", "troi", "tay", "chan",
    "phat", "bieu", "tap", "duc", "nam", "nu", "dung", "ngoi", "di", "chay", "an", "uong"
}

_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
            project_id = os.environ.get("PROJECT_ID", "")
            location = os.environ.get("VERTEX_LOCATION", "global")
            _gemini_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Gemini Client for translation: {e}")
            _gemini_client = False
    return _gemini_client if _gemini_client is not False else None


def is_english_query(text: str) -> bool:
    """
    Fast heuristic to check whether a search query is primarily English.
    Returns False immediately if any Vietnamese diacritical marks are found.
    """
    if not text or not isinstance(text, str):
        return False

    # 1. Contains Vietnamese accented characters -> Definitely Vietnamese
    if any(c in VI_DIACRITICS for c in text):
        return False

    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if not tokens:
        return False

    # 2. Count known tokens
    vi_count = sum(1 for t in tokens if t in COMMON_VI_UNACCENTED)
    en_count = sum(1 for t in tokens if t in COMMON_EN_WORDS)

    if vi_count > en_count:
        return False

    if en_count > 0:
        return True

    # If all tokens are ASCII alphabet words without Vietnamese hints, treat as English
    return len(tokens) >= 2


@lru_cache(maxsize=2048)
def translate_en_to_vi(text: str) -> str:
    """
    Translates an English video retrieval query into natural Vietnamese suitable for
    matching with Vietnamese shot captions. Caches translations for performance.
    """
    clean_text = text.strip()
    if not clean_text:
        return clean_text

    client = _get_gemini_client()
    if client is not None:
        try:
            from google.genai import types
            model_id = os.environ.get("MODEL_ID", "gemini-2.5-flash-lite")
            prompt = (
                "Translate this English video retrieval search query into natural Vietnamese "
                "for matching video captions. Keep visual terms accurate (e.g. cận cảnh, góc nhìn trên cao). "
                f"Return ONLY the translated Vietnamese text, without any quotes or explanations:\n{clean_text}"
            )
            response = client.models.generate_content(
                model=model_id,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=100,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            translated = response.text.strip().strip('"\'')
            if translated:
                logger.info(f"Gemini translated: '{clean_text}' -> '{translated}'")
                return translated
        except Exception as e:
            logger.warning(f"Gemini translation failed: {e}. Falling back...")

    # Fallback: deep-translator if installed
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target="vi").translate(clean_text)
        if translated:
            return translated.strip()
    except Exception:
        pass

    return clean_text


def translate_query_to_vi_if_needed(text: str) -> Tuple[str, bool]:
    """
    Detects if the query is in English. If so, translates to Vietnamese.
    Returns:
        (query_to_use, was_translated: bool)
    """
    if is_english_query(text):
        translated = translate_en_to_vi(text)
        if translated and translated != text:
            return translated, True
    return text, False
