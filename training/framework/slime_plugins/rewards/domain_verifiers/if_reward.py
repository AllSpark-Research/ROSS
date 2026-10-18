from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

from .common import ensure_text, mean, strip_trailing_special_tokens

JsonDict = dict[str, Any]
KwargsDict = dict[str, Any]


InstructionVerifier = Callable[["VerificationContext", KwargsDict], bool]
INSTRUCTION_REGISTRY: dict[str, InstructionVerifier] = {}


def register_instruction(*instruction_ids: str) -> Callable[[InstructionVerifier], InstructionVerifier]:
    def decorator(fn: InstructionVerifier) -> InstructionVerifier:
        for instruction_id in instruction_ids:
            if instruction_id in INSTRUCTION_REGISTRY:
                raise ValueError(f"Duplicate IF instruction registration: {instruction_id}")
            INSTRUCTION_REGISTRY[instruction_id] = fn
        return fn

    return decorator


def _normalize_instruction_ids(raw_ids: Any) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized: list[str] = []
    for item in raw_ids:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_kwargs_list(raw_kwargs: Any, num_instructions: int) -> list[KwargsDict]:
    if isinstance(raw_kwargs, list):
        kwargs_list = [dict(item) if isinstance(item, dict) else {} for item in raw_kwargs]
    elif isinstance(raw_kwargs, dict):
        kwargs_list = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        kwargs_list = [{} for _ in range(num_instructions)]

    if len(kwargs_list) < num_instructions:
        kwargs_list.extend({} for _ in range(num_instructions - len(kwargs_list)))
    return kwargs_list[:num_instructions]


def _clean_response_text(text: Any) -> str:
    return strip_trailing_special_tokens(ensure_text(text)).strip()


def _normalize_relation(value: Any) -> str:
    return str(value or "").strip().lower()


def _compare_relation(actual: float, expected: float, relation: str) -> bool:
    normalized = _normalize_relation(relation)
    if normalized in {"less than", "lt", "<"}:
        return actual < expected
    if normalized in {"at least", "gte", ">=", "greater than or equal to"}:
        return actual >= expected
    if normalized in {"at most", "lte", "<=", "less than or equal to"}:
        return actual <= expected
    if normalized in {"equal", "equal to", "=="}:
        return actual == expected
    raise ValueError(f"Unsupported comparison relation: {relation!r}")


WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
PLACEHOLDER_RE = re.compile(r"\[[^\[\]\n]+\]")
TITLE_RE = re.compile(r"<<\s*[^<>\n].*?\s*>>", re.DOTALL)
HIGHLIGHT_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!\*)\*\*([^*\n]+)\*\*(?!\*)")
SECTION_MARKER_RE = re.compile(r"^\s*([A-Z][A-Z0-9 _-]*?)\s+(\d+)\s*$", re.MULTILINE)
BIGRAM_BLOCK_RE = re.compile(r"<<\s*([^<>]+?)\s*>>")
PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
CONSTRAINED_OPTIONS_RE = re.compile(r"one of the following options:\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _normalize_word(token: str) -> str:
    token = PUNCT_STRIP_RE.sub("", ensure_text(token).strip()).lower()
    return token


def _extract_words(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(text)]


def _extract_normalized_words(text: str) -> list[str]:
    return [_normalize_word(word) for word in _extract_words(text) if _normalize_word(word)]


def _count_word_occurrences(text: str, keyword: str) -> int:
    target = _normalize_word(keyword)
    if not target:
        return 0
    return sum(1 for word in _extract_normalized_words(text) if word == target)


def _count_letter_occurrences(text: str, letter: str) -> int:
    target = ensure_text(letter).lower()
    if not target:
        return 0
    return sum(1 for char in ensure_text(text).lower() if char == target)


def _strip_outer_quotes(text: str) -> str:
    stripped = ensure_text(text).strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        return stripped[1:-1].strip()
    return stripped


def _split_on_divider(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"\n\s*(?:\*{3}|\*\s+\*\s+\*)\s*\n", text)
    return [part.strip() for part in parts if part.strip()]


def _split_on_blank_lines(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?:\r?\n\s*){2,}", text)
    return [part.strip() for part in parts if part.strip()]


def _extract_lines(text: str) -> list[str]:
    return [line.strip() for line in ensure_text(text).splitlines() if line.strip()]


def _split_sentences(text: str) -> list[str]:
    raw = ensure_text(text).strip()
    if not raw:
        return []

    split = re.split(r"(?<=[.!?])(?:['\"”’)\]]*)\s+|\n+", raw)
    sentences = [part.strip() for part in split if part.strip()]
    if len(sentences) > 1:
        return sentences

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    return [raw]


def _count_cased_words(text: str, mode: str) -> int:
    count = 0
    for word in _extract_words(text):
        if word.lower() == word.upper():
            continue
        if mode == "lower" and word == word.lower():
            count += 1
        if mode == "upper" and word == word.upper():
            count += 1
    return count


def _has_only_case(text: str, mode: str) -> bool:
    has_cased = False
    for char in ensure_text(text):
        if not char.isalpha():
            continue
        if char.lower() == char.upper():
            continue
        has_cased = True
        if mode == "upper" and char != char.upper():
            return False
        if mode == "lower" and char != char.lower():
            return False
    return has_cased


def _contains_any_keywords(text: str, keywords: list[str]) -> bool:
    return all(_count_word_occurrences(text, keyword) > 0 for keyword in keywords)


def _extract_bracketed_placeholders(text: str) -> list[str]:
    return [match.group(0) for match in PLACEHOLDER_RE.finditer(text)]


def _extract_options_from_prompt(prompt_text: str) -> list[str]:
    match = CONSTRAINED_OPTIONS_RE.search(prompt_text)
    if not match:
        return []
    content = match.group(1)
    return [option.strip() for option in re.findall(r"'([^']+)'|\"([^\"]+)\"", content) for option in option if option]


def _parse_json_payload(text: str) -> Any:
    raw = ensure_text(text).strip()
    if not raw:
        return None
    fenced = JSON_FENCE_RE.fullmatch(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _count_bullet_lines(text: str) -> int:
    return sum(1 for line in _extract_lines(text) if re.match(r"^[-*]\s+\S+", line))


def _extract_section_markers(text: str) -> list[tuple[str, int]]:
    markers: list[tuple[str, int]] = []
    for match in SECTION_MARKER_RE.finditer(text):
        markers.append((match.group(1).strip(), int(match.group(2))))
    return markers


def _get_nonempty_lines_excluding_markers(text: str) -> list[str]:
    lines = []
    for line in _extract_lines(text):
        if SECTION_MARKER_RE.match(line):
            continue
        if re.fullmatch(r"(?:\*{3}|\*\s+\*\s+\*)", line):
            continue
        lines.append(line)
    return lines


def _token_is_palindrome(token: str) -> bool:
    normalized = _normalize_word(token)
    return len(normalized) >= 3 and normalized == normalized[::-1]


def _script_ratio(text: str, ranges: list[tuple[int, int]]) -> float:
    total = 0
    matched = 0
    for char in ensure_text(text):
        if char.isspace() or char.isdigit():
            continue
        category = unicodedata.category(char)
        if not category.startswith("L"):
            continue
        total += 1
        codepoint = ord(char)
        if any(start <= codepoint <= end for start, end in ranges):
            matched += 1
    if total == 0:
        return 0.0
    return matched / total


SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "ar": [(0x0600, 0x06FF)],
    "bg": [(0x0400, 0x04FF)],
    "bn": [(0x0980, 0x09FF)],
    "fa": [(0x0600, 0x06FF)],
    "gu": [(0x0A80, 0x0AFF)],
    "he": [(0x0590, 0x05FF)],
    "hi": [(0x0900, 0x097F)],
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
    "kn": [(0x0C80, 0x0CFF)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "ml": [(0x0D00, 0x0D7F)],
    "mr": [(0x0900, 0x097F)],
    "ne": [(0x0900, 0x097F)],
    "pa": [(0x0A00, 0x0A7F)],
    "ru": [(0x0400, 0x04FF)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "th": [(0x0E00, 0x0E7F)],
    "uk": [(0x0400, 0x04FF)],
    "ur": [(0x0600, 0x06FF)],
}


LATIN_LANGUAGE_STOPWORDS: dict[str, set[str]] = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "ein"},
    "en": {"the", "and", "is", "of", "to", "in", "that", "with"},
    "es": {"el", "la", "los", "las", "y", "de", "que", "para"},
    "fi": {"ja", "on", "se", "että", "ei", "kun", "tai", "voi"},
    "fr": {"le", "la", "les", "et", "de", "des", "que", "pour"},
    "it": {"il", "la", "gli", "le", "e", "di", "che", "per"},
    "pl": {"i", "że", "na", "się", "jest", "nie", "to", "w"},
    "pt": {"o", "a", "os", "as", "e", "de", "que", "para"},
    "sw": {"na", "ya", "kwa", "katika", "ni", "si", "wa", "za"},
    "vi": {"và", "là", "của", "cho", "trong", "một", "những", "được"},
}


NON_LATIN_LANGUAGE_HINTS: dict[str, set[str]] = {
    "ar": {"ال", "في", "من", "على"},
    "bg": {"и", "в", "на", "че", "за"},
    "bn": {"এবং", "এই", "একটি", "জন্য"},
    "fa": {"و", "در", "به", "از"},
    "gu": {"અને", "છે", "માટે"},
    "he": {"של", "את", "עם", "על"},
    "hi": {"और", "है", "के", "में"},
    "ja": {"です", "ます", "の", "に", "は"},
    "kn": {"ಮತ್ತು", "ಇದು", "ನಲ್ಲಿ"},
    "ko": {"그리고", "입니다", "에서", "하는"},
    "ml": {"എന്ന", "ഈ", "ഒരു", "മറ്റും"},
    "mr": {"आहे", "आणि", "मध्ये", "एक"},
    "ne": {"र", "मा", "को", "छ"},
    "pa": {"ਅਤੇ", "ਇਹ", "ਦੇ", "ਵਿੱਚ"},
    "ru": {"и", "в", "не", "на", "что"},
    "ta": {"மற்றும்", "இந்த", "ஒரு", "இல்"},
    "te": {"మరియు", "ఈ", "ఒక", "లో"},
    "th": {"และ", "ใน", "ของ", "ที่"},
    "uk": {"і", "в", "не", "на", "що"},
    "ur": {"اور", "میں", "ہے", "کے"},
}


def _language_score(text: str, language: str) -> float:
    language = str(language or "").lower()
    normalized = ensure_text(text)
    lowered_words = _extract_normalized_words(normalized.lower())

    score = 0.0
    if language in SCRIPT_RANGES:
        ratio = _script_ratio(normalized, SCRIPT_RANGES[language])
        score += 3.0 * ratio
        hints = NON_LATIN_LANGUAGE_HINTS.get(language, set())
        hint_hits = sum(1 for hint in hints if hint in normalized)
        score += min(hint_hits, 3)
        return score

    stopwords = LATIN_LANGUAGE_STOPWORDS.get(language, set())
    if not stopwords:
        return 0.0

    stopword_hits = sum(1 for word in lowered_words if word in stopwords)
    latin_chars = sum(
        1
        for char in normalized
        if char.isalpha() and "LATIN" in unicodedata.name(char, "")
    )
    non_latin_letters = sum(
        1
        for char in normalized
        if char.isalpha() and "LATIN" not in unicodedata.name(char, "")
    )
    if latin_chars > 0 and non_latin_letters == 0:
        score += 1.0
    score += stopword_hits
    return score


def _is_target_language(text: str, language: str) -> bool:
    language = str(language or "").lower()
    if not language:
        return False

    target_score = _language_score(text, language)
    if language in SCRIPT_RANGES:
        return target_score >= 1.8

    all_latin_scores = {lang: _language_score(text, lang) for lang in LATIN_LANGUAGE_STOPWORDS}
    best_lang = max(all_latin_scores, key=all_latin_scores.get, default="")
    return target_score >= 2.0 and best_lang == language


@dataclass(slots=True)
class RuleEvaluation:
    instruction_id: str
    passed: bool


@dataclass
class VerificationContext:
    response: str
    prompt_text: str

    @cached_property
    def stripped_response(self) -> str:
        return _strip_outer_quotes(self.response)

    @cached_property
    def words(self) -> list[str]:
        return _extract_words(self.stripped_response)

    @cached_property
    def normalized_words(self) -> list[str]:
        return _extract_normalized_words(self.stripped_response)

    @cached_property
    def sentences(self) -> list[str]:
        return _split_sentences(self.stripped_response)

    @cached_property
    def sentence_words(self) -> list[list[str]]:
        return [_extract_words(sentence) for sentence in self.sentences]

    @cached_property
    def markdown_divider_paragraphs(self) -> list[str]:
        return _split_on_divider(self.stripped_response)

    @cached_property
    def blank_line_paragraphs(self) -> list[str]:
        return _split_on_blank_lines(self.stripped_response)

    @cached_property
    def generic_paragraphs(self) -> list[str]:
        if self.markdown_divider_paragraphs:
            return self.markdown_divider_paragraphs
        return self.blank_line_paragraphs


@register_instruction("keywords:existence")
def verify_keywords_existence(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    keywords = [str(item) for item in kwargs.get("keywords", []) if str(item).strip()]
    return bool(keywords) and _contains_any_keywords(ctx.stripped_response, keywords)


@register_instruction("keywords:forbidden_words")
def verify_keywords_forbidden_words(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    forbidden_words = [str(item) for item in kwargs.get("forbidden_words", []) if str(item).strip()]
    return all(_count_word_occurrences(ctx.stripped_response, word) == 0 for word in forbidden_words)


@register_instruction("keywords:word_once")
def verify_keywords_word_once(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    return _count_word_occurrences(ctx.stripped_response, kwargs.get("keyword")) == 1


@register_instruction("keywords:keyword_specific_position")
def verify_keywords_specific_position(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    sentence_index = int(kwargs.get("n", 0))
    word_index = int(kwargs.get("m", 0))
    keyword = _normalize_word(kwargs.get("keyword"))
    if sentence_index <= 0 or word_index <= 0 or not keyword:
        return False
    if sentence_index > len(ctx.sentence_words):
        return False
    sentence_words = [_normalize_word(word) for word in ctx.sentence_words[sentence_index - 1] if _normalize_word(word)]
    if word_index > len(sentence_words):
        return False
    return sentence_words[word_index - 1] == keyword


@register_instruction("keywords:frequency", "keywords:word_count_different_numbers")
def verify_keywords_frequency(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    keyword = kwargs.get("keyword")
    expected = int(kwargs.get("frequency", 0))
    relation = kwargs.get("relation", "equal")
    actual = _count_word_occurrences(ctx.stripped_response, keyword)
    return _compare_relation(actual, expected, relation)


@register_instruction("keywords:letter_frequency", "letters:letter_counting2")
def verify_letter_frequency(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("let_frequency", 0))
    relation = kwargs.get("let_relation", "equal")
    actual = _count_letter_occurrences(ctx.stripped_response, kwargs.get("letter"))
    return _compare_relation(actual, expected, relation)


@register_instruction("keywords:no_adjacent_consecutive")
def verify_no_adjacent_consecutive(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    initials: list[str] = []
    for word in ctx.normalized_words:
        if word and "a" <= word[0] <= "z":
            initials.append(word[0])
    for left, right in zip(initials, initials[1:], strict=False):
        if abs(ord(left) - ord(right)) == 1:
            return False
    return True


@register_instruction("keywords:palindrome")
def verify_palindrome(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return any(_token_is_palindrome(word) for word in ctx.words)


@register_instruction("keywords:start_end")
def verify_start_end(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    words = ctx.normalized_words
    if not words:
        return False
    last_word = words[-1]
    stripped = ctx.stripped_response.rstrip()
    if not re.search(rf"{re.escape(last_word)}\s*$", stripped, re.IGNORECASE):
        return False
    return words[0] == last_word


@register_instruction("detectable_format:title")
def verify_detectable_title(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return TITLE_RE.search(ctx.stripped_response) is not None


@register_instruction("detectable_format:bigram_wrapping")
def verify_bigram_wrapping(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    blocks = BIGRAM_BLOCK_RE.findall(ctx.stripped_response)
    if not blocks:
        return False
    cleaned = BIGRAM_BLOCK_RE.sub("", ctx.stripped_response)
    if cleaned.strip():
        return False
    return all(len(_extract_words(block)) == 2 for block in blocks)


@register_instruction("detectable_format:number_bullet_lists")
def verify_number_bullet_lists(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_bullets", 0))
    return expected > 0 and _count_bullet_lines(ctx.stripped_response) == expected


@register_instruction("detectable_format:number_highlighted_sections")
def verify_number_highlighted_sections(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_highlights", 0))
    if expected <= 0:
        return False
    return len(HIGHLIGHT_RE.findall(ctx.stripped_response)) >= expected


@register_instruction("detectable_format:square_brackets")
def verify_square_brackets(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    stripped = ctx.stripped_response
    if not stripped:
        return False
    placeholders = PLACEHOLDER_RE.findall(stripped)
    if not placeholders:
        return False
    remainder = PLACEHOLDER_RE.sub(" ", stripped)
    return not WORD_RE.search(remainder)


@register_instruction("detectable_format:sentence_hyphens")
def verify_sentence_hyphens(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    lines = _get_nonempty_lines_excluding_markers(ctx.stripped_response)
    if not lines:
        return False
    return all(" " not in line and "-" in line for line in lines)


@register_instruction("detectable_format:multiple_sections")
def verify_multiple_sections(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_sections", 0))
    splitter = str(kwargs.get("section_spliter") or "SECTION").strip().upper()
    if expected <= 0:
        return False
    markers = _extract_section_markers(ctx.stripped_response)
    if len(markers) != expected:
        return False
    return all(name.upper() == splitter and index == position for position, (name, index) in enumerate(markers, start=1))


@register_instruction("detectable_format:json_format")
def verify_json_format(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return _parse_json_payload(ctx.stripped_response) is not None


@register_instruction("detectable_format:constrained_response")
def verify_constrained_response(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    options = _extract_options_from_prompt(ctx.prompt_text)
    if not options:
        return False
    response = ctx.stripped_response.strip()
    return response in options


@register_instruction("length_constraints:number_words")
def verify_num_words(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_words", 0))
    return _compare_relation(len(ctx.normalized_words), expected, kwargs.get("relation", "equal"))


@register_instruction("length_constraints:number_sentences")
def verify_num_sentences(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_sentences", 0))
    return _compare_relation(len(ctx.sentences), expected, kwargs.get("relation", "equal"))


@register_instruction("length_constraints:number_paragraphs")
def verify_num_paragraphs(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_paragraphs", 0))
    return expected > 0 and len(ctx.generic_paragraphs) == expected


@register_instruction("length_constraints:nth_paragraph_first_word")
def verify_nth_paragraph_first_word(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected_paragraphs = int(kwargs.get("num_paragraphs", 0))
    nth_paragraph = int(kwargs.get("nth_paragraph", 0))
    first_word = _normalize_word(kwargs.get("first_word"))
    paragraphs = ctx.generic_paragraphs
    if expected_paragraphs > 0 and len(paragraphs) != expected_paragraphs:
        return False
    if nth_paragraph <= 0 or nth_paragraph > len(paragraphs) or not first_word:
        return False
    paragraph_words = _extract_normalized_words(paragraphs[nth_paragraph - 1])
    return bool(paragraph_words) and paragraph_words[0] == first_word


@register_instruction("count:lowercase_counting")
def verify_lowercase_counting(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("N", 0))
    return _count_cased_words(ctx.stripped_response, "lower") <= expected


@register_instruction("count:count_increment_word")
def verify_count_increment_word(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    return (
        _count_word_occurrences(ctx.stripped_response, kwargs.get("keyword1")) == 1
        and _count_word_occurrences(ctx.stripped_response, kwargs.get("keyword2")) == 2
    )


@register_instruction("count:count_unique")
def verify_count_unique(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    words = ctx.normalized_words
    return len(words) == len(set(words))


@register_instruction("count:counting_composition")
def verify_counting_composition(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected_sentences = int(kwargs.get("n_sent", 0))
    expected_words = int(kwargs.get("n_words", 0))
    paragraphs = ctx.markdown_divider_paragraphs or ctx.blank_line_paragraphs
    if not paragraphs or expected_sentences <= 0 or expected_words <= 0:
        return False
    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        if len(sentences) != expected_sentences:
            return False
        for sentence in sentences:
            if len(_extract_normalized_words(sentence)) != expected_words:
                return False
    return True


@register_instruction("punctuation:no_comma")
def verify_no_comma(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return "," not in ctx.stripped_response


@register_instruction("punctuation:punctuation_dot")
def verify_no_dot(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return "." not in ctx.stripped_response


@register_instruction("punctuation:punctuation_exclamation")
def verify_no_exclamation(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return "!" not in ctx.stripped_response


@register_instruction("change_case:english_capital")
def verify_english_capital(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return _has_only_case(ctx.stripped_response, "upper")


@register_instruction("change_case:english_lowercase")
def verify_english_lowercase(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return _has_only_case(ctx.stripped_response, "lower")


@register_instruction("change_case:capital_word_frequency")
def verify_capital_word_frequency(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("capital_frequency", 0))
    relation = kwargs.get("capital_relation", "equal")
    actual = _count_cased_words(ctx.stripped_response, "upper")
    return _compare_relation(actual, expected, relation)


@register_instruction("startend:end_checker")
def verify_end_checker(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    phrase = ensure_text(kwargs.get("end_phrase")).strip()
    if not phrase:
        return False
    return ctx.stripped_response.rstrip().endswith(phrase)


@register_instruction("startend:quotation")
def verify_quotation(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    stripped = ensure_text(ctx.response).strip()
    return len(stripped) >= 2 and stripped[0] == stripped[-1] == '"'


@register_instruction("first_word:first_word_answer")
def verify_first_word_answer(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = _normalize_word(kwargs.get("first_word"))
    return bool(ctx.normalized_words) and ctx.normalized_words[0] == expected


@register_instruction("first_word:first_word_sent")
def verify_first_word_sent(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = _normalize_word(kwargs.get("first_word"))
    if not expected or not ctx.sentence_words:
        return False
    for sentence in ctx.sentence_words:
        normalized = [_normalize_word(word) for word in sentence if _normalize_word(word)]
        if not normalized or normalized[0] != expected:
            return False
    return True


@register_instruction("last_word:last_word_answer")
def verify_last_word_answer(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = _normalize_word(kwargs.get("last_word"))
    return bool(ctx.normalized_words) and ctx.normalized_words[-1] == expected


@register_instruction("last_word:last_word_sent")
def verify_last_word_sent(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = _normalize_word(kwargs.get("last_word"))
    if not expected or not ctx.sentence_words:
        return False
    for sentence in ctx.sentence_words:
        normalized = [_normalize_word(word) for word in sentence if _normalize_word(word)]
        if not normalized or normalized[-1] != expected:
            return False
    return True


@register_instruction("paragraphs:paragraphs")
def verify_paragraphs_markdown(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return len(ctx.markdown_divider_paragraphs) == 2


@register_instruction("paragraphs:paragraphs2")
def verify_paragraphs_blank_lines(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    return len(ctx.blank_line_paragraphs) == 2


@register_instruction("detectable_content:number_placeholders")
def verify_number_placeholders(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("num_placeholders", 0))
    return expected > 0 and len(_extract_bracketed_placeholders(ctx.stripped_response)) >= expected


@register_instruction("detectable_content:postscript")
def verify_postscript(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    marker = ensure_text(kwargs.get("postscript_marker")).strip()
    if not marker:
        return False
    lines = _extract_lines(ctx.stripped_response)
    if not lines:
        return False
    return lines[-1].startswith(marker)


@register_instruction("letters:letter_counting")
def verify_letter_counting(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    expected = int(kwargs.get("N", 0))
    relation = kwargs.get("relation", "equal")
    actual = sum(1 for char in ctx.stripped_response if char.isalpha())
    return _compare_relation(actual, expected, relation)


@register_instruction("copy:repeat_phrase")
def verify_repeat_phrase(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    phrase = ensure_text(kwargs.get("phrase")).strip()
    expected = int(kwargs.get("small_n", 0))
    base_words = _extract_normalized_words(phrase)
    response_words = ctx.normalized_words
    if expected <= 0 or len(base_words) < 3:
        return False

    matched = 0
    for start in range(0, max(0, len(response_words) - len(base_words) + 1)):
        window = response_words[start : start + len(base_words)]
        if len(window) != len(base_words):
            continue
        diff_positions = [idx for idx, (left, right) in enumerate(zip(window, base_words, strict=False)) if left != right]
        if len(diff_positions) != 1:
            continue
        diff_index = diff_positions[0]
        if diff_index in {0, len(base_words) - 1}:
            continue
        matched += 1
    return matched == expected


@register_instruction("language:response_language")
def verify_response_language(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    return _is_target_language(ctx.stripped_response, kwargs.get("language"))


@register_instruction("combination:two_responses")
def verify_two_responses(ctx: VerificationContext, kwargs: KwargsDict) -> bool:
    del kwargs
    parts = [part.strip() for part in ctx.stripped_response.split("******") if part.strip()]
    return len(parts) == 2


def _evaluate_instruction(ctx: VerificationContext, instruction_id: str, kwargs: KwargsDict) -> RuleEvaluation:
    verifier = INSTRUCTION_REGISTRY.get(instruction_id)
    if verifier is None:
        raise ValueError(
            f"Unsupported Nemotron IF instruction_id {instruction_id!r}. "
            f"Supported ids: {sorted(INSTRUCTION_REGISTRY)}"
        )
    return RuleEvaluation(instruction_id=instruction_id, passed=bool(verifier(ctx, kwargs)))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: JsonDict | None = None,
) -> dict[str, Any]:
    del data_source, ground_truth

    extra_info = extra_info or {}
    if extra_info.get("overlong"):
        return {
            "score": 0.0,
            "strict_score": 0.0,
            "instruction_score": 0.0,
            "follow_all_instructions": False,
            "follow_instruction_list": [],
            "instruction_results": [],
            "num_instructions": 0,
            "num_followed": 0,
        }

    instruction_ids = _normalize_instruction_ids(extra_info.get("instruction_id_list"))
    if not instruction_ids:
        return {
            "score": 0.0,
            "strict_score": 0.0,
            "instruction_score": 0.0,
            "follow_all_instructions": False,
            "follow_instruction_list": [],
            "instruction_results": [],
            "num_instructions": 0,
            "num_followed": 0,
        }

    kwargs_list = _normalize_kwargs_list(extra_info.get("kwargs"), len(instruction_ids))
    ctx = VerificationContext(
        response=_clean_response_text(solution_str),
        prompt_text=ensure_text(extra_info.get("prompt_text")),
    )

    evaluations = [
        _evaluate_instruction(ctx, instruction_id, kwargs)
        for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=False)
    ]
    follow_list = [evaluation.passed for evaluation in evaluations]
    num_instructions = len(follow_list)
    num_followed = sum(1 for passed in follow_list if passed)
    instruction_score = mean([1.0 if passed else 0.0 for passed in follow_list])
    strict_score = 1.0 if all(follow_list) else 0.0

    aggregation = str(extra_info.get("aggregation", "strict"))
    if aggregation == "strict":
        score = strict_score
    elif aggregation == "instruction_avg":
        score = instruction_score
    elif aggregation == "avg_plus_all_bonus":
        all_bonus = float(extra_info.get("all_bonus", 0.2))
        score = min(instruction_score + (all_bonus if strict_score else 0.0), 1.0)
    else:
        raise ValueError(
            f"Unsupported IF reward aggregation: {aggregation!r}. "
            "Expected 'strict', 'instruction_avg', or 'avg_plus_all_bonus'."
        )

    return {
        "score": float(score),
        "strict_score": float(strict_score),
        "instruction_score": float(instruction_score),
        "follow_all_instructions": bool(strict_score),
        "follow_instruction_list": follow_list,
        "instruction_results": [
            {"instruction_id": evaluation.instruction_id, "passed": evaluation.passed}
            for evaluation in evaluations
        ],
        "num_instructions": num_instructions,
        "num_followed": num_followed,
    }
