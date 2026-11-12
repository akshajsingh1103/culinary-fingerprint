"""Deterministic, conservative, idempotent ingredient-name normalisation.

This single module is shared by the descriptive network builder, the
predictive (training-only) network builder, the Fingerprint Scorer and the
Naive Bayes baseline, so every component sees identical ingredient tokens.

Stages (applied in order, each logged in ``merge_log.csv`` when it changes a name):

1. ``unicode``    – NFKC normalisation, curly quotes -> ASCII quotes, removal of
                    trademark symbols (``®``, ``™``) and accent folding
                    (``é`` -> ``e``). These are encoding variants, not identity changes.
2. ``lowercase``  – case folding.
3. ``whitespace`` – strip and collapse repeated internal whitespace.
4. ``singularize``– the *final* token is singularised only if it appears in the
                    explicit whitelist :data:`PLURAL_WHITELIST` of unambiguous
                    count-noun plurals. No stemming; words such as ``cloves``,
                    ``leaves``, ``greens``, ``noodles``, ``peas``, ``beans``,
                    ``chilies``/``chiles`` (also spelling variants),
                    ``asparagus``, ``molasses`` are deliberately untouched.
5. ``synonym``    – a small whole-name dictionary :data:`SYNONYMS`, each entry with
                    a written rationale. Distinctions such as ground vs whole
                    spices, fresh vs dried, cheese types, oils and peppers are preserved.

Digits, quantities, units and brand words are *not* removed (only the ``®``/``™``
glyphs are), because removing them can change ingredient identity.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

# Final-token plural -> singular. Only plain count nouns whose singular denotes the
# same ingredient. Inspected against the actual vocabulary of the three cuisines.
PLURAL_WHITELIST: dict[str, str] = {
    "tomatoes": "tomato", "onions": "onion", "potatoes": "potato", "carrots": "carrot",
    "mushrooms": "mushroom", "eggs": "egg", "olives": "olive", "shallots": "shallot",
    "almonds": "almond", "tomatillos": "tomatillo", "raisins": "raisin", "scallions": "scallion",
    "peppercorns": "peppercorn", "lentils": "lentil", "chickpeas": "chickpea", "radishes": "radish",
    "leeks": "leek", "walnuts": "walnut", "hazelnuts": "hazelnut", "peanuts": "peanut",
    "pecans": "pecan", "cashews": "cashew", "apples": "apple", "pistachios": "pistachio",
    "artichokes": "artichoke", "pears": "pear", "bananas": "banana", "figs": "fig", "plums": "plum",
    "grapes": "grape", "jalapenos": "jalapeno", "plantains": "plantain", "beets": "beet",
    "prawns": "prawn", "clams": "clam", "mussels": "mussel", "scallops": "scallop",
    "anchovies": "anchovy", "eggplants": "eggplant", "chestnuts": "chestnut", "prunes": "prune",
    "yams": "yam", "parsnips": "parsnip", "oranges": "orange", "cranberries": "cranberry",
    "strawberries": "strawberry", "raspberries": "raspberry", "blueberries": "blueberry",
    "blackberries": "blackberry", "cherries": "cherry", "peaches": "peach", "currants": "currant",
    "capers": "caper", "tortillas": "tortilla", "seeds": "seed", "breasts": "breast",
    "fillets": "fillet", "thighs": "thigh", "drumsticks": "drumstick", "sausages": "sausage",
    "lemons": "lemon", "limes": "lime", "cucumbers": "cucumber", "zucchinis": "zucchini",
    "avocados": "avocado", "dates": "date", "apricots": "apricot", "mangoes": "mango",
    "tostadas": "tostada", "meatballs": "meatball", "cutlets": "cutlet",
}

# Whole-name synonyms (applied after singularisation). Key -> (target, rationale).
SYNONYMS: dict[str, tuple[str, str]] = {
    "scallion": ("green onion", "US/UK names for the same young Allium (A. fistulosum / immature A. cepa) sold as one product"),
    "garbanzo beans": ("chickpea", "garbanzo and chickpea are two names for the same legume, Cicer arietinum"),
    "asafetida": ("asafoetida", "alternative spelling of the same spice (Ferula resin)"),
    "cuminseed": ("cumin seed", "spacing variant of 'cumin seed'"),
    "chat masala": ("chaat masala", "transliteration variant of the same spice blend"),
}

_QUOTES = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00b4": "'", "\u0060": "'"}
_SYMBOLS = ("\u00ae", "\u2122", "\u00a9")  # ®, ™, ©
_WS = re.compile(r"\s+")


SINGULARIZE_EXCEPTIONS: frozenset[str] = frozenset({
    "veal scallops",  # culinary cut (scaloppine), not the bivalve mollusk 'scallop'
})


def _unicode_clean(s: str) -> str:
    for k, v in _QUOTES.items():
        s = s.replace(k, v)
    for sym in _SYMBOLS:
        s = s.replace(sym, " ")  # replaced before NFKC so '™' (\u2122) does not become 'TM'
    s = unicodedata.normalize("NFKC", s)
    for k, v in _QUOTES.items():
        s = s.replace(k, v)
    for sym in _SYMBOLS:
        s = s.replace(sym, " ")
    # accent folding: decompose and drop combining marks
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", s)


def _whitespace(s: str) -> str:
    return _WS.sub(" ", s.strip())


def _singularize(s: str) -> str:
    if not s or s in SINGULARIZE_EXCEPTIONS:
        return s
    head, sep, last = s.rpartition(" ")
    if last in PLURAL_WHITELIST:
        return f"{head}{sep}{PLURAL_WHITELIST[last]}"
    return s


def _synonym(s: str) -> str:
    return SYNONYMS[s][0] if s in SYNONYMS else s


STAGES = (
    ("unicode", _unicode_clean),
    ("lowercase", str.lower),
    ("whitespace", _whitespace),
    ("singularize", _singularize),
    ("synonym", _synonym),
)
FORMAT_STAGES = ("unicode", "lowercase", "whitespace")


def normalize_with_trace(name: str) -> tuple[str, list[str], dict[str, str]]:
    """Normalise one name and return ``(final, rules_that_changed_it, name_after_each_stage)``."""
    if not isinstance(name, str):
        raise TypeError(f"Ingredient must be a string, got {type(name).__name__}")
    rules: list[str] = []
    after: dict[str, str] = {}
    cur = name
    for rule, fn in STAGES:
        new = fn(cur)
        if new != cur:
            rules.append(rule)
        after[rule] = new
        cur = new
    return cur, rules, after


def normalize_ingredient(name: str) -> str:
    """Return the canonical form of one ingredient name."""
    return normalize_with_trace(name)[0]


def normalize_recipe(ingredients: list[str]) -> list[str]:
    """Normalise a recipe's ingredient list, drop empties, deduplicate, and sort.

    Sorting makes the output order-independent and deterministic.
    """
    out = {normalize_ingredient(x) for x in ingredients}
    out.discard("")
    return sorted(out)


def normalizer_fingerprint() -> str:
    """Hash of this module's source; used to invalidate caches when rules change."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
