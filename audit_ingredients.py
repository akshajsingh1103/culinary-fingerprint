
import json
import re
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/raw/train.json")


def basic_normalize(name):
    """Only safe formatting changes; does not merge synonyms."""
    return re.sub(r"\s+", " ", name.strip().lower())


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH.resolve()}")
        print("Run this script from the culinary-fingerprint folder.")
        return

    with DATA_PATH.open("r", encoding="utf-8") as f:
        recipes = json.load(f)

    if not isinstance(recipes, list):
        raise ValueError("Expected train.json to contain a JSON list.")

    required = {"id", "cuisine", "ingredients"}
    if recipes and not required.issubset(recipes[0]):
        raise ValueError(
            f"Unexpected recipe fields: {recipes[0].keys()}"
        )

    print(f"Recipes: {len(recipes):,}")
    print("Cuisine counts:")
    for cuisine, count in Counter(
        r["cuisine"] for r in recipes
    ).most_common():
        print(f"  {cuisine}: {count:,}")

    raw_names = []
    for recipe in recipes:
        ingredients = recipe.get("ingredients", [])
        if not isinstance(ingredients, list):
            continue
        raw_names.extend(
            x for x in ingredients if isinstance(x, str)
        )

    unique_raw = set(raw_names)
    unique_basic = {basic_normalize(x) for x in raw_names}

    print("\nIngredient audit")
    print(f"Total ingredient occurrences: {len(raw_names):,}")
    print(f"Unique raw ingredient strings: {len(unique_raw):,}")
    print(f"Unique after lowercase/whitespace cleanup: "
          f"{len(unique_basic):,}")

    print("\nFormatting issues:")
    print("Leading/trailing spaces:",
          sum(x != x.strip() for x in unique_raw))
    print("Uppercase/mixed-case names:",
          sum(x != x.lower() for x in unique_raw))
    print("Repeated internal whitespace:",
          sum(bool(re.search(r'\s{2,}', x)) for x in unique_raw))

    # Find distinct raw strings that become identical after
    # safe lowercase/whitespace cleanup.
    groups = {}
    for name in unique_raw:
        groups.setdefault(basic_normalize(name), set()).add(name)

    collisions = {
        normalized: sorted(originals)
        for normalized, originals in groups.items()
        if len(originals) > 1
    }

    print(f"\nFormatting-equivalent groups: {len(collisions):,}")
    for normalized, originals in sorted(collisions.items())[:40]:
        print(f"  {originals} -> {normalized}")

    # Repeated ingredients within a single recipe can affect
    # co-occurrence counts, so report them separately.
    duplicate_recipe_count = 0
    for recipe in recipes:
        ingredients = recipe.get("ingredients", [])
        cleaned = [basic_normalize(x) for x in ingredients
                   if isinstance(x, str)]
        if len(cleaned) != len(set(cleaned)):
            duplicate_recipe_count += 1

    print("\nRecipe-level duplicates after basic cleanup:",
          duplicate_recipe_count)
    print("\nNote: this does NOT check plural forms, synonyms, "
          "or ingredient meaning.")


if __name__ == "__main__":
    main()
