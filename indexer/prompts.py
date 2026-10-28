"""Prompt catalogue for the nightly image indexer.

Each entry is one model call per image. Keys are referenced by
INDEXER_PROMPTS (comma-separated), default "scene,objects".

Keep prompts search-oriented: the outputs go verbatim into an SQLite
FTS5 index that the owner queries with plain Russian words.
"""

PROMPTS: dict[str, str] = {
    # "Суть изображения" — основной поисковый текст.
    # Deliberately short: reasoning models (qwen3-vl) sink into long
    # deliberation when given many rules, burn the token budget and return
    # nothing. A short prompt with the three cases inline works reliably.
    "scene": (
        "Что на изображении? Ответь по-русски, 3-5 предложений. Если это "
        "текст — приведи его дословно. Если скриншот — скажи, что за экран, "
        "и что на нём. Если это товары — назови их и цены, если видны."
    ),
    # "Объекты" — плоский список для поиска по словам вроде «кошка», «велосипед».
    # People get a short identifying phrase ("девушка в красном свитере")
    # rather than a bare gender word: one phrase per person stops the model
    # from inventing a man and a woman for a single person.
    "objects": (
        "Перечисли, что видно на изображении. Каждый объект — на новой "
        "строке, 1-3 слова, до 15 строк. Человека описывай одной фразой "
        "с полом, если различается. Отвечай по-русски."
    ),
    # Дополнительный промпт, выключен по умолчанию (INDEXER_PROMPTS=scene,objects,text).
    "text": (
        "Извлеки весь видимый текст построчно, как написано. Язык любой. "
        "Если текста нет, выведи ровно: NO_TEXT"
    ),
}


def get_prompts(keys: str) -> dict[str, str]:
    """Return {key: prompt} for a comma-separated key list; raises on unknown."""
    chosen = [k.strip() for k in keys.split(",") if k.strip()]
    unknown = [k for k in chosen if k not in PROMPTS]
    if unknown:
        raise SystemExit(
            f"unknown prompt(s): {', '.join(unknown)}; available: {', '.join(PROMPTS)}"
        )
    if not chosen:
        raise SystemExit("INDEXER_PROMPTS is empty")
    return {k: PROMPTS[k] for k in chosen}
