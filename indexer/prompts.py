"""Prompt catalogue for the nightly image indexer.

Each entry is one model call per image. Keys are referenced by
INDEXER_PROMPTS (comma-separated), default "scene,objects".

Keep prompts search-oriented: the outputs go verbatim into an SQLite
FTS5 index that the owner queries with plain Russian words.
"""

PROMPTS: dict[str, str] = {
    # "Суть изображения" — основной поисковый текст.
    "scene": (
        "Ты помогаешь вести поисковый индекс домашнего фотоархива. Опиши "
        "изображение максимально информативно: что за место и что за событие, "
        "что происходит, кто присутствует, время суток, погода, улица или "
        "интерьер, настроение кадра. 3-5 полных предложений по-русски, "
        "конкретно и по существу. Не выдумывай деталей, которых не видно. "
        "Если на изображении есть крупный текст (вывеска, документ, экран) — "
        "кратко перескажи его суть."
    ),
    # "Объекты" — плоский список для поиска по словам вроде «кошка», «велосипед».
    "objects": (
        "Перечисли по-русски все заметные объекты, людей, животных, растения "
        "и предметы, которые видны на изображении. По одному объекту на "
        "строку, короткой фразой из 1-3 слов, без нумерации, без точек, без "
        "пояснений. Не более 25 строк. Людей указывай обобщённо (мужчина, "
        "женщина, ребёнок, группа людей), без предположений о личности."
    ),
    # Дополнительный промпт, выключен по умолчанию (INDEXER_PROMPTS=scene,objects,text).
    "text": (
        "Извлеки весь видимый на изображении текст: надписи, вывески, "
        "таблички, экраны, документы. Пиши так, как написано, построчно: "
        "одна строка — один фрагмент. Язык любой. Если текста нет, выведи "
        "ровно: NO_TEXT"
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
