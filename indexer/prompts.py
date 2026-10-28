"""Prompt catalogue for the nightly image indexer.

Each entry is one model call per image. Keys are referenced by
INDEXER_PROMPTS (comma-separated), default "scene,objects".

Keep prompts search-oriented: the outputs go verbatim into an SQLite
FTS5 index that the owner queries with plain Russian words.
"""

PROMPTS: dict[str, str] = {
    # "Суть изображения" — основной поисковый текст. The three branches matter:
    # a pure-text image (quote, sign, note) used to come back as "нет текста",
    # because a scene-first prompt made the model look for a scene and refuse.
    "scene": (
        "Что на изображении? Ответь по-русски, 3-5 предложений.\n"
        "Если изображение состоит в основном из текста (цитата, надпись, "
        "объявление, документ) — приведи этот текст дословно в кавычках и "
        "коротко скажи, что это.\n"
        "Если это скриншот приложения или сайта — скажи, что за экран и что "
        "на нём происходит; из надписей приводи только главное.\n"
        "Если это обычная фотография — опиши место, событие, что происходит, "
        "кто присутствует, время суток, обстановку и настроение.\n"
        "Не выдумывай того, чего не видно."
    ),
    # "Объекты" — плоский список для поиска по словам вроде «кошка», «велосипед».
    # People get their own line with a short identifying phrase: a plain
    # gender word alone made the model duplicate "мужчина"/"женщина" for the
    # same person, while a phrase pins it to one concrete person.
    "objects": (
        "Перечисли, что видно на изображении: людей, предметы, растения, "
        "здания.\n"
        "Правила: каждый объект — отдельная строка из 1-3 слов, без запятых, "
        "без нумерации. Не более 15 строк. Ничего не повторяй и не "
        "перечисляй одно и то же разными словами.\n"
        "Люди: каждого человека описывай одной строкой целиком, например "
        "«девушка в красном пальто» или «мужчина в костюме». Не разбивай "
        "одного человека на несколько строк (не пиши отдельно про волосы, "
        "усы, одежду, обувь, украшения) и не выдумывай людей там, где их "
        "нет. Если людей на изображении нет — не упоминай их вообще.\n"
        "Если на изображении товары или скриншот магазина, перечисляй сами "
        "товары; людей с картинок-примеров и превью не перечисляй.\n"
        "Пол указывай только если он различим, иначе «человек».\n"
        "Описывай только то, что реально видно. Не придумывай объекты.\n"
        "Если изображение состоит в основном из текста (цитата, надпись, "
        "документ), вместо списка объектов напиши одной строкой: "
        "«изображение с текстом».\n"
        "Отвечай строго по-русски."
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
