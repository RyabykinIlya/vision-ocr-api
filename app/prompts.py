"""Named prompt presets for common recognition tasks."""

PRESETS: dict[str, str] = {
    "ocr": (
        "You are an OCR engine. Read every piece of text visible in the "
        "image(s) and transcribe it verbatim. Preserve line breaks, reading "
        "order, and table structure (use markdown tables when a table is "
        "present). Do not translate, summarize, or add commentary. If no "
        "text is visible, reply with exactly: NO_TEXT_FOUND."
    ),
    "ocr_translate": (
        "Read every piece of text visible in the image(s) and transcribe it, "
        "then translate the transcription into Russian. Preserve line breaks "
        "and table structure (use markdown tables when a table is present). "
        "Output only the translated text, no commentary."
    ),
    "describe": (
        "Describe what is shown in the image(s) in clear, concise prose. "
        "Mention the key objects, text, and any notable details. Reply in "
        "Russian."
    ),
    "table": (
        "Extract every table visible in the image(s) and output it as "
        "markdown tables, preserving column headers and row order exactly. "
        "If there is no table, reply with exactly: NO_TABLE_FOUND."
    ),
    "diagram": (
        "Explain the diagram or chart shown in the image(s): what kind of "
        "diagram it is, its components, and how they relate. Reply in "
        "Russian."
    ),
}

DEFAULT_PRESET = "ocr"
