import re

def remove_diacritic(input_text: str) -> str:
    """Removes accents from arabic text

    Calling this method on arabic text removes diacritics from the string.

    Args:
        input_text: The text to be cleaned.

    Returns:
        The text after removing all accents.
    """
    remove = re.compile(r"['ِ''ُ''ٓ''ٰ''ْ''ٌ''ٍ''ً''ّ''َ'`\"]")
    if input_text is None:
        return None
    return re.sub(remove, "", input_text.strip())
