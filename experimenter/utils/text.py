import re

def remove_diacritic(input_text: str) -> str:
    """
    Removes accents from arabic text

    Args:
        input_text (str): The text to be cleaned.

    Returns:
        str: the text after removing all accents.
    """
    remove = re.compile(r"['ِ''ُ''ٓ''ٰ''ْ''ٌ''ٍ''ً''ّ''َ'`\"]")
    if input_text is None:
        return None
    return re.sub(remove, "", input_text.strip())
