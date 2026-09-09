import re


def segment_document(text: str) -> list[dict]:
    """
    Segment the complete OCR text into question-wise sections.

    The input should be the complete document text, not individual pages.

    Example:
        Q1 What is inheritance?
        ...
        A2 Define OOP.
        ...
        A3 What is Polymorphism?
        ...
        Q4)
        ...

    Returns:
        [
            {
                "question_id": "Q1",
                "text": "..."
            },
            ...
        ]
    """

    if not text or not text.strip():
        return []

    # ---------------------------------------------------------------
    # Normalize text
    # ---------------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # ---------------------------------------------------------------
    # Find question headers
    # ---------------------------------------------------------------
    #
    # Matches:
    #
    # Q1 What is inheritance?
    # Q2 Define OOP
    # A2 Define OOP       <- OCR may read Q as A
    # Q3 What is...
    # A3 What is...       <- OCR may read Q as A
    # Q4)
    #
    # Does NOT match:
    #
    # 1) Single inheritance
    # 2) Multiple inheritance
    # 3) Compile-time polymorphism
    #
    # because those don't begin with Q/A.
    #

    pattern = re.compile(
        r"(?m)"
        r"^[ \t]*"
        r"(?:Q|A)"
        r"[ \t]*"
        r"(\d+)"
        r"[ \t]*"
        r"(?:[.):\-])?"
        r"(?:[ \t]+|$)"
    )

    matches = list(pattern.finditer(text))

    # ---------------------------------------------------------------
    # No question headers found
    # ---------------------------------------------------------------

    if not matches:
        return [
            {
                "question_id": "Q1",
                "text": text.strip(),
            }
        ]

    segments = []

    # ---------------------------------------------------------------
    # Split between question headers
    # ---------------------------------------------------------------

    for index, match in enumerate(matches):

        question_number = match.group(1)

        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        segment_text = text[start:end].strip()

        if not segment_text:
            continue

        segments.append(
            {
                "question_id": f"Q{question_number}",
                "text": segment_text,
            }
        )

    return segments