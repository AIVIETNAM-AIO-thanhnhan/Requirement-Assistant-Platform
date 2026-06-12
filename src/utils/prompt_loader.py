from pathlib import Path

TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent
    / "templates"
    / "prompts"
)


def load_prompt(relative_path: str) -> str:
    """
    Example:

    load_prompt(
        "production/answer_with_sources.md"
    )

    load_prompt(
        "experiments/qa_prompt_v1.md"
    )
    """

    prompt_path = TEMPLATE_DIR / relative_path

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_path}"
        )

    return prompt_path.read_text(
        encoding="utf-8"
    )