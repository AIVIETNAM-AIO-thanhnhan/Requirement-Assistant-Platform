"""cleaner.py — Làm sạch text (nhẹ tay, giữ nguyên nội dung)."""
import re
import unicodedata


def clean_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC + gọn khoảng trắng / dòng trống thừa."""
    text = unicodedata.normalize("NFC", text)  # gộp dấu tiếng Việt về 1 dạng
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
