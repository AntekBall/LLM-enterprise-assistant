from src.data.loader import load_documents, parse_markdown


def test_parse_markdown_strips_front_matter_and_shortcodes():
    raw = '---\ntitle: "Vacation Policy"\n---\n\nHello {{< note >}} world\n\n\n\nEnd'
    text, title = parse_markdown(raw, fallback_title="fallback")
    assert title == "Vacation Policy"
    assert "---" not in text
    assert "{{" not in text
    assert "\n\n\n" not in text


def test_load_documents_sets_department(tmp_path):
    dept = tmp_path / "legal"
    dept.mkdir()
    (dept / "a.md").write_text("---\ntitle: A\n---\nBody text", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].department == "legal"
    assert docs[0].title == "A"