from src.data.loader import load_documents, parse_markdown


def test_parse_markdown_strips_front_matter_and_shortcodes():
    raw = '---\ntitle: "Vacation Policy"\n---\n\nHello {{< note >}} world\n\n\n\nEnd'
    text, title = parse_markdown(raw, fallback_title="fallback")
    assert title == "Vacation Policy"
    assert "---" not in text
    assert "{{" not in text
    assert "\n\n\n" not in text


def test_parse_markdown_removes_links_and_invisible_chars():
    raw = "See [the policy](/handbook/x/) for\u200b details."
    text, _ = parse_markdown(raw, fallback_title="t")
    assert text == "See the policy for details."


def test_parse_markdown_removes_images():
    raw = "Before ![diagram](/img/a.png) after."
    text, _ = parse_markdown(raw, fallback_title="t")
    assert text == "Before  after."


def test_parse_markdown_uses_fallback_title_without_front_matter():
    text, title = parse_markdown("Just text.", fallback_title="my-file")
    assert title == "my-file"
    assert text == "Just text."


def test_load_documents_sets_department(tmp_path):
    dept = tmp_path / "legal"
    dept.mkdir()
    (dept / "a.md").write_text("---\ntitle: A\n---\nBody text", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].department == "legal"
    assert docs[0].title == "A"