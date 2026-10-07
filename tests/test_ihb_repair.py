"""Small adversarial fixtures for book/OCR repair and its quality gates."""

from datatrove.data import Document
from datatrove.pipeline.filters.gopher_repetition_filter import GopherRepetitionFilter

from ihb_trove.filters import BookQualityFilter
from ihb_trove.repair import (
    ImagePlaceholderRepair,
    LocalRepeatedSpanRepair,
    LongLineLoopRepair,
    PageStructureRepair,
    RepairMetrics,
    UnicodeNormalizer,
)
from ihb_trove.tokenization import UnicodeTokenizer


def process(text: str, *steps: object) -> Document:
    doc = Document(text=text, id="book-1", metadata={"source": "fixture"})
    stream = iter([doc])
    for step in steps:
        stream = step.run(stream)
    return next(stream)


def test_recurring_header_footer_and_ocr_variants() -> None:
    pages = []
    for index in range(6):
        header = "SÁCH NẤU ĂN GIA ĐÌNH" if index != 2 else "SÁCH NẤU AN GIA ĐINH"
        footer = "Các món bánh truyền thống" if index != 3 else "Các món bánh truyển thống"
        pages.append(f"{header}\nMón {index}: Trộn bột và hấp chín.\n{footer}")
    doc = process("\n---\n".join(pages), UnicodeNormalizer(), PageStructureRepair(), RepairMetrics())
    assert doc.text.count("SÁCH NẤU") == 0
    assert doc.text.count("Các món bánh") == 0
    assert doc.text.count("Trộn bột") == 6
    assert doc.metadata["repair"]["headers_removed"] == 6
    assert doc.metadata["repair"]["footers_removed"] == 6
    assert doc.metadata["source"] == "fixture"


def test_constant_numeric_book_title_is_header_but_numbered_recipes_are_not() -> None:
    pages = [
        f"100 cách chăm sóc tóc và da\nMón {index}: Trộn bột và hấp chín.\nGhi chú riêng của trang {index}."
        for index in range(8)
    ]
    doc = process("\n---\n".join(pages), PageStructureRepair(), RepairMetrics())
    assert doc.metadata["repair"]["headers_removed"] == 8
    assert doc.text.count("100 cách chăm sóc") == 0
    assert doc.text.count("Trộn bột") == 8


def test_page_number_with_consistent_offset_and_boundary_join() -> None:
    pages = [
        f"{index + 21}\nĐây là phần hướng dẫn nấu ăn rất chi tiết cho gia đình và"
        if index == 0
        else f"{index + 21}\ntrộn cùng một ít nước rồi hấp chín món thứ {index}."
        for index in range(4)
    ]
    doc = process("\f".join(pages), PageStructureRepair(), RepairMetrics())
    assert doc.metadata["repair"]["pages_detected"] == 4
    assert doc.metadata["repair"]["page_numbers_removed"] == 4
    assert doc.metadata["repair"]["boundary_joins"] == 1
    assert "gia đình và trộn cùng" in doc.text
    assert "\n21\n" not in doc.text


def test_local_page_number_offset_shift_and_inline_footer() -> None:
    pages = []
    for index in range(9):
        number = index + (21 if index < 5 else 22)
        footer = "Các món bánh truyền thống"
        last = f"Cần khuấy thật đều cho tới khi bột trở nên mềm mịn...{footer}" if index == 2 else footer
        pages.append(f"{number}\nCông thức {index}: làm bánh cùng gia đình.\n{last}")
    doc = process("\n---\n".join(pages), PageStructureRepair(), RepairMetrics())
    assert doc.metadata["repair"]["page_numbers_removed"] == 9
    assert doc.metadata["repair"]["footers_removed"] == 9
    assert "bột trở nên mềm mịn" in doc.text
    assert "mịn...Các món" not in doc.text


def test_configurable_marker_and_non_boundary_number() -> None:
    text = "7\nĐịnh lượng 42 g bột ở giữa phần nội dung.\n[PAGE]\n8\nNấu trên lửa nhỏ.\n[PAGE]\n9\nĐể nguội."
    doc = process(text, PageStructureRepair(page_markers=("[PAGE]",)), RepairMetrics())
    assert doc.metadata["repair"]["pages_detected"] == 3
    assert doc.metadata["repair"]["page_numbers_removed"] == 3
    assert "42 g bột" in doc.text


def test_nearby_paragraph_and_multiline_span_dedup_but_preserve_headings() -> None:
    paragraph = "Đun đường với nước ấm, thêm bột vào và khuấy đều tới khi hỗn hợp sánh mịn và có mùi thơm tự nhiên trong căn bếp."
    lines = "Cho phần bột này vào nồi nhỏ để đun thêm mười phút nữa.\nKhuấy đều tay rồi tắt bếp và để nguội trước khi ăn."
    text = f"Nguyên liệu\n\n{paragraph}\n\nNguyên liệu\n\n{paragraph}\n\n{lines}\nDòng xen giữa.\n{lines}\n\nQuy trình chế biến\n\nQuy trình chế biến"
    doc = process(text, LocalRepeatedSpanRepair(), RepairMetrics())
    assert doc.text.count(paragraph) == 1
    assert doc.text.count("Cho phần bột này") == 1
    assert doc.text.count("Nguyên liệu") == 2
    assert doc.text.count("Quy trình chế biến") == 2
    assert doc.metadata["repair"]["duplicate_blocks_removed"] == 2


def test_repeated_short_book_headings_across_pages_are_preserved() -> None:
    pages = [f"Nguyên liệu\n{index} chén đậu và một ít nước.\nQuy trình chế biến\nĐun mềm rồi thưởng thức." for index in range(7)]
    doc = process("\n---\n".join(pages), PageStructureRepair(), LocalRepeatedSpanRepair(), RepairMetrics())
    assert doc.text.count("Nguyên liệu") == 7
    assert doc.text.count("Quy trình chế biến") == 7
    assert doc.metadata["repair"]["headers_removed"] == 0
    assert doc.metadata["repair"]["duplicate_blocks_removed"] == 0


def test_gopher_does_not_reject_books_for_repeated_short_headings() -> None:
    sections = []
    for index in range(100):
        body = " ".join(f"thanhphan{index}nguyenlieu{word}" for word in range(24))
        sections.append(f"Nguyên liệu\n\n{body}")
    doc = Document(text="\n\n".join(sections), id="recipe-book")
    language_neutral_gopher = GopherRepetitionFilter(
        language=None,
        dup_line_frac=None,
        dup_para_frac=None,
        dup_line_char_frac=0.50,
        dup_para_char_frac=0.50,
        top_n_grams=(),
        dup_n_grams=(),
    )
    assert language_neutral_gopher.filter(doc) is True
    assert language_neutral_gopher.filter(Document(text="", id="repaired-empty")) is True


def test_metrics_and_permissive_quality_gate() -> None:
    doc = process("Bánh ngọt và chè đậu.\n\n" * 8, UnicodeNormalizer(), LocalRepeatedSpanRepair(), RepairMetrics())
    repair = doc.metadata["repair"]
    for field in ("pages_detected", "headers_removed", "footers_removed", "page_numbers_removed", "boundary_joins", "duplicate_blocks_removed", "chars_before", "chars_after", "removed_fraction", "repair_confidence", "repair_version"):
        assert field in repair
    assert repair["chars_before"] >= repair["chars_after"]
    assert BookQualityFilter(min_words=8).filter(doc) is True
    doc.metadata["repair"]["removed_fraction"] = 0.8
    doc.metadata["repair"]["repair_confidence"] = 0.1
    assert BookQualityFilter().filter(doc) is True


def test_quality_gate_keeps_short_and_mixed_script_documents() -> None:
    quality = BookQualityFilter()
    for text in ("短い文。", "مرحبا بالعالم", "Привет мир", "Hello", "नमस्ते दुनिया"):
        assert quality.filter(Document(text=text, id="multilingual")) is True
    assert quality.filter(Document(text="  \n", id="empty")) == (False, "empty_text")


def test_quality_gate_only_rejects_extreme_repeated_long_lines() -> None:
    repeated_line = "This long line is repeated by a damaged OCR pass and should be caught. " * 3
    text = "\n".join([repeated_line] * 8)
    assert BookQualityFilter().filter(Document(text=text, id="repeated")) == (False, "extreme_repeated_lines")


def test_unicode_tokenizer_handles_multiple_scripts_without_locale_setting() -> None:
    tokenizer = UnicodeTokenizer()
    assert tokenizer.word_tokenize("Hello, привет 世界こんにちは สวัสดี") == [
        "Hello", "привет", "世", "界", "こ", "ん", "に", "ち", "は", "ส", "วั", "ส", "ดี"
    ]
    assert tokenizer.sent_tokenize("Hello! 你好。Tail without punctuation") == [
        "Hello!", "你好。", "Tail without punctuation"
    ]


def test_unicode_normalizer_ligatures_soft_hyphens_and_whitespace() -> None:
    text = "ﬁne ﬂour\ninter\u00ad\nnational\nsoft\u00adhyphen   \n\n\n\nNext line.   "
    doc = process(text, UnicodeNormalizer(), RepairMetrics())
    repair = doc.metadata["repair"]
    assert doc.text == "fine flour\ninternational\nsofthyphen\n\n\nNext line."
    assert repair["ligatures_normalized"] == 2
    assert repair["soft_hyphens_removed"] == 2
    assert repair["soft_hyphen_line_joins"] == 1
    assert repair["trailing_whitespace_removed"] == 6
    assert repair["blank_lines_collapsed"] == 1


def test_visible_hyphen_line_join_is_opt_in() -> None:
    text = "inter-\nnational\nBắc-\nNam"
    unchanged = process(text, UnicodeNormalizer())
    assert unchanged.text == text
    repaired = process(text, UnicodeNormalizer(dehyphenate_line_breaks=True))
    assert repaired.text == "international\nBắcNam"
    assert repaired.metadata["repair"]["line_end_hyphens_joined"] == 2


def test_unicode_normalizer_can_preserve_whitespace_and_ligatures() -> None:
    text = "ﬁle   \n\n\nparagraph"
    doc = process(
        text,
        UnicodeNormalizer(
            normalize_ligatures=False,
            trim_trailing_whitespace=False,
            max_consecutive_blank_lines=None,
        ),
    )
    assert doc.text == text


def test_text_only_image_references_and_generated_note_are_removed() -> None:
    text = (
        "Mô tả hình ảnh là một phần nội dung thật.\n"
        "![Ảnh](https://example.com/image.jpg)\n"
        "Nội dung trước ![hình](image_url) nội dung sau.\n"
        "<img alt='placeholder' src='image.png'>\n"
        "Note: Since the image is not available in this text-based environment, an image placeholder is provided.\n"
        "Kết luận vẫn được giữ."
    )
    doc = process(text, ImagePlaceholderRepair(), RepairMetrics())
    assert "Mô tả hình ảnh" in doc.text
    assert "Nội dung trước  nội dung sau." in doc.text
    assert "Kết luận vẫn được giữ." in doc.text
    assert "![" not in doc.text and "<img" not in doc.text and "Note:" not in doc.text
    assert doc.metadata["repair"]["image_placeholders_removed"] == 3
    assert doc.metadata["repair"]["image_notes_removed"] == 1


def test_exact_long_line_ocr_loop_is_collapsed_without_erasing_headings() -> None:
    phrase = "Ta sẽ trở thành người bạn thân thiết và tin tưởng họ hơn."
    text = "Nguyên liệu\n" + "Mở đầu. " + (phrase + " ") * 25 + "Kết luận.\nNguyên liệu"
    doc = process(text, LongLineLoopRepair(), RepairMetrics())
    assert doc.text.count(phrase) == 1
    assert doc.text.count("Nguyên liệu") == 2
    assert "Kết luận." in doc.text
    assert doc.metadata["repair"]["long_line_loops_removed"] == 1
    assert doc.metadata["repair"]["loop_chars_removed"] > 1000
