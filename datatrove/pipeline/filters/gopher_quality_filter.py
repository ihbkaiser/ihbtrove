# Modified DataTrove snapshot supplied by the user; see VENDORING.md.
import numpy as np

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.utils.typeshelper import Languages


STOP_WORDS = [
    "a lô","a ha","ai","ai ai","ai nấy","ai đó","alô","amen","anh","anh ấy",
    "ba","ba ba","ba bản","ba cùng","ba họ","ba ngày","ba ngôi","ba tăng",
    "bao giờ","bao lâu","bao nhiêu","bao nả","bay biến",
    "biết","biết bao","biết bao nhiêu","biết chắc","biết chừng nào","biết mình",
    "biết mấy","biết thế","biết trước","biết việc","biết đâu","biết đâu chừng",
    "biết đâu đấy","biết được",
    "buổi","buổi làm","buổi mới","buổi ngày","buổi sớm",
    "bà","bà ấy","bài","bài bác","bài bỏ","bài cái","bác",
    "bán","bán cấp","bán dạ","bán thế",
    "bây bẩy","bây chừ","bây giờ","bây nhiêu",
    "bèn","béng","bên","bên bị","bên có","bên cạnh",
    "bông","bước","bước khỏi","bước tới","bước đi",
    "bạn","bản","bản bộ","bản riêng","bản thân","bản ý",
    "bất chợt","bất cứ","bất giác","bất kì","bất kể","bất kỳ","bất luận",
    "bất ngờ","bất nhược","bất quá","bất quá chỉ","bất thình lình",
    "bất tử","bất đồ",
    "bấy","bấy chầy","bấy chừ","bấy giờ","bấy lâu","bấy lâu nay",
    "bấy nay","bấy nhiêu",
    "bập bà bập bõm","bập bõm",
    "bắt đầu","bắt đầu từ",
    "bằng","bằng cứ","bằng không","bằng người","bằng nhau","bằng như",
    "bằng nào","bằng nấy","bằng vào","bằng được","bằng ấy",
    "bển","bệt","bị","bị chú","bị vì",
    "bỏ","bỏ bà","bỏ cha","bỏ cuộc","bỏ không","bỏ lại","bỏ mình","bỏ mất",
    "bỏ mẹ","bỏ nhỏ","bỏ quá","bỏ ra","bỏ riêng","bỏ việc","bỏ xa",
    "bỗng","bỗng chốc","bỗng dưng","bỗng không","bỗng nhiên","bỗng nhưng",
    "bỗng thấy","bỗng đâu",
    "bộ","bộ thuộc","bộ điều","bội phần",
    "bớ","bởi","bởi ai","bởi chưng","bởi nhưng","bởi sao","bởi thế",
    "bởi thế cho nên","bởi tại","bởi vì","bởi vậy","bởi đâu",
    "cao","cao lâu","cao ráo","cao răng","cao sang","cao số","cao thấp",
    "cao thế","cao xa",
    "cha","cha chả","chao ôi",
    "chia sẻ","chiếc",
    "cho","cho biết","cho chắc","cho hay","cho nhau","cho nên","cho rằng",
    "cho rồi","cho thấy","cho tin","cho tới","cho tới khi","cho về",
    "cho ăn","cho đang","cho được","cho đến","cho đến khi","cho đến nỗi",
    "chung","chung cho","chung chung","chung cuộc","chung cục","chung nhau",
    "chung qui","chung quy","chung quy lại","chung ái",
    "chuyển","chuyển tự","chuyển đạt","chuyện","chuẩn bị",
    "chính","chính bản","chính giữa","chính là","chính thị","chính điểm",
    "chú","chú dẫn","chú khách","chú mày","chú mình",
    "chúng","chúng mình","chúng ta","chúng tôi","chúng ông",
    "chưa","chưa bao giờ","chưa chắc","chưa có","chưa cần","chưa dùng",
    "chưa dễ","chưa kể","chưa tính","chưa từng",
    "chắc","chắc chắn","chắc dạ","chắc hẳn","chắc lòng","chắc người",
    "chắc vào","chắc ăn",
    "chẳng lẽ","chẳng những","chẳng nữa","chẳng phải",
    "chỉ","chỉ chính","chỉ có","chỉ là","chỉ tên",
    "chị","chị bộ","chị ấy",
    "chịu","chịu chưa","chịu lời","chịu tốt","chịu ăn",
    "chứ","chứ ai","chứ còn","chứ gì","chứ không","chứ không phải",
    "chứ lại","chứ như","chứ sao",
    "có","có ai","có chuyện","có chăng","có chăng là","có chứ","có cơ",
    "có dễ","có họ","có khi","có ngày","có người","có nhiều","có nhà",
    "có phải","có số","có tháng","có thế","có thể","có vẻ","có ý",
    "có ăn","có điều","có điều kiện","có đáng","có đâu","có được",
    "cũng","cũng như","cũng nên","cũng thế","cũng vậy","cũng vậy thôi",
    "cũng được",
    "của","của ngọt","của tin",
    "cứ","cứ như","cứ việc","cứ điểm",
    "do","do vì","do vậy","do đó",
    "dù","dù cho","dù gì","dù rằng","dù sao",
    "dưới","dưới nước",
    "dần dần","dẫu","dẫu rằng","dẫu sao",
    "gần","gần bên","gần hết","gần ngày","gần như","gần đây","gần đến",
    "giờ","giờ này","giờ đây","giờ đến",
    "giữa","giữa lúc",
    "hay","hay không","hay là","hay sao",
    "hiện nay","hiện tại",
    "hoàn toàn","hoặc","hoặc là",
    "hãy","hơn","hơn nữa","hầu hết",
    "hết","hết rồi",
    "khi","khi nào","khi trước",
    "không","không ai","không bao giờ","không còn","không có",
    "không phải","không thể","không được",
    "là","là vì","là phải",
    "làm","làm cho","làm gì","làm sao","làm thế nào","làm được",
    "lại","lại còn","lại nữa",
    "lúc","lúc nào","lúc này","lúc đó",
    "mà","mà không","mà lại","mà thôi",
    "mọi","mọi người","mọi việc",
    "một","một cách","một khi","một số","một vài",
    "này","nào","nào cũng","nào là",
    "nên","nếu","nếu không","nếu như",
    "như","như thế","như vậy","nhưng","nhưng mà",
    "ra","ra sao","ra sao",
    "rằng","rồi","rồi thì",
    "sau","sau đó","sau này",
    "thì","thì ra","thì thôi",
    "thế","thế là","thế nên","thế thì",
    "trong","trong khi","trong lúc","trong đó",
    "trước","trước hết","trước khi","trước đó",
    "tuy","tuy nhiên",
    "và","vì","vì sao","vì vậy",
    "vẫn","vậy","vậy là","vậy nên",
    "về","về sau","về phần",
    "với","với nhau",
    "đang","đã","đã là","đã lâu",
    "đâu","đâu có","đây","đó",
    "được","đến","đến khi","đến nay",
    "để","để cho","đối với",
]


class GopherQualityFilter(BaseFilter):
    name = "🥇 Gopher Quality"

    def __init__(
        self,
        min_doc_words: int | None = 50,
        max_doc_words: int | None = 100000,
        min_avg_word_length: int | None = 3,
        max_avg_word_length: int | None = 10,
        max_symbol_word_ratio: float | None = 0.1,
        max_bullet_lines_ratio: float | None = 0.95,
        max_ellipsis_lines_ratio: float | None = 0.75,
        max_non_alpha_words_ratio: float | None = 0.5,
        min_stop_words: int | None = 2,
        stop_words: list[str] | None = None,
        exclusion_writer: DiskWriter = None,
        language: str = Languages.vietnamese,
    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        Args:
            min_doc_words:
            max_doc_words:
            min_avg_word_length:
            max_avg_word_length:
            max_symbol_word_ratio:
            max_bullet_lines_ratio:
            max_ellipsis_lines_ratio:
            max_non_alpha_words_ratio:
            min_stop_words:
            stop_words:
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_word_ratio = max_symbol_word_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.max_non_alpha_words_ratio = max_non_alpha_words_ratio  # TODO rename to min_alpha_words_ratio
        self.min_stop_words = min_stop_words
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED


        Returns: False if sample.text does not pass any of the heuristic tests

        """
        text = doc.text
        words = split_into_words(text, self.language)
        n_words = len(words)

        non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
        n_non_symbol_words_words = len(non_symbol_words)

        # words < min_doc_words or words > max_doc_words
        if self.min_doc_words and n_non_symbol_words_words < self.min_doc_words:
            return False, "gopher_short_doc"
        if self.max_doc_words and n_non_symbol_words_words > self.max_doc_words:
            return False, "gopher_long_doc"

        # mean word length is outside the range of 3 to 10 characters
        avg_n_words = np.mean([len(w) for w in non_symbol_words])
        if self.min_avg_word_length and avg_n_words < self.min_avg_word_length:
            return False, "gopher_below_avg_threshold"
        if self.max_avg_word_length and avg_n_words > self.max_avg_word_length:
            return False, "gopher_above_avg_threshold"

        # symbol-to-word ratio greater than 0.1 for either the hash symbol or the ellipsis
        if self.max_symbol_word_ratio and text.count("#") / n_words > self.max_symbol_word_ratio:
            return False, "gopher_too_many_hashes"
        if self.max_symbol_word_ratio and (text.count("...") + text.count("…")) / n_words > self.max_symbol_word_ratio:
            return False, "gopher_too_many_ellipsis"

        # any document with more than 90 % of lines starting with a bullet point,
        # or more than 30 % ending with an ellipsis.
        lines = text.splitlines()
        if (
            self.max_bullet_lines_ratio
            and sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / len(lines)
            > self.max_bullet_lines_ratio
        ):
            return False, "gopher_too_many_bullets"
        if (
            self.max_ellipsis_lines_ratio
            and sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / len(lines)
            > self.max_ellipsis_lines_ratio
        ):
            return False, "gopher_too_many_end_ellipsis"

        # that 80 % of words in a document contain at least one alphabetic character
        if (
            self.max_non_alpha_words_ratio
            # nb of words with at least 1 alpha char < 0.8
            and sum([any((c.isalpha() for c in w)) for w in words]) / n_words < self.max_non_alpha_words_ratio
        ):
            return False, "gopher_below_alpha_threshold"

        # stop word filter
        if self.min_stop_words and len(self.stop_words.intersection(set(words))) < self.min_stop_words:
            return False, "gopher_enough_stop_words"

        return True
