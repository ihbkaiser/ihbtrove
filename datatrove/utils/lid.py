# Modified DataTrove snapshot supplied by the user; see VENDORING.md.
from abc import abstractmethod
import os
from pathlib import Path

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.utils._import_utils import check_required_dependencies

from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 0


class LID:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages

    @abstractmethod
    def predict(self, doc: Document) -> tuple[tuple[str, int], dict[str, float]]:
        """
        Predicts the likelihood of the document being written in given languages, alongside with the most likely language
        Args:
            doc (Document): Document to predict languages for
        Returns:
            dict[str, float]: Languages and score
        """
        raise NotImplementedError


class FastTextLID(LID):
    MODEL_URL = None
    MODEL_SUBFOLDER = None

    def __init__(self, languages: list[str] | None = None, k: int = -1) -> None:
        """
        Args:
            languages (list[str]): Languages to predict
            k (int, optional): Number of top-k languages to consider, all languages outside of k will be considered as being predicted with 0.0
        """
        super().__init__(languages)
        self._model = None
        self.k = k

    @property
    def model(self):
        if self._model is None:
            check_required_dependencies("lid", [("fasttext", "fasttext-numpy2-wheel")])
            from fasttext.FastText import _FastText

            model_file = cached_asset_path_or_download(
                self.MODEL_URL,
                namespace="lid",
                subfolder=self.MODEL_SUBFOLDER,
                desc="fast-text language identifier model",
            )
            self._model = _FastText(model_file)
        return self._model

    def predict(self, doc: Document) -> tuple[tuple[str, int], dict[str, float]]:
        langs, scores = self.model.predict(doc.text.replace("\n", " "), k=self.k)
        lang_pairs = {lang.split("__")[2]: score.item() for lang, score in zip(langs, scores)}
        best_lang_pair = max(lang_pairs.items(), key=lambda x: x[1])
        return best_lang_pair, {
            lang: lang_pairs.get(lang, 0.0) for lang in self.languages
        } if self.languages else lang_pairs


class LangDetectLID(LID):
    def __init__(self, languages=None, k = -1):
        super().__init__(languages)
        self.k = k

    def predict(self, doc: Document) :
        text = (doc.text or "").strip()
        if not text:
            return ("unknown", 0), {}

        try:
            preds = detect_langs(text)
        except Exception:
            return ("unknown", 0), {}

        lang_pairs = {p.lang: float(p.prob) for p in preds}
        if not lang_pairs:
            return ("unknown", 0), {}

        best_lang_pair = max(lang_pairs.items(), key=lambda x: x[1])

        if self.languages:
            filtered = {
                lang: lang_pairs.get(lang, 0.0) for lang in self.languages
            }
        else:
            filtered = lang_pairs

        return best_lang_pair, filtered


class FT176LID(FastTextLID):
    # MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    MODEL_URL = "/workspace/storage-shared/nlp/maitn4/code/data-processing/utils/lid.176.bin"
    MODEL_SUBFOLDER = "ft176"

    def __init__(self, languages: list[str] | None = None, k: int = -1) -> None:
        super().__init__(languages, k)
        # IHB-Trove modification: retain the source archive's local model path.
        # An explicit local model takes precedence; the public URL is used only
        # when neither local path is available. The model itself is not bundled.
        configured = os.environ.get("IHB_TROVE_FT176_MODEL")
        if configured:
            if not Path(configured).is_file():
                raise FileNotFoundError(f"IHB_TROVE_FT176_MODEL does not exist: {configured}")
            self.MODEL_URL = configured
        elif not Path(self.MODEL_URL).is_file():
            self.MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


class GlotLID(FastTextLID):
    MODEL_SUBFOLDER = "glotlid"

    def __init__(self, languages: list[str] | None = None, k: int = -1, version: str = "v3") -> None:
        """
        Args:
            languages (list[str]): Languages to predict
            k (int, optional): Number of top-k languages to consider, all languages outside k will be considered as being predicted with 0.0
            version (str, optional): GlotLID version to use
        """
        super().__init__(languages, k)
        self.MODEL_URL = f"hf://cis-lmu/glotlid/model_{version}.bin"


# We don't support CLD3, not only it's worse than fasttext, but installation is really problematic, because of old version of protobuffers
