# Modified DataTrove snapshot supplied by the user; see VENDORING.md.
from .csv import CSVReader
from .huggingface import HuggingFaceDatasetReader
from .ipc import IpcReader
from .jsonl import JsonlReader
from .parquet import ParquetReader
from .warc import WarcReader
