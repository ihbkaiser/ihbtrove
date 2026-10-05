# Modified DataTrove snapshot supplied by the user; see VENDORING.md.
from .local import LocalPipelineExecutor
from .ray import RayPipelineExecutor
from .slurm import SlurmPipelineExecutor
