# ColliderML hits view: reuse the colliderml manual-dir split logic with a hits example
# generator (tracker + calo hits become the X elements; targets come from the hit-level
# columns written by mlpf/data/colliderml/postprocessing.py).
from mlpf.conf import Dataset
from mlpf.heptfds.colliderml_utils.utils import NUM_SPLITS  # noqa: F401  (re-export)
from mlpf.heptfds.colliderml_utils.utils import split_sample as _split_sample
from mlpf.heptfds.edm4hep_utils.utils_hits import X_FEATURES, Y_FEATURES  # noqa: F401  (re-exports)
from mlpf.heptfds.edm4hep_utils.utils_hits import generate_examples as _generate_examples


def generate_examples(files):
    # the hits view keeps calorimeter hit time (col 9) and layer (col 14) at zero
    for key, rec in _generate_examples(files, Dataset.COLLIDERML_HITS):
        calo = rec["X"][:, 0] == 2
        rec["X"][calo, 9] = 0.0
        rec["X"][calo, 14] = 0.0
        yield key, rec


def split_sample(path, builder_config, num_splits=NUM_SPLITS):
    return _split_sample(path, builder_config, num_splits=num_splits, gen_fn=generate_examples)
