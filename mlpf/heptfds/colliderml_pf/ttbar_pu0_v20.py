from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from mlpf.heptfds.colliderml_pf.ttbar import _CITATION, _X_FEATURES
from mlpf.heptfds.colliderml_utils.utils import V20_PU0_LAYOUT, generate_examples_v20, split_sample_v20, v20_config_names

_DESCRIPTION = """
ColliderML v20 ttbar without pileup (the production with Pandora particle flow), converted for MLPF training.
Splits are assigned by event id: configs 1-10 train = 9 of the 90 adapter train shards each (ids 0-7679); config 1
test = ids 7680-8319; config 2 val_read = ids 8320-8959; r1-r10 test = one 128-event held-out report shard each
(ids 8960-10239).
"""


class CollidermlTtbarPu0V20Pf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release: event-id splits, plus event_id and gen_target per event.",
    }
    DESCRIPTION = _DESCRIPTION
    LAYOUT = V20_PU0_LAYOUT
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    The converter output (`python3 -m mlpf.data.colliderml.postprocessing ...`) of the v20 adapter shards
    (`python3 -m mlpf.data.colliderml.v20_adapter ...`), in `dl_manager.manual_dir`. Files are classified by the
    event ids they hold, so their names do not matter.
    """

    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=n) for n in v20_config_names(LAYOUT)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, 17), dtype=np.float32),
                    "ytarget": tfds.features.Tensor(shape=(None, 14), dtype=np.float32),
                    "ycand": tfds.features.Tensor(shape=(None, 14), dtype=np.float32),
                    "genmet": tfds.features.Scalar(dtype=np.float32),
                    "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                    "event_id": tfds.features.Scalar(dtype=np.int64),
                    "gen_target": tfds.features.Tensor(shape=(None, 10), dtype=np.float32),
                }
            ),
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=_X_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return split_sample_v20(Path(dl_manager.manual_dir), self.builder_config.name, self.LAYOUT)

    def _generate_examples(self, files):
        return generate_examples_v20(files)
