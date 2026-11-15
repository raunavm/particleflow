from pathlib import Path

import tensorflow_datasets as tfds

from mlpf.heptfds.colliderml_pf.ttbar_pu0_v20 import CollidermlTtbarPu0V20Pf
from mlpf.heptfds.colliderml_utils.utils import V22_PU200_LAYOUT, split_sample_v22_block, v22_block_config_names

_DESCRIPTION = """
ColliderML v22 ttbar with ~200 pileup interactions, converted for MLPF training. Training events beyond the first 1,024
(ids 1664-10047), one config per 100-event source file (e01600 ... e10000), each with a single train split.
Same record schema as colliderml_ttbar_pu0_v20_pf.
"""


class CollidermlTtbarPu200V22PfBlocks(CollidermlTtbarPu0V20Pf):
    RELEASE_NOTES = {"1.2.0": "Initial release, the record schema of colliderml_ttbar_pu0_v20_pf 1.2.0."}
    DESCRIPTION = _DESCRIPTION
    LAYOUT = V22_PU200_LAYOUT
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=n) for n in v22_block_config_names(LAYOUT)]

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return split_sample_v22_block(Path(dl_manager.manual_dir), self.builder_config.name, self.LAYOUT)
