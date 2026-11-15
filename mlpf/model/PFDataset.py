import sys
import bisect
import os
import random
import resource
import math
from types import SimpleNamespace

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.utils.data

from mlpf.logger import _logger
from mlpf.conf import DatasetSamplerMode, MLPFConfig, OutputMode, Y_FEATURES, dataset_input_type_id, dataset_source_id
from mlpf.model.crops import choose_crops, crop_record, n_crops, positions


# https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
# file_system strategy can be better when hitting file descriptor limits
SHARING_STRATEGY = os.environ.get("MLPF_SHARING_STRATEGY", "file_system")
if SHARING_STRATEGY not in torch.multiprocessing.get_all_sharing_strategies():
    raise ValueError(f"Unsupported MLPF_SHARING_STRATEGY: {SHARING_STRATEGY}")

try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _logger.info(f"Initial RLIMIT_NOFILE: soft={soft}, hard={hard}")
    if soft < hard:
        _logger.info(f"Attempting to set RLIMIT_NOFILE soft limit to hard limit: {hard}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        _logger.info(f"New RLIMIT_NOFILE: soft={soft}, hard={hard}")
except Exception as e:
    _logger.warning(f"Could not set RLIMIT_NOFILE: {e}")


def gen_target_to_set(gen_target):
    """Set targets in the Y_FEATURES column order from gen_target rows, with log pt and log energy.

    gen_target columns are cls (0..5), charge, pt, eta, sin_phi, cos_phi, energy, ispu, gp_to_track, gp_to_cluster.
    jet_idx is -1 and particle_number is the gen_target row + 1.
    """
    gen = np.asarray(gen_target, dtype=np.float32).reshape(-1, 10)
    ys = np.zeros((len(gen), len(Y_FEATURES)), dtype=np.float32)
    for name, col in (("PDG", 0), ("charge", 1), ("eta", 3), ("sin_phi", 4), ("cos_phi", 5), ("ispu", 7), ("gp_to_track", 8), ("gp_to_cluster", 9)):
        ys[:, Y_FEATURES.index(name)] = gen[:, col]
    ys[:, Y_FEATURES.index("pt")] = np.log(np.clip(gen[:, 2], 1e-8, None))
    ys[:, Y_FEATURES.index("energy")] = np.log(np.clip(gen[:, 6], 1e-8, None))
    ys[:, Y_FEATURES.index("jet_idx")] = -1
    ys[:, Y_FEATURES.index("particle_number")] = np.arange(1, len(gen) + 1)
    return ys


# per-element and off-track fields of the ColliderML v20 TFDS; apply_colliderml_data keeps only what a flag reads
COLLIDERML_EXTRA_KEYS = ("X_time_layer", "frac_elem", "frac_target", "frac", "X_offtrack", "offtrack_target", "offtrack_own", "extras_stored")


HOST_TRACK_FRACTION = 0.5  # a target hosts a track when it owns the majority of its hits


def host_class(cls, tracked):
    """Class by host on class indices 0..5 (none, charged hadron, neutral hadron, photon, electron, muon), as the converter's
    map_neutral_to_charged and map_charged_to_neutral do: a track-hosted photon or neutral hadron becomes a charged hadron,
    a calorimeter-hosted electron a photon, and a calorimeter-hosted charged hadron or muon a neutral hadron."""
    cls, tracked = np.asarray(cls), np.asarray(tracked, bool)
    out = cls.copy()
    out[tracked & np.isin(cls, (2, 3))] = 1
    out[~tracked & (cls == 4)] = 3
    out[~tracked & np.isin(cls, (1, 5))] = 2
    return out


def apply_colliderml_data(ret, flags, feature_dim=None):
    """Drop the ColliderML extra fields from a record and apply the colliderml_data flags."""
    extras = {k: ret.pop(k) for k in COLLIDERML_EXTRA_KEYS if k in ret}
    if flags is not None and flags.use_layer:
        if int(extras.get("extras_stored", 0)) != 1:
            raise ValueError("colliderml_data.use_layer needs a ColliderML v20 TFDS of version 1.2.0")
        ret["X"] = np.concatenate([ret["X"], extras["X_time_layer"][:, 1:2]], axis=1)
    if flags is not None and flags.host_class_targets:
        gen = np.array(ret["gen_target"], dtype=np.float32).reshape(-1, 10)
        gen[:, 0] = host_class(gen[:, 0].astype(np.int64), (gen[:, 8] > HOST_TRACK_FRACTION) & (np.asarray(ret["gen_target_track"]) >= 0))
        ret["gen_target"] = gen
    if "extras_stored" in extras and feature_dim is not None and ret["X"].shape[1] != feature_dim:
        raise ValueError(f"X has {ret['X'].shape[1]} columns after the colliderml_data flags, the model reads {feature_dim}")
    return ret


def colliderml_skipped_keys(flags, feature_keys):
    """The extra fields among feature_keys that no colliderml_data flag reads; they are not parsed."""
    read = {"extras_stored"} | ({"X_time_layer"} if flags is not None and flags.use_layer else set())
    return tuple(k for k in COLLIDERML_EXTRA_KEYS if k in feature_keys and k not in read)


def skipping_reader(features, skip, decoders=None):
    """(parser, decoders) that read a record without the features in skip, which come back as None."""
    from tensorflow_datasets.core import example_parser

    specs = {k: v for k, v in features.get_serialized_info().items() if k not in skip}
    return example_parser.ExampleParserNp(specs), {**(decoders or {}), **{k: tfds.decode.SkipDecoding() for k in skip}}


def uses_hit_grouping(config):
    """Set output with hit grouping, the only reader of the element particle_number in training."""
    set_decoder = config.model.set_decoder
    return config.model.output_mode == OutputMode.SET and set_decoder is not None and set_decoder.hit_grouping


def align_particle_numbers(rec):
    """Renumber the element particle_number of rec["ytarget"] to gen_target row + 1.

    Element rows carry the allocator's particle numbers after merges, while gen_target lists every visible particle. The
    kept particles are an in-order subsequence of gen_target, and each host row keeps its gen_target gp_to_track and
    gp_to_cluster, so an in-order match on those two keys maps one numbering to the other.
    """
    gen = np.asarray(rec["gen_target"], dtype=np.float32).reshape(-1, 10)
    y = np.array(rec["ytarget"], dtype=np.float32)
    pn_index, gt_index, gc_index = (Y_FEATURES.index(k) for k in ("particle_number", "gp_to_track", "gp_to_cluster"))
    pn = y[:, pn_index].astype(np.int64)
    host = y[:, 0] != 0
    n_post = int(pn[host].max()) if host.any() else 0
    if n_post == len(gen):
        return
    post = np.zeros((n_post, 2), dtype=np.float32)
    post[pn[host] - 1] = y[host][:, [gt_index, gc_index]]
    mapping = np.zeros(n_post + 1, dtype=np.int64)
    j = 0
    for k in range(n_post):
        while j < len(gen) and not (gen[j, 8] == post[k, 0] and gen[j, 9] == post[k, 1]):
            j += 1
        if j == len(gen):
            raise ValueError(f"event {rec.get('event_id')}: kept particle {k + 1} of {n_post} has no gen_target row with its keys")
        mapping[k + 1] = j + 1
        j += 1
    y[:, pn_index] = mapping[np.clip(pn, 0, n_post)]
    rec["ytarget"] = y


class TFDSDataSource:
    def __init__(
        self,
        ds,
        sort,
        pad_to_multiple=None,
        feature_dim=None,
        max_open_readers=None,
        build_target_set=False,
        set_target_source="element_rows",
        align_particle_numbers=False,
        colliderml_data=None,
        raw=False,
    ):
        """raw: return the decoded record; the caller (CropDataset) finalizes it."""
        self.ds = ds
        tmp = self.ds.dataset_info
        self.ds.dataset_info = SimpleNamespace()
        self.ds.dataset_info.name = tmp.name
        self.ds.dataset_info.features = tmp.features
        self.ds.dataset_info.config_name = tmp.config_name
        self.sort = sort
        self.pad_to_multiple = pad_to_multiple
        self.feature_dim = feature_dim
        self.max_open_readers = max_open_readers
        self.build_target_set = build_target_set
        self.set_target_source = set_target_source
        self.align_particle_numbers = align_particle_numbers
        self.colliderml_data = colliderml_data
        feature_keys = tmp.features.keys() if hasattr(tmp.features, "keys") else ()
        self.skip_keys = colliderml_skipped_keys(colliderml_data, feature_keys)
        self._reader = None
        self.raw = raw

    def build_gen_target_set(self, ret):
        if self.align_particle_numbers:
            align_particle_numbers(ret)
        ret["ytarget_set"] = gen_target_to_set(ret["gen_target"])

    def _close_extra_readers(self):
        # ArrayRecordDataSource caches one reader per shard file forever, so the
        # number of open file descriptors grows with the number of distinct shards
        # touched. Cap the number of concurrently open readers and close the rest;
        # reads are positional and stateless, so closing and reopening is safe.
        if not self.max_open_readers:
            return
        readers = getattr(self.ds.data_source, "_readers", None)
        if readers is None:
            return
        open_indices = [idx for idx, reader in enumerate(readers) if reader is not None]
        if len(open_indices) > self.max_open_readers:
            for idx in open_indices[: len(open_indices) - self.max_open_readers]:
                try:
                    readers[idx].close()
                except Exception:
                    pass
                readers[idx] = None

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        assert len(item) == 1
        # getitems requires a list
        records = self.ds.data_source.__getitems__(item)
        self._close_extra_readers()

        features = self.ds.dataset_info.features
        if self.skip_keys:
            if self._reader is None:
                self._reader = skipping_reader(features, self.skip_keys, self.ds.decoders)
            parser, decoders = self._reader
            ret = [features.decode_example_np(parser.parse_example(record), decoders=decoders) for record in records]
        else:
            ret = [features.deserialize_example_np(record, decoders=self.ds.decoders) for record in records]
        assert len(ret) == 1
        ret = ret[0]
        ds_name = self.ds.dataset_info.name
        ret["source_id"] = np.int64(dataset_source_id(ds_name))
        ret["input_type_id"] = np.int64(dataset_input_type_id(ds_name))
        apply_colliderml_data(ret, self.colliderml_data, self.feature_dim)

        Xshape = ret["X"].shape
        _logger.debug(f"Getting item={item}, ds={ds_name}:{self.ds.dataset_info.config_name}, X={Xshape}")

        if self.raw:
            return ret
        return self.finalize(ret)

    def finalize(self, ret):
        """Per-record transforms after decoding, and after cropping: padding, sorting, label remapping and targets."""
        ds_name = self.ds.dataset_info.name
        if self.feature_dim is not None:
            if ret["X"].shape[1] > self.feature_dim:
                raise ValueError(f"Input feature dimension {ret['X'].shape[1]} exceeds configured feature_dim={self.feature_dim}")
            if ret["X"].shape[1] < self.feature_dim:
                ret["X"] = np.pad(
                    ret["X"],
                    ((0, 0), (0, self.feature_dim - ret["X"].shape[1])),
                    mode="constant",
                    constant_values=0,
                )

        # sort the elements in each event in pT descending order
        # the transformer is permutation-covariant, but this can be helpful for other types of models
        if self.sort:
            sortidx = np.argsort(ret["X"][:, 1])[::-1]
            ret["X"] = ret["X"][sortidx]
            ret["ycand"] = ret["ycand"][sortidx]
            ret["ytarget"] = ret["ytarget"][sortidx]
            if "elem_interior" in ret:
                ret["elem_interior"] = ret["elem_interior"][sortidx]

        # Pad elements to the nearest multiple if specified
        if self.pad_to_multiple and self.pad_to_multiple > 0:
            current_len = ret["X"].shape[0]
            if current_len % self.pad_to_multiple != 0:
                num_to_pad = self.pad_to_multiple - (current_len % self.pad_to_multiple)
                for key_to_pad in ["X", "ycand", "ytarget"]:
                    if key_to_pad in ret:  # Ensure key exists
                        array_to_pad = ret[key_to_pad]
                        pad_width = ((0, num_to_pad), (0, 0))  # Pad only the first axis
                        ret[key_to_pad] = np.pad(array_to_pad, pad_width, mode="constant", constant_values=0)
                if "elem_interior" in ret:
                    ret["elem_interior"] = np.pad(ret["elem_interior"], (0, num_to_pad), mode="constant", constant_values=False)

        if ds_name.startswith("cms_"):
            # track, target label neutral hadron -> reconstruct as charged hadron
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 1) & (ret["ytarget"][:, 0] == 2)] = 1

            # track, target label photon -> reconstruct as charged hadron
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 1) & (ret["ytarget"][:, 0] == 5)] = 1

            # ECAL cluster, target label charged hadron -> reconstruct as photon
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 4) & (ret["ytarget"][:, 0] == 1)] = 5

            # HCAL cluster, target label charged hadron -> reconstruct as neutral hadron
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 5) & (ret["ytarget"][:, 0] == 1)] = 2

            # ECAL cluster, target label electron -> reconstruct as photon
            # ret["ytarget"][:, 0][(ret["X"][:, 0]==4) & (ret["ytarget"][:, 0] == 6)] = 5

            ret["ytarget"][:, 0][(ret["X"][:, 0] == 5) & (ret["ytarget"][:, 0] == 6)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 4) & (ret["ytarget"][:, 0] == 7)] = 5
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 5) & (ret["ytarget"][:, 0] == 7)] = 2

            # HFEM cluster, reconstruct as HFEM
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 8) & (ret["ytarget"][:, 0] != 0)] = 4

            # HFHAD cluster, reconstruct as HFHAD
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 9) & (ret["ytarget"][:, 0] != 0)] = 3

            ret["ytarget"][:, 0][(ret["X"][:, 0] == 10) & (ret["ytarget"][:, 0] == 1)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 11) & (ret["ytarget"][:, 0] == 1)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 10) & (ret["ytarget"][:, 0] == 6)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 11) & (ret["ytarget"][:, 0] == 6)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 10) & (ret["ytarget"][:, 0] == 7)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 11) & (ret["ytarget"][:, 0] == 7)] = 2

            # set pt for HO which would otherwise be 0
            msk_ho = ret["X"][:, 0] == 10
            eta = ret["X"][:, 2][msk_ho]
            e = ret["X"][:, 5][msk_ho]
            ret["X"][:, 1][msk_ho] = np.sqrt(e**2 - (np.tanh(eta) * e) ** 2)

        if self.build_target_set and self.set_target_source == "gen_target":
            if "ytarget_set" not in ret:
                self.build_gen_target_set(ret)
        elif self.build_target_set:
            # Existing hit datasets store one full target row on an exclusive
            # representative hit for every particle. Other related hits contain
            # only particle_number, so nonzero class rows form the compact set.
            target_rows = ret["ytarget"][:, 0] != 0
            ytarget_set = ret["ytarget"][target_rows].copy()

            particle_number_idx = Y_FEATURES.index("particle_number")
            if ytarget_set.shape[1] > particle_number_idx and len(ytarget_set) > 0:
                particle_numbers = ytarget_set[:, particle_number_idx]
                # Some old datasets predate particle_number. If it is present,
                # require the representative rows to be one-per-particle.
                if np.any(particle_numbers != 0):
                    if np.any(particle_numbers == 0) or len(np.unique(particle_numbers)) != len(particle_numbers):
                        raise ValueError("Full ytarget rows must have unique, nonzero particle_number values in set mode")

            # Set outputs have no input-object anchor. Train pt and energy in
            # absolute log space instead of using the legacy log(target/input).
            if len(ytarget_set) > 0:
                ytarget_set[:, 2] = np.log(np.clip(ytarget_set[:, 2], 1e-8, None))
                ytarget_set[:, 6] = np.log(np.clip(ytarget_set[:, 6], 1e-8, None))
            ret["ytarget_set"] = ytarget_set

        # transform pt -> log(pt / elem pt), same for energy
        # where target does not exist, set to 0
        with np.errstate(divide="ignore"):
            target_pt = np.log(ret["ytarget"][:, 2] / ret["X"][:, 1])
        target_pt[np.isnan(target_pt)] = 0
        target_pt[np.isinf(target_pt)] = 0
        ret["ytarget_pt_orig"] = ret["ytarget"][:, 2].copy()
        ret["ytarget"][:, 2] = target_pt

        with np.errstate(divide="ignore"):
            target_e = np.log(ret["ytarget"][:, 6] / ret["X"][:, 5])
        target_e[ret["ytarget"][:, 0] == 0] = 0
        target_e[np.isnan(target_e)] = 0
        target_e[np.isinf(target_e)] = 0
        ret["ytarget_e_orig"] = ret["ytarget"][:, 6].copy()
        ret["ytarget"][:, 6] = target_e

        return ret

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return "TFDSDataSource(ds={}, pad_to_multiple={})".format(self.ds.__repr__(), self.pad_to_multiple)


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(
        self,
        data_dir,
        name,
        split,
        num_samples=None,
        sort=False,
        pad_to_multiple=512,
        feature_dim=None,
        max_open_readers=None,
        build_target_set=False,
        set_target_source="element_rows",
        align_particle_numbers=False,
        colliderml_data=None,
        raw=False,
    ):
        """
        Args
            data_dir: path to tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
            name: sample and version (e.g. `clic_edm_ttbar_pf:1.5.0`)
            split: "train" or "test" (if "valid" then will use "test")
        """
        if split == "valid":
            split = "test"

        try:
            builder = tfds.builder(name, data_dir=data_dir)
        except Exception as e:
            _logger.error(
                "Could not find dataset {} in {}, please check that you have downloaded the correct version of the dataset".format(name, data_dir)
            )
            _logger.error(e)
            sys.exit(1)

        _logger.debug(f"PFDataset opening dataset {name} in {builder.data_path} for split {split}")
        self.ds = TFDSDataSource(
            builder.as_data_source(split=split),
            sort=sort,
            pad_to_multiple=pad_to_multiple,
            feature_dim=feature_dim,
            max_open_readers=max_open_readers,
            build_target_set=build_target_set,
            set_target_source=set_target_source,
            align_particle_numbers=align_particle_numbers,
            colliderml_data=colliderml_data,
            raw=raw,
        )

        if num_samples and num_samples < len(self.ds):
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def __len__(self):
        return len(self.ds)


class CropDataset(torch.utils.data.Dataset):
    """Crops of the events of a ConcatDataset of raw TFDSDataSources.

    A key is epoch * N + event index (EpochKeySampler), so the crops of an event depend only on (seed, epoch, event_id).
    An item is the list of the event's finalized crops. Training draws crops_per_event crops per epoch with the run seed;
    validation always draws val_crops with val_seed at epoch 0.
    """

    def __init__(self, concat, mode, crop_cfg, seed=0):
        if mode not in ("train", "valid"):
            raise ValueError(f"CropDataset mode must be 'train' or 'valid', not {mode!r}")
        self.concat = concat
        self.mode = mode
        self.cfg = crop_cfg
        self.seed = seed
        self.n_crops = n_crops(crop_cfg.eta_edges, crop_cfg.nphi)

    def __len__(self):
        return len(self.concat)

    def __getitem__(self, key):
        epoch, idx = divmod(int(key), len(self))
        rec = self.concat[idx]
        src = self.concat.datasets[bisect.bisect_right(self.concat.cumulative_sizes, idx)]
        src = getattr(src, "dataset", src)  # torch Subset with num_samples
        if src.build_target_set:
            # set targets before cropping, so the element particle numbers keep pointing at them
            src.build_gen_target_set(rec)
        event_id = int(rec["event_id"])
        if self.mode == "train":
            crops = choose_crops(self.seed, epoch, event_id, self.cfg.crops_per_event, self.n_crops)
        else:
            crops = choose_crops(self.cfg.val_seed, 0, event_id, self.cfg.val_crops, self.n_crops)
        pos = positions(rec)
        out = []
        for crop in crops:
            r = crop_record(rec, crop, self.cfg.guard, self.cfg.eta_edges, self.cfg.nphi, pos=pos)
            r["crop_id"] = np.int64(crop)
            out.append(src.finalize(r))
        return out


class PFBatch:
    def __init__(self, **kwargs):
        self.attrs = list(kwargs.keys())

        # write out the possible attributes here explicitly
        self.X = kwargs["X"]
        self.ytarget = kwargs.get("ytarget")
        self.ytarget_set = kwargs.get("ytarget_set")
        self.ytarget_pt_orig = kwargs.get("ytarget_pt_orig", None)
        self.ytarget_e_orig = kwargs.get("ytarget_e_orig", None)
        self.pythia = kwargs.get("pythia", None)
        self.ycand = kwargs.get("ycand", None)
        self.genmet = kwargs.get("genmet", None)
        self.genjets = kwargs.get("genjets", None)
        # gen reference including pileup (ColliderML TFDS >= 1.4.0, where genmet/genjets are the hard scatter only)
        self.genmet_withpu = kwargs.get("genmet_withpu", None)
        self.genjets_withpu = kwargs.get("genjets_withpu", None)
        self.targetjets = kwargs.get("targetjets", None)
        self.source_id = kwargs.get("source_id", None)
        self.input_type_id = kwargs.get("input_type_id", None)
        self.elem_interior = kwargs.get("elem_interior", None)  # crops: 1 = interior element, 0 = guard or padding
        self.event_id = kwargs.get("event_id", None)
        self.crop_id = kwargs.get("crop_id", None)
        self.mask = self.X[:, :, 0] != 0
        self.target_mask = self.ytarget_set[:, :, 0] != 0 if self.ytarget_set is not None else None

    def to(self, device, **kwargs):
        attrs = {}
        for attr in self.attrs:
            this_attr = getattr(self, attr)
            attrs[attr] = this_attr.to(device, **kwargs) if this_attr is not None else None
        return PFBatch(**attrs)


# pads items with variable lengths (seq_len1, seq_len2, ...) to [batch, max(seq_len), ...]
class Collater:
    def __init__(self, per_particle_keys_to_get, per_event_keys_to_get, **kwargs):
        super(Collater, self).__init__(**kwargs)
        self.per_particle_keys_to_get = per_particle_keys_to_get  # these quantities are a variable-length tensor per each event
        self.per_event_keys_to_get = per_event_keys_to_get  # these quantities are one value (scalar) per event

    def __call__(self, inputs):
        ret = {}
        # CropDataset items are lists of crops: the batch is events x crops
        inputs = [rec for inp in inputs for rec in (inp if isinstance(inp, list) else [inp])]

        # per-particle quantities need to be padded across events of different size
        for key_to_get in self.per_particle_keys_to_get:
            ret[key_to_get] = torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(inp[key_to_get], dtype=torch.float32) for inp in inputs],
                batch_first=True,
            )

        # per-event quantities can be stacked across events
        for key_to_get in self.per_event_keys_to_get:
            if not all(key_to_get in inp for inp in inputs):
                continue
            ret[key_to_get] = torch.stack([torch.as_tensor(inp[key_to_get]) for inp in inputs])
        return PFBatch(**ret)


class ShardConsecutiveSampler(torch.utils.data.Sampler):
    """
    Samples elements consecutively from each sub-dataset in a ConcatDataset.
    This helps to minimize the number of open file descriptors when each sub-dataset
    is backed by a separate file and the file is opened lazily.
    Batches then come from one sub-dataset at a time: for a group of several samples,
    InterleavedShardSampler (sampler_mode interleaved-shards) mixes them in every batch.
    """

    def __init__(self, concat_dataset, shuffle=True, seed=0):
        self.concat_dataset = concat_dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        shard_indices_lists = []
        start_idx = 0
        for end_idx in self.concat_dataset.cumulative_sizes:
            shard_indices_lists.append(list(range(start_idx, end_idx)))
            start_idx = end_idx

        if self.shuffle:
            # Shuffle the order of shards
            rng.shuffle(shard_indices_lists)
            # Shuffle within each shard
            for shard_list in shard_indices_lists:
                rng.shuffle(shard_list)

        indices = [idx for shard_list in shard_indices_lists for idx in shard_list]
        return iter(indices)

    def __len__(self):
        return len(self.concat_dataset)


def _build_interleaved_shard_indices(concat_dataset, shuffle=True, seed=0):
    """Event order mixing the ConcatDataset shards at the same relative pace.

    Each shard is shuffled and its k-th of n events is placed at (k + 0.5) / n, its fractional progress through
    the shard, and all events are sorted by that position (ties in shard order). Any prefix of the order then
    holds every shard's proportional share to within about one event per shard, so small shards do not run out
    early.
    """
    rng = np.random.default_rng(seed)
    indices, positions = [], []
    start_idx = 0
    for end_idx in concat_dataset.cumulative_sizes:
        # an empty shard contributes empty arrays
        num_events = end_idx - start_idx
        shard_indices = np.arange(start_idx, end_idx)
        if shuffle:
            rng.shuffle(shard_indices)
        indices.append(shard_indices)
        positions.append((np.arange(num_events) + 0.5) / num_events)
        start_idx = end_idx

    order = np.argsort(np.concatenate(positions), kind="stable")
    return np.concatenate(indices)[order].tolist()


class InterleavedShardSampler(torch.utils.data.Sampler):
    """Interleave events across ConcatDataset shards, in proportion to the shard sizes over the whole epoch.

    This avoids long single-source stretches when a training config combines
    multiple samples under one physical dataset, without a tail of only the
    largest shards at the end of the epoch (see _build_interleaved_shard_indices).
    """

    def __init__(self, concat_dataset, shuffle=True, seed=0):
        self.concat_dataset = concat_dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(_build_interleaved_shard_indices(self.concat_dataset, shuffle=self.shuffle, seed=self.seed + self.epoch))

    def __len__(self):
        return len(self.concat_dataset)


class _DistributedShardSampler(torch.utils.data.distributed.DistributedSampler):
    """Base of the distributed shard samplers: every rank builds the same global event order (_global_indices), pads it
    (or truncates it with drop_last) to total_size as DistributedSampler does, and takes its own contiguous block."""

    def __init__(self, dataset, world_size=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def _global_indices(self):
        raise NotImplementedError

    def __iter__(self):
        indices = self._global_indices()

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedShardConsecutiveSampler(_DistributedShardSampler):
    """
    A distributed version of ShardConsecutiveSampler.
    Each rank handles a subset of shards, ensuring that each node only opens its subset of shards:
    its contiguous block of the shard-consecutive order spans only a few shards.
    """

    def _global_indices(self):
        rng = random.Random(self.seed + self.epoch)

        # 1. Get all shard ranges
        shard_ranges = []
        start_idx = 0
        for end_idx in self.dataset.cumulative_sizes:
            shard_ranges.append((start_idx, end_idx))
            start_idx = end_idx

        # 2. Shuffle shard order (deterministically across nodes)
        if self.shuffle:
            rng.shuffle(shard_ranges)

        # 3. Build the full list of indices
        all_indices = []
        for s_start, s_end in shard_ranges:
            shard_indices = list(range(s_start, s_end))
            if self.shuffle:
                # Use a deterministic seed for internal shard shuffling that is the same across all ranks
                # so that all_indices is identical on all ranks.
                # We use s_start as part of the seed to ensure different shards have different internal shuffles.
                shard_rng = random.Random(self.seed + self.epoch + s_start)
                shard_rng.shuffle(shard_indices)
            all_indices.extend(shard_indices)
        return all_indices


class DistributedInterleavedShardSampler(_DistributedShardSampler):
    """Distributed version of InterleavedShardSampler.

    Each rank gets a contiguous block of the interleaved order: every block of it is proportional, while a strided
    split (indices[rank::num_replicas]) would alias with its near-periodic pattern and skew each rank's mix.
    """

    def _global_indices(self):
        return _build_interleaved_shard_indices(self.dataset, shuffle=self.shuffle, seed=self.seed + self.epoch)


class ResumableSampler(torch.utils.data.Sampler):
    """A wrapper for a sampler that allows saving and restoring the state."""

    def __init__(self, sampler):
        _logger.debug("Creating ResumableSampler.")
        self.sampler = sampler
        self.start_index = 0
        self.name = ""

    def __iter__(self):
        indices = list(self.sampler)
        return iter(indices[self.start_index :])

    def __len__(self):
        return len(self.sampler) - self.start_index

    def load_state_dict(self, state_dict):
        # start_index counts whole batches, so a restored partial last batch would point past the end
        self.start_index = min(state_dict["start_index"], len(self.sampler))
        _logger.info(f"ResumableSampler {self.name} loading state: start_index={self.start_index}")

    def reset(self):
        _logger.info(f"ResumableSampler {self.name} resetting")
        self.load_state_dict({"start_index": 0})

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class EpochKeySampler(torch.utils.data.Sampler):
    """Yields epoch * num_events + index for the indices of inner, so the epoch reaches persistent workers with each key."""

    def __init__(self, inner, num_events):
        self.inner = inner
        self.num_events = num_events
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.inner, "set_epoch"):
            self.inner.set_epoch(epoch)

    def __iter__(self):
        offset = self.epoch * self.num_events
        return iter([offset + idx for idx in self.inner])

    def __len__(self):
        return len(self.inner)


class InterleavedIterator(object):
    """Will combine DataLoaders of different lengths and batch sizes."""

    def __init__(self, data_loaders, names=None):
        _logger.info(f"Creating InterleavedIterator with {len(data_loaders)} data loaders.")
        self.data_loaders = data_loaders
        self.names = list(names) if names is not None else [str(i) for i in range(len(data_loaders))]
        self.last_loader_index = None  # loader of the most recently yielded batch
        self.data_loaders_iter = [iter(dl) for dl in data_loaders]
        dl_lens = [len(dl) for dl in data_loaders]
        _logger.debug(f"InterleavedIterator len(data_loaders)={dl_lens}")
        max_loader_size = max(dl_lens)

        self.loader_ds_indices = []

        # iterate loaders interleaved up to the maximum size
        for i in range(max_loader_size):
            for iloader, loader in enumerate(data_loaders):
                if i < len(loader):
                    self.loader_ds_indices.append(iloader)

        # cursor to keep track which data loader index to yield from
        self.cur_index = 0
        self.batches_yielded_per_loader = [0] * len(self.data_loaders)
        self.name = ""

        _logger.debug(f"InterleavedIterator at {self.cur_index}/{len(self.loader_ds_indices)}")

    def reset(self):
        _logger.debug(f"Resetting InterleavedIterator {self.name} state")
        self.cur_index = 0
        for loader in self.data_loaders:
            if hasattr(loader.sampler, "reset"):
                loader.sampler.reset()
        self.data_loaders_iter = [iter(dl) for dl in self.data_loaders]
        self.batches_yielded_per_loader = [0] * len(self.data_loaders)

    def __iter__(self):
        # Only reset if the iterator is exhausted
        _logger.debug(f"Resetting InterleavedIterator {self.name}: {self.cur_index}/{len(self.loader_ds_indices)}")
        if self.cur_index >= len(self.loader_ds_indices):
            _logger.debug(f"Resetting exhausted InterleavedIterator {self.name}, {self.cur_index}>={len(self.loader_ds_indices)}.")
            self.reset()
        return self

    def __next__(self):
        _logger.debug(f"InterleavedIterator {self.name}.__next__ {self.cur_index}/{len(self.loader_ds_indices)}")
        if self.cur_index >= len(self.loader_ds_indices):
            _logger.debug(f"InterleavedIterator {self.name}.__next__ raising StopIteration")
            raise StopIteration

        iloader = self.loader_ds_indices[self.cur_index]
        try:
            ret = next(self.data_loaders_iter[iloader])
        # This should not happen, and is here for debugging
        except StopIteration as e:
            _logger.error(
                f"Unexpected StopIteration from data loader {self.name}:"
                + f" iloader={iloader}, "
                + f"batches_yielded_per_loader={self.batches_yielded_per_loader} "
                + f"len(loader)={len(self.data_loaders_iter[iloader])}"
            )
            raise e
        self.cur_index += 1
        self.batches_yielded_per_loader[iloader] += 1
        self.last_loader_index = iloader
        return ret

    def __len__(self):
        # batches per epoch; unlike the lengths of the data loaders, it does not shrink when the state is restored mid-epoch
        return len(self.loader_ds_indices)

    def state_dict(self):
        return {
            "cur_index": self.cur_index,
            "batches_yielded_per_loader": self.batches_yielded_per_loader,
        }

    def load_state_dict(self, state_dict):
        self.cur_index = state_dict["cur_index"]
        self.batches_yielded_per_loader = state_dict["batches_yielded_per_loader"]
        _logger.info(
            f"InterleavedIterator {self.name} loading state: cur_index={self.cur_index}, batches_yielded_per_loader={self.batches_yielded_per_loader}"
        )

        for i, loader in enumerate(self.data_loaders):
            start_index = self.batches_yielded_per_loader[i] * loader.batch_size
            _logger.info(f"InterleavedIterator {self.name} advancing sampler {i} to {start_index}")
            loader.sampler.load_state_dict({"start_index": start_index})

        # create fresh iterators from the original dataloaders
        # these will use the advanced samplers and have fresh workers and empty queues
        self.data_loaders_iter = [iter(dl) for dl in self.data_loaders]


def next_group(counts, weights):
    """The group to draw the next batch from: the one furthest behind its share weights[i] / sum(weights).

    Group i's next batch is due at (counts[i] + 0.5) / weights[i]; picking the earliest spreads every group's
    batches evenly over the steps. Deterministic in the counts, so all DDP ranks pick the same group.
    """
    return min(range(len(weights)), key=lambda i: ((counts[i] + 0.5) / weights[i], i))


class WeightedMixIterator(object):
    """Endless training iterator over one DataLoader per dataset group.

    Each step yields one batch from the group picked by next_group, so group i gets a weights[i] / sum(weights)
    share of the steps (weights=None: proportional to each group's batches per epoch). Each group loops over its
    own data and starts a new sampler epoch when exhausted. Empty groups are skipped.

    State: each group's sampler position (epochs, batches_in_epoch) and its batches picked under the current
    weights (scheduled). Resuming with other weights resets scheduled, so the new shares apply from then on.
    """

    def __init__(self, data_loaders, weights=None, names=None, world_size=1):
        _logger.info(f"Creating WeightedMixIterator with {len(data_loaders)} data loaders.")
        self.data_loaders = data_loaders
        self.lens = [len(dl) for dl in data_loaders]
        self.weights = list(weights) if weights is not None else [float(n) for n in self.lens]
        if len(self.weights) != len(data_loaders):
            raise ValueError(f"got {len(self.weights)} weights for {len(data_loaders)} data loaders")
        self.names = list(names) if names is not None else [str(i) for i in range(len(data_loaders))]
        self.events_per_batch = [dl.batch_size * world_size for dl in data_loaders]  # summed over ranks
        self.active = [i for i, n in enumerate(self.lens) if n > 0]
        for i, n in enumerate(self.lens):
            if n == 0:
                _logger.warning(f"WeightedMixIterator: group {self.names[i]} has no batches and is left out of the schedule")
        self.scheduled = [0] * len(data_loaders)  # batches picked per loader under the current weights
        self.batches_in_epoch = [0] * len(data_loaders)
        self.epochs = [0] * len(data_loaders)
        self.iterators = [None] * len(data_loaders)
        self.last_loader_index = None  # group of the most recently yielded batch
        self.name = ""

    @property
    def batches(self):
        """Batches yielded per loader over all epochs."""
        return [e * n + b for e, n, b in zip(self.epochs, self.lens, self.batches_in_epoch)]

    def __iter__(self):
        # start every group's workers together, rather than each at its group's first batch
        for i in self.active:
            if self.iterators[i] is None:
                self._start(i)
        return self

    def _start(self, i):
        """A fresh iterator for group i, from its current epoch and batch."""
        loader = self.data_loaders[i]
        loader.sampler.set_epoch(self.epochs[i])
        loader.sampler.load_state_dict({"start_index": self.batches_in_epoch[i] * loader.batch_size})
        self.iterators[i] = iter(loader)

    def _next_from(self, i):
        if self.iterators[i] is None:
            self._start(i)
        try:
            return next(self.iterators[i])
        except StopIteration:
            self.epochs[i] += 1
            self.batches_in_epoch[i] = 0
            _logger.info(f"WeightedMixIterator {self.name}: group {self.names[i]} starts epoch {self.epochs[i]}")
            self._start(i)
            return next(self.iterators[i])

    def __next__(self):
        if not self.active:
            raise RuntimeError(f"WeightedMixIterator {self.name}: every data loader is empty")
        pick = next_group([self.scheduled[i] for i in self.active], [self.weights[i] for i in self.active])
        i = self.active[pick]
        batch = self._next_from(i)
        self.scheduled[i] += 1
        self.batches_in_epoch[i] += 1
        self.last_loader_index = i
        _logger.debug(f"WeightedMixIterator {self.name}: group {self.names[i]} batch {self.batches_in_epoch[i]} of epoch {self.epochs[i]}")
        return batch

    def __len__(self):
        # batches in one pass over every group; the iterator itself never ends
        return sum(self.lens)

    def describe(self, num_steps):
        """One line per group: weight, step share, batch size and the expected passes over its data."""
        total = sum(self.weights[i] for i in self.active)
        lines = []
        for i in range(len(self.data_loaders)):
            share = self.weights[i] / total if i in self.active else 0.0
            passes = num_steps * share / self.lens[i] if self.lens[i] else 0.0
            lines.append(
                f"{self.names[i]}: weight={self.weights[i]:g} step_share={share:.3f} batches_per_epoch={self.lens[i]} "
                f"events_per_step={self.events_per_batch[i]} expected_passes_in_{num_steps}_steps={passes:.2f}"
            )
        return lines

    def stats(self):
        """Per-group scalars for logging: step fraction so far, passes over its data, events seen."""
        batches = self.batches
        total = max(1, sum(batches))
        out = {}
        for i, name in enumerate(self.names):
            out[f"{name}/step_fraction"] = batches[i] / total
            out[f"{name}/passes"] = batches[i] / self.lens[i] if self.lens[i] else 0.0
            out[f"{name}/events_seen"] = batches[i] * self.events_per_batch[i]
        return out

    def state_dict(self):
        return {
            "batches_in_epoch": list(self.batches_in_epoch),
            "epochs": list(self.epochs),
            "scheduled": list(self.scheduled),
            "weights": list(self.weights),
        }

    def load_state_dict(self, state_dict):
        if "loader_state_dict" in state_dict:
            # checkpoint from the earlier EndlessIterator(InterleavedIterator): one shared epoch
            in_epoch = list(state_dict["loader_state_dict"]["batches_yielded_per_loader"])
            state_dict = {"batches_in_epoch": in_epoch, "epochs": [state_dict["epoch"]] * len(in_epoch)}
            _logger.warning(f"WeightedMixIterator {self.name}: converted a pre-weighting loader state; the batch order changes from here on")
        for key in ("batches_in_epoch", "epochs"):
            if len(state_dict[key]) != len(self.data_loaders):
                raise ValueError(f"loader state has {len(state_dict[key])} groups, this run has {len(self.data_loaders)}")
        self.batches_in_epoch = list(state_dict["batches_in_epoch"])
        self.epochs = list(state_dict["epochs"])
        if state_dict.get("weights") == self.weights:
            self.scheduled = list(state_dict["scheduled"])
        else:
            # other weights (or batches per epoch, with the default weights): next_group would catch up on the old counts
            # with a long run of one group, so the schedule restarts and the new shares apply from this step on
            _logger.warning(
                f"WeightedMixIterator {self.name}: the loader state has step weights {state_dict.get('weights')}, this run "
                f"{self.weights}; the step shares follow the new weights from here on"
            )
            self.scheduled = [0] * len(self.data_loaders)
        _logger.info(f"WeightedMixIterator {self.name}: groups {self.names} resume at batches {self.batches_in_epoch} of epochs {self.epochs}")
        # fresh iterators, started from the restored positions, with fresh workers and empty queues
        self.iterators = [None] * len(self.data_loaders)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        if worker_id == 0:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            _logger.info(f"Worker 0 RLIMIT_NOFILE: soft={soft}, hard={hard}")
    except Exception:
        pass


def _train_mix_iterator(physical_groups, data_loaders, world_size):
    """WeightedMixIterator over the train groups, with their configured weights (all or none, see MLPFConfig)."""
    names = list(physical_groups)
    weights = [physical_groups[name].weight for name in names]
    return WeightedMixIterator(data_loaders, weights=None if weights[0] is None else weights, names=names, world_size=world_size)


def get_interleaved_dataloaders(world_size, rank, config: MLPFConfig, use_cuda, use_ray, shuffle_train=True):
    loaders = {}
    samplers = {}
    # build train, valid dataset and dataloaders
    for split in ["train", "valid"]:
        loaders[split] = []
        samplers[split] = []
        dataset_config = getattr(config, f"{split}_dataset")
        if dataset_config is None:
            continue

        for type_ in dataset_config[config.dataset.value]:
            dataset = []
            physical_ds = dataset_config[config.dataset.value][type_]
            for sample_name, sample in physical_ds.samples.items():
                version = sample.version
                split_configs = sample.splits
                _logger.info(f"sample={sample_name} split={split} split_configs={split_configs}")

                nevents = None
                n_split_val = getattr(config, f"n{split}")
                split_configs_to_use = list(split_configs)
                if n_split_val is not None:
                    if n_split_val < len(split_configs_to_use):
                        split_configs_to_use = split_configs_to_use[:n_split_val]
                    nevents = max(1, n_split_val // len(split_configs_to_use))

                for split_config in split_configs_to_use:
                    ds = PFDataset(
                        config.data_dir_for(sample_name),
                        f"{sample_name}/{split_config}:{version}",
                        split,
                        num_samples=nevents,
                        sort=config.sort_data,
                        pad_to_multiple=config.pad_to_multiple_elements,
                        feature_dim=config.input_dim,
                        max_open_readers=config.max_open_readers,
                        build_target_set=config.model.output_mode == OutputMode.SET,
                        set_target_source=config.set_target_source,
                        align_particle_numbers=uses_hit_grouping(config),
                        colliderml_data=config.colliderml_data,
                        raw=config.crop.enabled,
                    ).ds

                    if (rank == 0) or (rank == "cpu"):
                        _logger.info(f"{split}_dataset: {sample_name}, {len(ds)}", color="blue")

                    dataset.append(ds)
            dataset = torch.utils.data.ConcatDataset(dataset)

            shuffle = False
            if shuffle_train:
                shuffle = split == "train"
            sampler_mode = DatasetSamplerMode(config.sampler_mode)
            _logger.info(f"{split}_dataset sampler_mode={sampler_mode.value} shuffle={shuffle}")
            if world_size > 1 and sampler_mode == DatasetSamplerMode.INTERLEAVED_SHARDS:
                sampler = DistributedInterleavedShardSampler(dataset, world_size=world_size, rank=rank, shuffle=shuffle, seed=config.seed)
            elif world_size > 1:
                sampler = DistributedShardConsecutiveSampler(dataset, world_size=world_size, rank=rank, shuffle=shuffle, seed=config.seed)
            elif sampler_mode == DatasetSamplerMode.INTERLEAVED_SHARDS:
                sampler = InterleavedShardSampler(dataset, shuffle=shuffle, seed=config.seed)
            else:
                sampler = ShardConsecutiveSampler(dataset, shuffle=shuffle, seed=config.seed)

            if config.crop.enabled:
                sampler = EpochKeySampler(sampler, len(dataset))
                dataset = CropDataset(dataset, split, config.crop, seed=config.seed)
            sampler = ResumableSampler(sampler)
            sampler.name = f"{type_}:{split}"

            # build dataloaders
            batch_size = physical_ds.batch_size * config.gpu_batch_multiplier
            per_particle_keys = ["X", "ytarget"]
            if config.model.output_mode == OutputMode.SET:
                per_particle_keys.append("ytarget_set")
            per_event_keys = ["genmet", "source_id", "input_type_id"]
            if config.crop.enabled:
                per_particle_keys.append("elem_interior")
                per_event_keys += ["event_id", "crop_id"]
            loader_generator = torch.Generator()
            rank_index = int(rank) if isinstance(rank, int) else 0
            split_offset = 0 if split == "train" else 1000
            loader_generator.manual_seed(config.seed + 10_000 * rank_index + split_offset + len(loaders[split]))

            worker_kwargs = {}
            if config.num_workers > 0:
                worker_kwargs = {
                    "prefetch_factor": config.prefetch_factor,
                    "worker_init_fn": set_worker_sharing_strategy,
                    "persistent_workers": True,
                }

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=Collater(per_particle_keys, per_event_keys),
                sampler=sampler,
                num_workers=config.num_workers,
                # Training uses fixed-size batches, but a bounded validation
                # sample can be smaller than one per-rank batch (for example,
                # nvalid=100 with 8 ranks and batch_size=64). Keep that partial
                # validation batch so every rank participates in evaluation.
                drop_last=split == "train",
                generator=loader_generator,
                **worker_kwargs,
            )

            loaders[split].append(loader)
            samplers[split].append(sampler)

        if split == "train":
            loaders[split] = _train_mix_iterator(dataset_config[config.dataset.value], loaders[split], world_size)
            if (rank == 0) or (rank == "cpu"):
                for line in loaders[split].describe(config.num_steps):
                    _logger.info(f"train mix: {line}")
        else:
            loaders[split] = InterleavedIterator(loaders[split], names=list(dataset_config[config.dataset.value]))
        loaders[split].name = split

    return loaders, samplers
