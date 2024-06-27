from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import concurrent.futures
from beat_this.utils import PAD_TOKEN, index_to_framewise
from beat_this.dataset.augment import precomputed_augmentation_filenames, augment_pitchtempo, augment_mask
from beat_this.utils import load_spect
import torch.nn.functional as F

DATASET_INFO = {
    "gtzan" : {"beat": True, "downbeat" : True},
    "hainsworth" : {"beat": True, "downbeat" : True},
    "ballroom" : {"beat": True, "downbeat" : True},
    "hjdb" : {"beat": True, "downbeat" : True},
    "beatles" : {"beat": True, "downbeat" : True},
    "rwc" : {"beat": True, "downbeat" : True},
    "harmonix" : {"beat": True, "downbeat": True},
    "tapcorrect" : {"beat": True, "downbeat": True},
    "jaah" : {"beat": True, "downbeat": True},
    "filosax" : {"beat": True, "downbeat": True},
    "asap" : {"beat": True, "downbeat": True},
    "groove_midi" : {"beat": True, "downbeat": True},
    "guitarset" : {"beat": True, "downbeat": True},
    "candombe" : {"beat": True, "downbeat": True},
    "simac" : {"beat": True, "downbeat" : False},
    "smc" : {"beat": True, "downbeat" : False},
}

class BeatTrackingDataset(Dataset):
    """
    A PyTorch Dataset for beat tracking. This dataset loads preprocessed spectrograms and beat annotations
    from a given data folder and provides them for training or evaluation.

    Args:
        metadata_df (pd.DataFrame): A DataFrame containing metadata about the dataset items.
            Each row should represent one item and contain at least a 'spect_folder' column.
        data_folder (Path or str): The base folder where the data is stored.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        train_length (int, optional): The length of the training sequences in frames. If None the entire piece is used. Defaults to 1500.
        deterministic (bool, optional): If True, the dataset always returns the same sequence for a given index.
            Defaults to False.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Possible keys are "tempo", "pitch", and "mask". Defaults to an empty dictionary.
    """

    def __init__(self, metadata_df: pd.DataFrame,
                 data_folder,
                 spect_fps=50,
                 train_length=1500, deterministic=False,
                 augmentations={}):
        self.spect_basepath = data_folder / "preprocessed" / "spectrograms"
        self.annotation_basepath = data_folder / "beat_annotations"
        self.fps = spect_fps
        self.train_length = train_length
        self.deterministic = deterministic
        self.augmentations = augmentations
        # load the annotations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            items = executor.map(self._load_dataset_item,
                                 (row for _, row in metadata_df.iterrows()))
        self.items = [item for item in items if item is not None]

    def _load_dataset_item(self, df_row):
        # stop if the audio is not there
        spect_folder = self.spect_basepath / df_row["spect_folder"]
        # check if all the necessary files are there
        for aug_filename in precomputed_augmentation_filenames(self.augmentations):
            if not (spect_folder / aug_filename).exists():
                print(f"Skipping {spect_folder} because not all necessary spectrograms are there.")
                return

        # load beat and produce a default if beat values are not found
        annotation_path = self.annotation_basepath / df_row["beat_path"]
        beat_annotation = np.loadtxt(annotation_path)
        if beat_annotation.ndim == 2 :
            beat_time = beat_annotation[:, 0]
            beat_value = beat_annotation[:, 1].astype(int)
        else:
            beat_time = beat_annotation
            beat_value = np.zeros_like(beat_time, dtype=np.int32)

        # stop if the annotations that are supposed to be there are not there
        if DATASET_INFO[df_row["dataset"]]["downbeat"]:
            if beat_annotation.ndim != 2 :
                print(f"Skipping {df_row['beat_path']} because it has {beat_annotation.ndim} columns but downbeat is supposed to be there.")
                return

        # create a downbeat mask to handle the case where the downbeat is not annotated
        downbeat_mask = DATASET_INFO[df_row["dataset"]]["downbeat"]
        # select all values in columns that start with spect_len, e.g. spect_len_ts-20
        spect_lengths = {int(key.replace("spect_len_ts","")): int(value) for key, value in df_row.items() if key.startswith("spect_len")}
        # take care of different subsections of rwc for the dataset name
        dataset_name = df_row["dataset"] if df_row["dataset"] != "rwc" else "rwc_" + spect_folder.name.split("_")[1]
        return {'spect_folder': str(spect_folder),
                'beat_time': beat_time,
                'beat_value': beat_value,
                'downbeat_mask': downbeat_mask,
                'dataset': dataset_name,
                'spect_lengths': spect_lengths,
                }


    def get_beat_count(self, index):
        """Return number of beats (including downbeats) of given item."""
        return len(self.items[index]["beat_time"])

    def get_downbeat_count(self, index):
        """Return number of downbeats of given item."""
        return (self.items[index]["beat_value"] == 1).sum()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if isinstance(index, (int,np.int64)): # when index is a single int
            item = self.items[index]

            # select a pitch shift and time stretch
            item = augment_pitchtempo(item, self.augmentations)
            # define the excerpt to use
            original_length = item["spect_length"]
            if self.train_length is not None:
                longer = original_length - self.train_length
            else:
                longer = 0
            if longer > 0: # if the piece is longer than the desired length
                if self.deterministic:
                    # select the middle of the excerpt
                    start_frame = longer // 2
                else:
                    start_frame = np.random.randint(0, longer)
                end_frame = start_frame + self.train_length
            else:
                start_frame = 0
                end_frame = original_length

            # load spectrogram
            spect = load_spect(item["spect_path"], start=start_frame, stop=end_frame)

            # augment the spectrogram with mask augmentation (if required)
            spect = augment_mask(spect, self.augmentations, self.fps)

            # prepare annotations
            framewise_truth_beat, framewise_truth_downbeat, truth_orig_beat, truth_orig_downbeat = prepare_annotations(item, start_frame, end_frame, self.fps)
            
            # restructure the item dict with the correct training information
            item = {"spect": spect,
                    "spect_path": str(item["spect_path"]),
                    "dataset" : item["dataset"],
                    "start_frame": start_frame,
                    "truth_beat": framewise_truth_beat,
                    "truth_downbeat": framewise_truth_downbeat,
                    "downbeat_mask": torch.as_tensor(item["downbeat_mask"]),
                    "padding_mask": np.ones(self.train_length, dtype=bool) if self.train_length is not None else np.ones(original_length, dtype=bool),
                    "dataset": item["dataset"],
                    "truth_orig_beat": truth_orig_beat,
                    "truth_orig_downbeat": truth_orig_downbeat,
                    }
            
            # pad all framewise tensors if needed
            if longer < 0:
                item["spect"] = np.pad(item["spect"], [(0, -longer), (0, 0)], constant_values=0)
                for k in "truth_beat", "truth_downbeat":
                    item[k] = np.pad(item[k], [(0, -longer)], constant_values=0)
                item["padding_mask"][longer:] = 0
            return item

        else: # when index is a list of ints
            return [self[i] for i in index]

class BeatDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for beat tracking. This DataModule handles the loading and preprocessing of the
    BeatTrackingDataset and prepares it for use with a PyTorch Lightning model.
    It can produce cross-validation or single  train/val/test splits.

    Args:
        data_dir (Path or str): The parent directory where the data (spectrograms and beat labels) is stored.
        batch_size (int, optional): The size of the batches to be generated by the DataLoader. Defaults to 8.
        train_length (int, optional): The length of the subsequences in frames. If None, the entire pieces are returner. Defaults to 1500.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 20.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Defaults to {"pitch": {"min": -5, "max": 6}, "time": {"min": -20, "max": 20, "stride": 4}}.
        test_dataset (str, optional): The name of the dataset to use for testing. Defaults to "gtzan".
        hung_data (bool, optional): If True, only use the datasets from the Hung et al. paper for training; validation is still on all datasets. Defaults to False.
        no_val (bool, optional): If True, train on all train+val data and do not use a validation set; for compatibility reason, the validation metrics are still computed, but are not meaningful. Defaults to False.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        length_based_oversampling_factor (int, optional): The factor by which to oversample based on sequence length. Defaults to 0.
        fold (int, optional): The fold number for cross-validation. If None, the single split is used. Defaults to None.
    """
    def __init__(self, data_dir, batch_size=8,
                 train_length=1500, num_workers=20, augmentations={"pitch": {"min": -5, "max": 6}, "tempo": {"min": -20, "max": 20, "stride": 4}},
                 test_dataset="gtzan", hung_data = False, no_val= False, spect_fps=50,
                 length_based_oversampling_factor=0, fold=None):
        super().__init__()
        self.save_hyperparameters()
        self.initialized = False
        self.spect_fps = spect_fps
        self.data_dir = data_dir
        # set up the paths
        annotation_dir = data_dir / 'beat_annotations'
        spect_dir = data_dir / 'preprocessed' / 'spectrograms'
        # load dataframe with all pieces information
        metadata_file = spect_dir / 'spectrograms_metadata.csv'
        self.metadata_df = pd.read_csv(metadata_file)
        # only keep datasets that are both in DATASET_INFO and in the metadata
        usable_datasets = handle_datasets_mismatch(self.metadata_df)
        self.usable_datasets = usable_datasets
        self.metadata_df = self.metadata_df[self.metadata_df.dataset.isin(usable_datasets)].reset_index(drop=True)
        # find and load train/val splits
        if fold is not None:
            # use cross-validation splits
            self.trainval_splits = {}
            for dataset in usable_datasets:
                if dataset == test_dataset:
                    continue
                print(annotation_dir / dataset)
                cv = pd.read_csv(annotation_dir / dataset / f"{dataset}_8-fold.folds", header=None, names=["piece", "fold"], sep="\t")
                cv["split"] = cv["fold"].apply(lambda fold_idx: 'val' if fold_idx == fold else 'train')
                self.trainval_splits[dataset] = cv
            print("Cross-validation split loaded for datasets", self.trainval_splits.keys(), "fold", fold)
        else:
            # use single splits
            self.trainval_splits = {}
            for dataset in usable_datasets:
                if dataset == test_dataset:
                    continue
                self.trainval_splits[dataset] = pd.read_csv(annotation_dir / dataset / 'split.csv')
            print("Manual splits loaded:", self.trainval_splits.keys())
        # remember remaining parameters
        self.batch_size = batch_size
        self.train_length = train_length
        self.test_dataset = test_dataset
        self.hung_data = hung_data
        self.no_val = no_val
        self.num_workers = num_workers
        self.augmentations = augmentations
        self.length_based_oversampling_factor = length_based_oversampling_factor
        # check if augmentations.keys() contains only 'mask', 'pitch' and 'time'
        if not set(augmentations.keys()).issubset({'mask', 'pitch', 'tempo'}):
            raise ValueError(f"Unsupported augmentations: {augmentations.keys()}")


    def setup(self, stage=None):
        if self.initialized:
            return
        # split the dataset in train, validation, and test
        test_idx = self.metadata_df.index[self.metadata_df["dataset"] == self.test_dataset]
        train_idx = np.zeros(0, dtype=int)
        val_idx = np.zeros(0, dtype=int)
        for dataset, split_df in self.trainval_splits.items():
            df_subset = self.metadata_df[self.metadata_df["dataset"] == dataset].copy()
            if df_subset.shape[0] != split_df.shape[0]:
                raise ValueError(f"Dataset {dataset} has {df_subset.shape[0]} pieces, but split file has {split_df.shape[0]} pieces")
            df_subset["piece"] = df_subset["spect_folder"].apply(lambda p: Path(p).name)
            if set(df_subset.piece) != set(split_df.piece):
                raise ValueError(f"Piece names and split file do not match for dataset {dataset}")
            split_data_df = df_subset.reset_index().merge(split_df, on='piece').set_index('index')
            train_idx = np.concatenate([train_idx, split_data_df.index[split_data_df.split == "train"]])
            val_idx = np.concatenate([val_idx, split_data_df.index[split_data_df.split == "val"]])
        # # append train-only datasets
        # treat rwc as 3 different datasets with rwc_popular, rwc_jazz, rwc_classical, rwc_royalty-free. Necessary for literature-compatible dataset selection
        if "rwc" in self.metadata_df.dataset.unique():
            self.metadata_df.loc[self.metadata_df.dataset == "rwc", "dataset"] = self.metadata_df.loc[self.metadata_df.dataset == "rwc", "spect_folder"].apply(lambda p: "rwc_" + Path(p).name.split("_")[1])
        if self.hung_data:
            # use the same train dataset from MODELING BEATS AND DOWNBEATS WITH A TIME-FREQUENCY TRANSFORMER (validation set stay the same with all datasets)
            hung_train_datasets = ["hainsworth", "ballroom", "hjdb", "beatles", "rwc_popular", "simac", "smc", "harmonix"]
            train_idx = train_idx[self.metadata_df["dataset"][train_idx].isin(hung_train_datasets)]
        if self.no_val:
            print("No validation set. Training on all data.")
            # train on all available data (escluding test). Validation metrics are still computed for code compatibility, but do not convey any useful information.
            if self.hung_data:
                # exclude the no hung datasets from the validation set before merging train and val
                val_idx = val_idx[self.metadata_df["dataset"][val_idx].isin(hung_train_datasets)]
            train_idx = np.concatenate([train_idx, val_idx.copy()])
        if self.length_based_oversampling_factor and self.train_length is not None:
            # oversample the training set according to the audio_length information, so that long pieces are more likely to be sampled
            old_len = len(train_idx)
            piece_oversampling_factor = np.round(self.length_based_oversampling_factor * self.metadata_df["spect_len_ts0"][train_idx].values / (self.train_length)).astype(int)
            piece_oversampling_factor = np.clip(piece_oversampling_factor, 1, None)
            train_idx = np.repeat(train_idx, piece_oversampling_factor)
            print(f"Training set oversampled from {old_len} to {len(train_idx)} excerpts.")
        # print which datasets were used
        trainsets = set(self.metadata_df["dataset"][train_idx].unique())
        print("Datasets in train set:", sorted(trainsets))
        valsets = set(self.metadata_df["dataset"][val_idx].unique())
        print("Datasets in val set:", sorted(valsets))
        testsets = set(self.metadata_df["dataset"][test_idx].unique())
        print("Datasets in test set:", sorted(testsets))
        # print which datasets were not used
        usedsets = trainsets | valsets | testsets
        csvsets = set(self.metadata_df["dataset"].unique())
        infosets = set(DATASET_INFO.keys())
        if (csvsets - usedsets):
            print("Datasets in spectrogram CSV, but not used:", sorted(csvsets - usedsets))
        if (infosets - usedsets):
            print("Datasets in DATASET_INFO, but not used:", sorted(infosets - usedsets))
        # go back to rwc dataset to avoid further problems with paths
        self.metadata_df.loc[self.metadata_df.dataset.str.startswith("rwc_"), "dataset"] = "rwc"

        print("Creating datasets...")
        shared_kwargs = dict(data_folder=self.data_dir,
                            spect_fps=self.spect_fps,
                            train_length=self.train_length)
        self.train_dataset = BeatTrackingDataset(self.metadata_df.iloc[train_idx].copy(),
                                                 deterministic=False,
                                                 augmentations=self.augmentations,
                                                 **shared_kwargs)
        self.val_dataset = BeatTrackingDataset(self.metadata_df.iloc[val_idx].copy(),
                                                deterministic=True,
                                                augmentations={},
                                                **shared_kwargs)
        self.test_dataset = BeatTrackingDataset(self.metadata_df.iloc[test_idx].copy(),
                                                deterministic=True,
                                                augmentations={},
                                                **shared_kwargs)
        print(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}, Test size: {len(self.test_dataset)}")
        self.initialized = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        # Warning: for performances, this only runs on the middle excerpt of the long pieces
        # The paper results are computed after training in the predict script
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # Warning: this only runs on the middle 30s excerpt of the long pieces
        # Consider updating if not testing on GTZAN dataset
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def get_train_positive_weights(self, widen_target_mask=3):
        """
        Computes the relation of negative targets to positive targets.
        `widen_target_mask` reduces the number of negative targets by the given
        factor times the number of positive targets (for ignoring a number of
        frames around each positive label). 
        For example a `widen_target_mask` of 3 will ignore 7 frames, 3 for each side plus the central.
        """
        # find the positive weight for the loss as a ratio between (down)beat and non-(down)beat annotation
        dataset = self.train_dataset
        all_frames = sum(i["spect_lengths"][0] for i in dataset.items)
        all_frames_db = sum(item["spect_lengths"][0] for item in dataset.items if item["downbeat_mask"] ) # consider only datasets which have downbeat information
        beat_frames = sum(len(i["beat_time"]) for i in dataset.items)
        downbeat_frames = sum(1 for item in dataset.items if item["downbeat_mask"] for b in item["beat_value"] if b==1)

        return {"beat" : int(np.round((all_frames - beat_frames * (widen_target_mask*2 +1)) / beat_frames)),
                "downbeat" : int(np.round((all_frames_db - downbeat_frames * (widen_target_mask*2 +1)) / downbeat_frames)),
                }


def prepare_annotations(item, start_frame, end_frame, fps):
    truth_bdb_time = item["beat_time"]
    truth_bdb_value = item["beat_value"]
    # convert beat time from seconds to frame
    truth_bdb_frame = (truth_bdb_time * fps).round().astype(int)
    # form annotations excerpt
    # filter out the annotations that are earlier than the start and shift left
    truth_bdb_frame -= start_frame 
    idx = np.searchsorted(truth_bdb_frame, 0)
    truth_bdb_frame = truth_bdb_frame[idx:]
    truth_bdb_value = truth_bdb_value[idx:]
    # filter out the annotations that are later than the end
    idx = np.searchsorted(truth_bdb_frame, end_frame - start_frame)
    truth_bdb_frame = truth_bdb_frame[:idx]
    truth_bdb_value = truth_bdb_value[:idx]
    # create beat and downbeat separated annotations
    truth_beat = truth_bdb_frame
    truth_downbeat = truth_bdb_frame[truth_bdb_value == 1]
    # transform beat downbeat to frame-wise annotations
    framewise_truth_beat = index_to_framewise(truth_beat, end_frame - start_frame)
    framewise_truth_downbeat = index_to_framewise(truth_downbeat, end_frame - start_frame)
    # create orig beat, downbeat annotations for unquantized evaluation
    truth_orig_beat = item["beat_time"]
    truth_orig_downbeat = truth_bdb_time[item["beat_value"] == 1] # (use the full beat_value)
    # filter out the annotations that are outside the excerpt, and shift them left to the excerpt time
    truth_orig_beat = truth_orig_beat[(truth_orig_beat >= start_frame/fps) & (truth_orig_beat < end_frame/fps)] - (start_frame/fps)
    truth_orig_downbeat = truth_orig_downbeat[(truth_orig_downbeat >= start_frame/fps) & (truth_orig_downbeat < end_frame/fps)] - (start_frame/fps)
    # convert to strings (trick to collate sequences of different lengths)
    truth_orig_beat = truth_orig_beat.tobytes()
    truth_orig_downbeat = truth_orig_downbeat.tobytes()
    return framewise_truth_beat, framewise_truth_downbeat, truth_orig_beat, truth_orig_downbeat


def handle_datasets_mismatch(metadata_df):
    # find what datasets of DATASET_INFO are not in the dataframe
    missing_datasets = set(DATASET_INFO.keys()) - set(metadata_df.dataset)
    if missing_datasets:
        print("Warning: Datasets in DATASET_INFO, but not in the metadata df:", missing_datasets)
        print("These datasets won't be used.")
    # find what datasets of the dataframe are not in DATASET_INFO
    missing_datasets = set(metadata_df.dataset) - set(DATASET_INFO.keys())
    if missing_datasets:
        print("Warning: Datasets in the metadata df, but not in DATASET_INFO:", missing_datasets)
        print("These datasets won't be used.")
    # return the datasets that are in both
    return set(metadata_df.dataset) & set(DATASET_INFO.keys())

def split_piece(spect : torch.Tensor, chunk_size : int, border_size : int=6, avoid_short_end : bool=True):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size` and return the chunks and starting positions.
    Consecutive chunks overlap by `border_size`, which is assumed to be discarded in the predictions, since the model
     is not really trained on this part due to the max-pool in the loss.
    If `avoid_short_end` is true, the last chunk start is shifted left to ends at the end of the piece, therefore the last chunk can potentially overlap with previous chunks more than border_size, otherwise it will be a shorter segment.
    If the piece is shorter than `chunk_size`, avoid_short_end is ignored and the piece is returned as a single shorter chunk.

    Args:
        spect (torch.Tensor): The input spectrogram tensor of shape (time x bins).
        chunk_size (int): The size of the chunks to produce.
        border_size (int, optional): The size of the border to overlap between chunks. Defaults to 6.
        avoid_short_end (bool, optional): If True, the last chunk is shifted left to end at the end of the piece. Defaults to True.
    """
    # generate the start and end indices 
    starts = np.arange(-border_size, len(spect)- 2*border_size, chunk_size - 2 * border_size)
    if avoid_short_end and len(spect) > chunk_size - border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - 2 *border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [spect[max(start,0):min(start+chunk_size,len(spect))] for start in starts]
    # pad the first and last chunk in the time dimension to account for the border
    chunks[0] = F.pad(chunks[0], (0, 0, border_size, 0), "constant", 0)
    chunks[-1] = F.pad(chunks[-1], (0, 0, 0, border_size), "constant", 0)
    return chunks, starts
