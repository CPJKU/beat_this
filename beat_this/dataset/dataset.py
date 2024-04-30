from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import concurrent.futures
from jbt.utils import PAD_TOKEN, index_to_framewise
from beat_this.dataset.augment import precomputed_augmentation_filenames, select_augmentation
from beat_this.utils import load_spect

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

    Attributes:
        spect_basepath (Path): The base path where the preprocessed spectrograms are stored.
        annotation_basepath (Path): The base path where the beat annotations are stored.
        fps (int): The frames per second of the spectrograms.
        train_length (int): The length of the training sequences in frames.
        deterministic (bool): If True, the dataset always returns the same sequence for a given index.
        augmentations (dict): A dictionary of data augmentations to apply.
        items (list): A list of loaded dataset items.

    Args:
        metadata_df (pd.DataFrame): A DataFrame containing metadata about the dataset items.
            Each row should represent one item and contain at least a 'spect_folder' column.
        data_folder (Path or str): The base folder where the data is stored.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        train_length (int, optional): The length of the training sequences in frames. Defaults to 1500.
        deterministic (bool, optional): If True, the dataset always returns the same sequence for a given index.
            Defaults to False.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Defaults to an empty dictionary.
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
            beat_value = np.ones_like(beat_time, dtype=np.int32) * PAD_TOKEN

        # stop if the annotations that are supposed to be there are not there
        if DATASET_INFO[df_row["dataset"]]["downbeat"]:
            if beat_annotation.ndim != 2 :
                print(f"Skipping {df_row['beat_path']} because it has {beat_annotation.ndim} columns but downbeat is supposed to be there.")
                return

        # create a loss mask depending on the annotations that are supposed to be there
        loss_mask = [True, DATASET_INFO[df_row["dataset"]]["downbeat"]]
        # select all values in columns that start with spect_len, e.g. spect_len_ts-20
        spect_lengths = {int(key.replace("spect_len_ts","")): int(value) for key, value in df_row.items() if key.startswith("spect_len")}
        return {'spect_folder': str(spect_folder),
                'beat_time': beat_time,
                'beat_value': beat_value,
                'loss_mask': loss_mask,
                'dataset': df_row['dataset'],
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
            item = select_augmentation(item, self.augmentations)
            # define the excerpt to use
            original_length = item["spect_length"]
            longer = original_length - self.train_length
            if longer > 0: # if the piece is longer than the train length
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
            spect_path = Path(item["spect_folder"]) / item["spect_path"]
            spect = load_spect(spect_path, start=start_frame, stop=end_frame)

            # augment the spectrogram with mask augmentation if needed
            if "mask" in self.augmentations.keys():
                # TODO: to be implemented
                pass

            # prepare annotations
            framewise_truth_beat, framewise_truth_downbeat, truth_orig_beat, truth_orig_downbeat = prepare_annotations(item, start_frame, end_frame, self.fps)
            
            # prepare dictionary to return
            item = {"spect": spect,
                    "spect_path": str(spect_path),
                    "start_frame": start_frame,
                    "truth_beat": framewise_truth_beat,
                    "truth_downbeat": framewise_truth_downbeat,
                    "loss_mask": torch.as_tensor(item["loss_mask"]),
                    "padding_mask": np.ones(self.train_length, dtype=bool),
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
        train_length (int, optional): The length of the training sequences in frames. Defaults to 1500.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 20.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Defaults to {"pitch": {"min": -5, "max": 6}, "time": {"min": -20, "max": 20, "stride": 4}}.
        test_dataset (str, optional): The name of the dataset to use for testing. Defaults to "gtzan".
        train_datasets (list, optional): A list of dataset names to use for training. If None, all datasets except the test_dataset are used. Defaults to None.
        val_datasets (list, optional): A list of dataset names to use for validation. If None, the test_dataset is used. Defaults to None.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        length_based_oversampling_factor (int, optional): The factor by which to oversample based on sequence length. Defaults to 0.
        test_mode (bool, optional): If True, the DataModule is in test mode and only loads the test dataset. Defaults to False.
        fold (int, optional): The fold number for cross-validation. If None, the single split is used. Defaults to None.
    """
    def __init__(self, data_dir, batch_size=8,
                 train_length=1500, num_workers=20, augmentations={"pitch": {"min": -5, "max": 6}, "time": {"min": -20, "max": 20, "stride": 4}},
                 test_dataset="gtzan", train_datasets=None, val_datasets=None, spect_fps=50,
                 lenght_based_oversampling_factor=0,
                 test_mode=False, fold=None):
        super().__init__()
        self.save_hyperparameters()
        self.initialized = False
        self.spect_fps = spect_fps
        self.data_dir = data_dir
        # set up the paths
        annotation_dir = data_dir / 'annotations'
        spect_dir = data_dir / 'preprocessed' / 'spectrograms'
        # load dataframe with all pieces information
        metadata_file = spect_dir / 'spectrograms_metadata.csv'
        self.metadata_df = pd.read_csv(metadata_file)
        # only keep datasets that are both in DATASET_INFO and in the metadata
        usable_datasets = handle_datasets_mismatch(self.metadata_df)
        self.metadata_df = self.metadata_df[self.metadata_df.dataset.isin(usable_datasets)].reset_index(drop=True)
        # find and load manual train/val splits
        if fold is not None:
            self.manual_splits = {}
            for dataset in usable_datasets:
                if dataset == test_dataset:
                    continue
                print(annotation_dir / dataset)
                cv = pd.read_csv(next((annotation_dir / dataset).glob('*.folds')),
                                 sep='\t', names=['piece', 'fold'])
                cv["split"] = cv["fold"].apply(lambda fold_idx: 'val' if fold_idx == fold else 'train')
                self.manual_splits[dataset] = cv
        else:
            self.manual_splits = {dataset: pd.read_csv(annotation_dir / dataset / 'split.csv')
                                for dataset in usable_datasets
                                if (annotation_dir / dataset / 'split.csv').exists()}
        # remember remaining parameters
        self.batch_size = batch_size
        self.train_length = train_length
        self.test_dataset = test_dataset
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.num_workers = num_workers
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.lenght_based_oversampling_factor = lenght_based_oversampling_factor
        # check if augmentations.keys() contains only 'mask', 'pitch' and 'time'
        if not set(augmentations.keys()).issubset({'mask', 'pitch', 'time'}):
            raise ValueError(f"Unsupported augmentations: {augmentations.keys()}")


    def setup(self, stage=None):
        if self.initialized:
            return
        # split the dataset in train, validation, and test
        # start with test set, since this is hardcoded
        test_idx = self.metadata_df.index[self.metadata_df["dataset"] == self.test_dataset]
        # create the train/val split
        # figure out those datasets that have a manual split
        manual_idx = self.metadata_df.index[self.metadata_df["dataset"].isin(set(self.manual_splits.keys()))]
        # if not split manually, datasets without downbeats are only used for training
        train_only_datasets = set(name for name, info in DATASET_INFO.items()
                                  if not name in self.manual_splits and not info["downbeat"])
        train_only_idx = self.metadata_df.index[self.metadata_df["dataset"].isin(train_only_datasets)]
        # now create a new dataframe and split randomly, stratified by datasets
        trainval_df = (self.metadata_df
                       .drop(index=test_idx, inplace=False)
                       .drop(index=manual_idx, inplace=False)
                       .drop(index=train_only_idx, inplace=False))
        if len(trainval_df):
            train_idx, val_idx = train_test_split(trainval_df.index, test_size=0.15, stratify=self.metadata_df["dataset"][trainval_df.index].tolist(), random_state=0)
        else:
            train_idx = np.zeros(0, dtype=int)
            val_idx = np.zeros(0, dtype=int)
        # append manual splits
        for dataset, split_df in self.manual_splits.items():
            df_subset = self.metadata_df[self.metadata_df["dataset"] == dataset].copy()
            if df_subset.shape[0] != split_df.shape[0]:
                raise ValueError(f"Dataset {dataset} has {df_subset.shape[0]} pieces, but split file has {split_df.shape[0]} pieces")
            df_subset["piece"] = df_subset["processed_path"].apply(lambda p: Path(p).name)
            if set(df_subset.piece) != set(split_df.piece):
                raise ValueError(f"Piece names and split file do not match for dataset {dataset}")
            split_data_df = df_subset.reset_index().merge(split_df, on='piece').set_index('index')
            train_idx = np.concatenate([train_idx, split_data_df.index[split_data_df.split == "train"]])
            val_idx = np.concatenate([val_idx, split_data_df.index[split_data_df.split == "val"]])
        # append train-only datasets
        train_idx = np.concatenate([train_idx, train_only_idx])
        # treat rwc as 3 different datasets with rwc_popular, rwc_jazz, rwc_classical, rwc_royalty-free. Necessary for literature-compatible dataset selection
        if "rwc" in self.metadata_df.dataset.unique():
            self.metadata_df.loc[self.metadata_df.dataset == "rwc", "dataset"] = self.metadata_df.loc[self.metadata_df.dataset == "rwc", "processed_path"].apply(lambda p: "rwc_" + Path(p).name.split("_")[1])
        # limit the training dataset if self.train_datasets is not None
        if self.train_datasets is not None:
            if self.train_datasets == ["hung"]:
                # use the same train dataset from MODELING BEATS AND DOWNBEATS WITH A TIME-FREQUENCY TRANSFORMER
                self.train_datasets = ["hainsworth", "ballroom", "hjdb", "beatles", "rwc_popular", "simac", "smc", "harmonix"]
            train_idx = train_idx[self.metadata_df["dataset"][train_idx].isin(self.train_datasets)]
        # limit the validation dataset if self.val_datasets is not None
        if self.val_datasets is not None:
            if self.val_datasets == [""]: # don't validate, use all pieces for training. Practically we still validate on a subset of the training pieces to avoid exceptions
                if self.train_datasets is not None:
                    # exclude the no_training datasets from the validation set before merging the two
                    val_idx = val_idx[self.metadata_df["dataset"][val_idx].isin(self.train_datasets)]
                train_idx = np.concatenate([train_idx, val_idx])
            else:
                val_idx = val_idx[self.metadata_df["dataset"][val_idx].isin(self.val_datasets)]
        if self.lenght_based_oversampling_factor:
            # oversample the training set according to the audio_length information, so that long pieces are more likely to be sampled
            old_len = len(train_idx)
            piece_oversampling_factor = np.round(self.lenght_based_oversampling_factor * self.metadata_df["audio_length"][train_idx].values / (512*self.maximum_train_length)).astype(int)
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
            print("Datasets in CSV, but not used:", sorted(csvsets - usedsets))
        if (infosets - usedsets):
            print("Datasets in DATASET_INFO, but not used:", sorted(infosets - usedsets))
        # go back to rwc dataset to avoid further problems with paths
        self.metadata_df.loc[self.metadata_df.dataset.str.startswith("rwc_"), "dataset"] = "rwc"

        if self.test_mode:
            max_count = 50
        else:
            max_count = None
        print("Creating datasets...")
        shared_kwargs = dict(data_folder=self.data_dir,
                            spect_fps=self.spect_fps,
                            train_length=self.train_length)
        self.train_dataset = BeatTrackingDataset(self.metadata_df.iloc[train_idx[:max_count]].copy(),
                                                 deterministic=False,
                                                 augmentations=self.augmentations,
                                                 **shared_kwargs)
        self.val_dataset = BeatTrackingDataset(self.metadata_df.iloc[val_idx[:max_count]].copy(),
                                                deterministic=True,
                                                augmentations={},
                                                **shared_kwargs)
        self.test_dataset = BeatTrackingDataset(self.metadata_df.iloc[test_idx[:max_count]].copy(),
                                                deterministic=True,
                                                augmentations={},
                                                **shared_kwargs)
        print(f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}, Test size: {len(self.test_dataset)}")
        self.initialized = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        # Warning: for performances, this only runs on the middle excerpt of the long pieces
        # The paper results are computed after training in the predict script
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # Warning: this only runs on the middle 30s excerpt of the long pieces
        # Consider updating if not testing on GTZAN dataset
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return self.test_dataloader()
    
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
        all_frames_db = sum(item["spect_lengths"][0] for item in dataset.items if item["loss_mask"][1] ) # consider only datasets which have downbeat information
        any_beat_frames = sum(len(i["beat_time"]) for i in dataset.items)
        downbeat_frames = sum(1 for item in dataset.items if item["loss_mask"][1] for b in item["beat_value"] if b==1)

        return {"any_beat" : int(np.round((all_frames - any_beat_frames * (widen_target_mask*2 +1)) / any_beat_frames)),
                "downbeat" : int(np.round((all_frames_db - downbeat_frames * (widen_target_mask*2 +1)) / downbeat_frames)),
                }


def prepare_annotations(item, start_frame, end_frame, fps):
    truth_bdb_time = item["beat_time"]
    # convert beat time from seconds to frame
    truth_bdb_frame = (truth_bdb_time * fps).round().astype(int)
    truth_bdb_value = item["beat_value"]
    # form annotations excerpt
    # filter out the annotations that are earlier than the start and shift left
    truth_bdb_frame -= start_frame 
    idx = np.searchsorted(truth_bdb_frame, 0)
    truth_bdb_frame = truth_bdb_frame[idx:]
    truth_bdb_frame = truth_bdb_frame[idx:]
    # filter out the annotations that are later than the end
    idx = np.searchsorted(truth_bdb_frame, end_frame - start_frame)
    truth_bdb_frame = truth_bdb_frame[:idx]
    truth_bdb_value = truth_bdb_value[:idx]
    # create beat and downbeat separated annotations
    truth_beat = truth_bdb_frame
    truth_downbeat = truth_bdb_frame[truth_bdb_value == 1]
    # transform beat downbeat (and variations) to frame-wise annotations
    framewise_truth_beat = index_to_framewise(truth_beat, end_frame - start_frame)
    framewise_truth_downbeat = index_to_framewise(truth_downbeat, end_frame - start_frame)
    # create orig beat, downbeat annotations for unquantized evaluation
    truth_orig_beat = truth_bdb_time
    truth_orig_downbeat = truth_bdb_time[item["beat_value"] == 1]
    # filter out the annotations that are outside the excerpt, and shift them left to the excerpt time
    truth_orig_beat = truth_orig_beat[(truth_orig_beat >= start_frame/fps) & (truth_orig_beat < end_frame/fps)] - (start_frame/fps)
    truth_orig_downbeat = truth_orig_downbeat[(truth_orig_downbeat >= start_frame/fps) & (truth_orig_downbeat < end_frame/fps)] - (start_frame/fps)
    # convert to strings (trick to collate sequences of different lengths)
    truth_orig_beat = truth_orig_beat.tobytes()
    truth_orig_downbeat = truth_orig_downbeat.tobytes()
    return framewise_truth_beat, framewise_truth_downbeat, truth_orig_beat, truth_orig_downbeat


def split_piece(audio, chunk_size, border_size=0, overlap=0, avoid_short_end=True):
    """
    Split an audio tensor into chunks of `chunk_size` and return the chunks and starting positions.
    Consecutive chunks overlap by `border_size`, which is assumed to be what the model drops in its forward pass.
    Additionally, consecutive chunks may overlap by `overlap`; predictions will need to be averaged over that.
    If `avoid_short_end` is true-ish, the last chunk ends at the end of the piece, otherwise it will be padded.
    """
    chunks = []
    starts = []
    start = -border_size
    while True:
        end = start + chunk_size
        if (start > -border_size) and (end > len(audio) + border_size) and avoid_short_end:
            start = len(audio) + border_size - chunk_size
            end = start + chunk_size
        chunk = audio[max(start, 0):end]
        if start < 0:
            # pad with zeros to the left of the recording
            pad_before = -start
        else:
            pad_before = 0
        if end > len(audio):
            # pad with zeros to the right of the recording
            pad_after = end - len(audio)
        else:
            pad_after = 0
        if pad_before or pad_after:
            chunk = pad(chunk, dim=0, before=pad_before, after=pad_after)
        chunks.append(chunk)
        starts.append(start + border_size)
        if end >= len(audio) + border_size:
            # we covered everything
            break
        else:
            start += chunk_size - border_size - overlap
    return chunks, starts


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


# test the dataset and datamodule
if __name__ == "__main__":
    data_dir = Path("/share/hel/home/francesco/beat_this/data")
    datamodule = BeatDataModule(data_dir)
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(batch["spect"].shape)
        print(batch["truth_beat"].shape)
        print(batch["padding_mask"].shape)
        break
    print("Pos weights:")
    print(datamodule.get_train_positive_weights())
