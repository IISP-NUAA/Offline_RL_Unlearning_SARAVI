import minari
import numpy as np
from typing import List, Union, Iterable, Iterator
from minari.dataset.episode_data import EpisodeData
from minari.dataset.minari_storage import MinariStorage
from minari.dataset.minari_dataset import MinariDataset


class MergedMinariDataset:
    """Merge multiple Minari datasets and allow unified episode sampling."""

    def __init__(self, datasets: List[Union[MinariDataset, str]]):
        """
        Args:
            datasets (list[MinariDataset | str]): list of MinariDataset instances or their data paths.
        """
        self._datasets: List[MinariDataset] = []
        self._generator = np.random.default_rng()

        for ds in datasets:
            if isinstance(ds, MinariDataset):
                self._datasets.append(ds)
            else:
                self._datasets.append(MinariDataset(ds))

        self._episode_offsets = []
        total = 0
        for ds in self._datasets:
            self._episode_offsets.append((total, total + len(ds)))
            total += len(ds)
        self._total_episodes = total

    def set_seed(self, seed: int):
        """Set random seed for unified sampling."""
        self._generator = np.random.default_rng(seed)
        for ds in self._datasets:
            ds.set_seed(seed)

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def total_steps(self) -> int:
        return sum(ds.total_steps for ds in self._datasets)

    @property
    def dataset_ids(self) -> List[str]:
        return [ds.id for ds in self._datasets]

    def _locate_dataset(self, global_idx: int):
        """Return (dataset, local_idx) corresponding to global index."""
        for ds, (start, end) in zip(self._datasets, self._episode_offsets):
            if start <= global_idx < end:
                return ds, global_idx - start
        raise IndexError("Episode index out of range")

    def __getitem__(self, idx: int) -> EpisodeData:
        ds, local_idx = self._locate_dataset(idx)
        episode = ds[local_idx]
        episode.infos["source_dataset_id"] = ds.id
        episode_dict = episode.__dict__.copy()
        # print('episode_dict:', episode_dict)
        # episode_dict.infos["source_dataset_id"] = ds.id
        return EpisodeData(**episode_dict)

    def __len__(self):
        return self._total_episodes

    def iterate_episodes(self) -> Iterator[EpisodeData]:
        """Iterate over all episodes with dataset ID info."""
        for ds in self._datasets:
            for ep in ds:
                ep.infos["source_dataset_id"]= ds.id
                ep_dict = ep.__dict__.copy()
                # ep_dict["source_dataset_id"] = ds.id
                yield EpisodeData(**ep_dict)

    def sample_episodes(self, n_episodes: int) -> Iterable[EpisodeData]:
        """Sample episodes across multiple datasets."""
        indices = self._generator.choice(
            np.arange(self._total_episodes), size=n_episodes, replace=False
        )
        episodes = []
        for idx in indices:
            episodes.append(self[idx])
        return episodes

    def filter_episodes(self, condition):
        """Filter episodes across all datasets."""
        filtered_subsets = []
        for ds in self._datasets:
            filtered_ds = ds.filter_episodes(condition)
            if len(filtered_ds) > 0:
                filtered_subsets.append(filtered_ds)
        return MergedMinariDataset(filtered_subsets)


if __name__ == '__main__':
    ds1 = minari.load_dataset("D4RL/antmaze/large-diverse-v1")
    ds2 = minari.load_dataset("D4RL/antmaze/umaze-v1")
    ds3 = minari.load_dataset("D4RL/antmaze/umaze-diverse-v1")
    merged = MergedMinariDataset([ds1, ds2,ds3])
    merged.set_seed(42)
    samples = merged.sample_episodes(5)
    print(merged.total_steps)
    for ep in samples:
        print(ep.infos['source_dataset_id'], len(ep.rewards))
