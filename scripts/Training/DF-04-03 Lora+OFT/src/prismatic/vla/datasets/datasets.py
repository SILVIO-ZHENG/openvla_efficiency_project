# =============================================================================
# DF-04-03 LoRA+OFT Dataset Pipeline
# =============================================================================
#
# EN: This file merges the official OpenVLA-OFT dataset pipeline with the
#     dissertation project's train-only normalization-statistics controls.
#
# EN: Main supported features:
#     1. Eight-step action chunks: 8 x 7 continuous actions.
#     2. Primary image plus optional wrist image(s).
#     3. Optional 8-D LIBERO proprioceptive state.
#     4. External train-only action/proprio normalization statistics.
#     5. Strict shape, configuration, and statistics validation.
#
# EN: This file prepares data only. The collator must preserve `actions`,
#     `pixel_values_wrist`, and `proprio`; the training script must then pass
#     them to the OFT action head and proprio projector.
# =============================================================================

"""
datasets.py

EN: Lightweight PyTorch dataset wrappers around the RLDS/TFDS pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import (
    OXE_NAMED_MIXTURES,
    get_oxe_dataset_kwargs_and_weights,
)


@dataclass
class RLDSBatchTransform:
    """
    EN: Converts one RLDS sample into the tensors and metadata required by
        OpenVLA-OFT training.
    """

    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        EN: Build an eight-step action-token sequence and return the associated
            continuous actions, image inputs, and optional proprio state.
        """
        observation = rlds_batch.get("observation")
        if not isinstance(observation, dict):
            raise TypeError(
                "RLDS batch observation must be a dictionary."
            )

        if "image_primary" not in observation:
            raise KeyError(
                "RLDS batch is missing observation.image_primary."
            )

        if "action" not in rlds_batch:
            raise KeyError(
                "RLDS batch is missing action. / RLDS batch 缺少 action。"
            )

        dataset_name = rlds_batch["dataset_name"]
        actions = np.asarray(rlds_batch["action"], dtype=np.float32)

        # EN: OFT requires one current action plus seven future actions.
        expected_action_shape = (NUM_ACTIONS_CHUNK, ACTION_DIM)
        if actions.shape != expected_action_shape:
            raise ValueError(
                "Expected action chunk shape "
                f"{expected_action_shape}, but got {actions.shape}."
            )

        current_action = actions[0]
        future_actions = actions[1:]

        primary_image = Image.fromarray(observation["image_primary"][0])
        language_instruction = (
            rlds_batch["task"]["language_instruction"].decode().lower()
        )

        # EN: Tokenize the current action and all future actions into one
        #     parallel-decoding action-token sequence.
        current_action_string = self.action_tokenizer(current_action)
        future_action_strings = self.action_tokenizer(future_actions)
        action_chunk_string = current_action_string + "".join(future_action_strings)
        action_chunk_token_count = len(action_chunk_string)

        expected_action_token_count = NUM_ACTIONS_CHUNK * ACTION_DIM
        if action_chunk_token_count != expected_action_token_count:
            raise ValueError(
                "Unexpected number of action tokens: "
                f"actual={action_chunk_token_count}, "
                f"expected={expected_action_token_count}."

            )

        # EN: Build the OpenVLA conversational prompt.
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {
                "from": "human",
                "value": (
                    "What action should the robot take to "
                    f"{language_instruction}?"
                ),
            },
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(),
            add_special_tokens=True,
        ).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        pixel_values = self.image_transform(primary_image)

        # EN: Compute language-model loss only on the 56 action tokens and,
        #     when enabled, the final stop token.
        labels[: -(action_chunk_token_count + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        output: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            # EN: Keep NumPy format for compatibility with the OFT collator.
            "actions": actions,
        }

        if self.use_wrist_image:
            wrist_pixel_values = []

            # EN: LIBERO exposes one wrist view; ALOHA may expose two wrist views.
            for key, value in observation.items():
                if "wrist" not in key or not key.startswith("image"):
                    continue

                wrist_image = Image.fromarray(value[0])
                transformed_wrist = self.image_transform(wrist_image)

                if not isinstance(transformed_wrist, torch.Tensor):
                    raise TypeError(
                        "Wrist image transform must return torch.Tensor; "
                        f"got {type(transformed_wrist).__name__}. "
                        f"实际为 {type(transformed_wrist).__name__}。"
                    )

                wrist_pixel_values.append(transformed_wrist)

            if not wrist_pixel_values:
                raise KeyError(
                    "use_wrist_image=True, but no wrist image was found in "
                    "the RLDS observation. / use_wrist_image=True，"
                )

            # EN: Concatenate multiple wrist views along the channel dimension.
            output["pixel_values_wrist"] = torch.cat(
                wrist_pixel_values,
                dim=0,
            )

        if self.use_proprio:
            if "proprio" not in observation:
                raise KeyError(
                    "use_proprio=True, but the RLDS batch does not contain "
                    "observation.proprio. / use_proprio=True，"
                )

            # EN: window_size=1 gives shape [1, PROPRIO_DIM]; keep the first row.
            proprio = np.asarray(
                observation["proprio"][0],
                dtype=np.float32,
            ).reshape(-1)

            if proprio.shape != (PROPRIO_DIM,):
                raise ValueError(
                    f"Expected a {PROPRIO_DIM}-D proprio vector, "
                    f"but got {proprio.shape}. "
                )

            if not np.all(np.isfinite(proprio)):
                raise ValueError(
                    "proprio contains NaN or Inf. / "
                )

            output["proprio"] = torch.from_numpy(proprio)

        return output


class RLDSDataset(IterableDataset):
    """
    EN: Wraps one RLDS dataset or an OXE mixture as a PyTorch IterableDataset.
    """

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        normalization_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.batch_transform = batch_transform
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.normalization_stats = normalization_stats

        # EN: Dataset loading options and sample-transform options must match.
        if self.batch_transform.use_wrist_image != self.use_wrist_image:
            raise ValueError(
                "RLDSBatchTransform.use_wrist_image and "
                "RLDSDataset.use_wrist_image must match."
            )

        if self.batch_transform.use_proprio != self.use_proprio:
            raise ValueError(
                "RLDSBatchTransform.use_proprio and "
                "RLDSDataset.use_proprio must match. "
            )

        if self.normalization_stats is not None and not isinstance(
            self.normalization_stats,
            dict,
        ):
            raise TypeError(
                "normalization_stats must be a dictionary or None. "
            )

        # EN: Resolve a named OXE mixture or treat data_mix as one dataset.
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        # EN: Load only the camera views requested by this experiment.
        if self.use_wrist_image:
            if "aloha" in self.data_mix.lower():
                load_camera_views = (
                    "primary",
                    "left_wrist",
                    "right_wrist",
                )
            else:
                load_camera_views = ("primary", "wrist")
        else:
            load_camera_views = ("primary",)

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=self.use_proprio,
            load_language=True,
            action_proprio_normalization_type=(
                ACTION_PROPRIO_NORMALIZATION_TYPE
            ),
        )

        # EN: Inject externally generated train-only statistics into every
        #     selected dataset before the RLDS pipeline is built.
        if self.normalization_stats is not None:
            missing_stats = [
                dataset_kwargs["name"]
                for dataset_kwargs in per_dataset_kwargs
                if dataset_kwargs["name"] not in self.normalization_stats
            ]
            if missing_stats:
                raise KeyError(
                    "Missing normalization statistics for datasets: "
                    + ", ".join(missing_stats)
                    + " / 缺少以下数据集的归一化统计量："
                    + ", ".join(missing_stats)
                )

            for dataset_kwargs in per_dataset_kwargs:
                dataset_name = dataset_kwargs["name"]
                dataset_statistics = self.normalization_stats[dataset_name]

                if not isinstance(dataset_statistics, dict):
                    raise TypeError(
                        f"Statistics for {dataset_name} must be a dictionary. / "
                        f"{dataset_name} 的统计量必须是字典。"
                    )

                if "action" not in dataset_statistics:
                    raise KeyError(
                        f"Statistics for {dataset_name} are missing action. / "
                        f"{dataset_name} 的统计量缺少 action。"
                    )

                if self.use_proprio and "proprio" not in dataset_statistics:
                    raise KeyError(
                        f"Statistics for {dataset_name} are missing proprio. / "
                        f"{dataset_name} 的统计量缺少 proprio。"
                    )

                dataset_kwargs["dataset_statistics"] = dataset_statistics

        rlds_config = {
            "traj_transform_kwargs": {
                # EN: One observation frame is used as input.
                "window_size": 1,
                # EN: Current action + seven future actions = eight actions.
                "future_action_window_size": NUM_ACTIONS_CHUNK - 1,
                "skip_unlabeled": True,
                "goal_relabeling_strategy": "uniform",
            },
            "frame_transform_kwargs": {
                "resize_size": resize_resolution,
                "num_parallel_calls": 16,
            },
            "dataset_kwargs_list": per_dataset_kwargs,
            "shuffle_buffer_size": shuffle_buffer_size,
            "sample_weights": weights,
            "balance_weights": True,
            "traj_transform_threads": len(mixture_spec),
            "traj_read_threads": len(mixture_spec),
            "train": train,
        }

        if image_aug:
            rlds_config["frame_transform_kwargs"].update(
                {
                    "image_augment_kwargs": {
                        "random_resized_crop": {
                            "scale": [0.9, 0.9],
                            "ratio": [1.0, 1.0],
                        },
                        "random_brightness": [0.2],
                        "random_contrast": [0.8, 1.2],
                        "random_saturation": [0.8, 1.2],
                        "random_hue": [0.05],
                        "augment_order": [
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    }
                }
            )

        self.dataset, self.dataset_length, self.dataset_statistics = (
            self.make_dataset(rlds_config)
        )

    def make_dataset(
        self,
        rlds_config: Dict[str, Any],
    ) -> Tuple[Any, int, Dict[str, Any]]:
        """
        EN: Materialize the configured interleaved RLDS pipeline.
        """
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        EN: Stream NumPy RLDS samples and convert them into training samples.
        """
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        """
        EN: IterableDataset intentionally does not support indexed access.
        """
        raise NotImplementedError(
            "IterableDataset does not implement map-style __getitem__. / "
        )


class EpisodicRLDSDataset(RLDSDataset):
    """
    EN: Returns complete episodes as lists of transformed steps.
    """

    def make_dataset(
        self,
        rlds_config: Dict[str, Any],
    ) -> Tuple[Any, int, Dict[str, Any]]:
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        if len(per_dataset_kwargs) != 1:
            raise ValueError(
                "EpisodicRLDSDataset supports only one dataset. / "
            )

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Iterator[Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            output = [
                self.batch_transform(
                    tree_map(lambda value: value[index], rlds_batch)
                )
                for index in range(rlds_batch["action"].shape[0])
            ]
            yield output


class DummyDataset(Dataset):
    """
    EN: Minimal synthetic dataset retained for local interface tests.
    """

    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # EN: Dummy q01/q99 values represent identity-like action scaling.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {
                    "q01": np.zeros((ACTION_DIM,), dtype=np.float32),
                    "q99": np.ones((ACTION_DIM,), dtype=np.float32),
                }
            }
        }

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = Image.fromarray(
            np.asarray(
                np.random.rand(224, 224, 3) * 255.0,
                dtype=np.uint8,
            )
        )
        action = np.asarray(
            np.random.rand(ACTION_DIM),
            dtype=np.float32,
        )
        instruction = "do something spectacular"

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {
                "from": "human",
                "value": (
                    "What action should the robot take to "
                    f"{instruction}?"
                ),
            },
            {
                "from": "gpt",
                "value": self.action_tokenizer(action),
            },
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(),
            add_special_tokens=True,
        ).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        pixel_values = self.image_transform(image)

        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }
