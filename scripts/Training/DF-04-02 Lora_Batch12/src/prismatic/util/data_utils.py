# =============================================================================
# DF-04-02 LoRA-OneStepCE Data Loading and Collation Utilities
# =============================================================================
#
# EN: This file preserves the original OpenVLA language-model collator and
#     extends the action-prediction collator for one-step OpenVLA action-token training.
#
# EN: The action collator now:
#     1. Pads input_ids and labels.
#     2. Stacks primary images.
#     3. Optionally concatenates wrist images along the channel dimension.
#     4. Converts and stacks one-step continuous action targets as [B, 1, 7].
#     5. Optionally stacks LIBERO proprio vectors as [B, 8].
#     6. Preserves dataset names for logging and validation.
#     7. Makes per-sample NaN/Inf checks configurable for preflight.
# =============================================================================

"""
data_utils.py

EN: General utilities and classes for data loading and collation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# EN: HuggingFace / LLaMA-2 ignore index used to mask labels from loss.
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """
    EN: Apply a function to every leaf value in a nested dictionary.
    """
    return {
        key: tree_map(fn, value) if isinstance(value, dict) else fn(value)
        for key, value in tree.items()
    }


def tree_map_with_key(
    fn: Callable,
    tree: dict,
    keys: Sequence = (),
) -> dict:
    """
    EN: Apply a function to every leaf while also passing its nested key path.
    """
    return {
        key: (
            tree_map_with_key(fn, value, (*keys, key))
            if isinstance(value, dict)
            else fn((*keys, key), value)
        )
        for key, value in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    """
    EN: Original OpenVLA collator for language or multimodal language modeling.
    """

    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(
            self.default_image_resolution,
            dtype=self.pixel_values_dtype,
        )

    def __call__(
        self,
        instances: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not instances:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        pixel_values = [instance["pixel_values"] for instance in instances]

        # EN: Only right-side padding is supported during training.
        if self.padding_side != "right":
            raise ValueError(
                f"Invalid padding_side={self.padding_side!r}; expected 'right'."
            )

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        # EN: Truncate sequences to the model context length.
        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        # EN: Identify examples that actually contain images.
        multimodal_indices = torch.tensor(
            [
                index
                for index, value in enumerate(pixel_values)
                if value is not None
            ],
            dtype=torch.long,
        )

        if len(multimodal_indices) == 0:
            pixel_values = torch.stack(
                [
                    self.dummy_pixel_values
                    for _ in range(len(input_ids))
                ]
            )
        elif isinstance(
            pixel_value_example := pixel_values[multimodal_indices[0]],
            torch.Tensor,
        ):
            multimodal_index_set = set(multimodal_indices.tolist())
            pixel_values = torch.stack(
                [
                    (
                        pixel_values[index]
                        if index in multimodal_index_set
                        else self.dummy_pixel_values
                    )
                    for index in range(len(input_ids))
                ]
            )
        elif isinstance(pixel_value_example, dict):
            multimodal_index_set = set(multimodal_indices.tolist())
            pixel_values = {
                key: torch.stack(
                    [
                        (
                            pixel_values[index][key]
                            if index in multimodal_index_set
                            else self.dummy_pixel_values
                        )
                        for index in range(len(input_ids))
                    ]
                )
                for key in pixel_value_example
            }
        else:
            raise ValueError(
                "Unsupported pixel_values type: "
                f"{type(pixel_value_example).__name__}."
            )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "multimodal_indices": multimodal_indices,
        }


@dataclass
class PaddedCollatorForActionPrediction:
    """
    EN: Collator for one-step OpenVLA action-token training.
    """

    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    # EN: Whether the batch must contain wrist-camera images.
    use_wrist_image: bool = False

    # EN: Whether the batch must contain robot proprioceptive state.
    use_proprio: bool = False

    # EN: Enable expensive per-sample NaN/Inf checks during preflight.
    #     Set this to False for the formal 200k-step training run.
    validate_finite_values: bool = False

    # EN: Expected LIBERO action and proprio dimensions.
    expected_action_chunk_len: int = 1
    expected_action_dim: int = 7
    expected_proprio_dim: int = 8

    def __post_init__(self) -> None:
        if self.model_max_length <= 0:
            raise ValueError(
                "model_max_length must be positive."
            )
        if self.expected_action_chunk_len <= 0:
            raise ValueError(
                "expected_action_chunk_len must be positive."
            )
        if self.expected_action_dim <= 0:
            raise ValueError(
                "expected_action_dim must be positive."
            )
        if self.expected_proprio_dim <= 0:
            raise ValueError(
                "expected_proprio_dim must be positive."
            )

    @staticmethod
    def _validate_consistent_presence(
        instances: Sequence[Dict[str, Any]],
        key: str,
    ) -> bool:
        """
        EN: Require a field to be present in either every sample or no sample.
        """
        presence = [key in instance for instance in instances]
        if any(presence) and not all(presence):
            raise ValueError(
                f"Mixed batch: field {key!r} is present in only some samples."
            )
        return all(presence)

    def _stack_pixel_tensors(
        self,
        pixel_values: Sequence[torch.Tensor],
        field_name: str,
    ) -> torch.Tensor:
        """
        EN: Validate and stack image tensors into [B, C, H, W].
        """
        tensors = []
        reference_shape = None

        for sample_index, value in enumerate(pixel_values):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Sample {sample_index} {field_name} must be a torch.Tensor; "
                    f"got {type(value).__name__}."
                )

            if value.ndim != 3:
                raise ValueError(
                    f"Sample {sample_index} {field_name} must have shape "
                    f"[C, H, W]; got {tuple(value.shape)}."
                )

            if reference_shape is None:
                reference_shape = tuple(value.shape)
            elif tuple(value.shape) != reference_shape:
                raise ValueError(
                    f"Inconsistent {field_name} shapes: expected "
                    f"{reference_shape}, got {tuple(value.shape)} at sample "
                    f"{sample_index}."
                )

            if (
                self.validate_finite_values
                and not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Sample {sample_index} {field_name} contains NaN or Inf."
                )

            tensors.append(value)

        return torch.stack(tensors, dim=0)

    def __call__(
        self,
        instances: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        EN: Build one one-step OpenVLA action-token training batch.

        EN: Output shapes:
            input_ids:          [B, sequence_length]
            labels:             [B, sequence_length]
            attention_mask:     [B, sequence_length]
            pixel_values:       [B, total_channels, H, W]
            actions:            [B, 1, 7]
            proprio:            [B, 8]          when enabled
            dataset_names:      Python list     when provided


            """
        if not instances:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        if self.padding_side != "right":
            raise ValueError(
                f"Invalid padding_side={self.padding_side!r}; expected 'right'."
            )

        # EN: Required token and image fields.
        for sample_index, instance in enumerate(instances):
            for required_key in (
                "input_ids",
                "labels",
                "pixel_values",
                "actions",
            ):
                if required_key not in instance:
                    raise KeyError(
                        f"Sample {sample_index} is missing {required_key!r}."
                    )

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        for sample_index, (sample_input_ids, sample_labels) in enumerate(
            zip(input_ids, labels)
        ):
            if not isinstance(sample_input_ids, torch.Tensor):
                raise TypeError(
                    f"Sample {sample_index} input_ids must be torch.Tensor."
                )
            if not isinstance(sample_labels, torch.Tensor):
                raise TypeError(
                    f"Sample {sample_index} labels must be torch.Tensor."
                )
            if sample_input_ids.ndim != 1 or sample_labels.ndim != 1:
                raise ValueError(
                    f"Sample {sample_index} input_ids and labels must be 1-D."
                )
            if sample_input_ids.shape != sample_labels.shape:
                raise ValueError(
                    f"Sample {sample_index} input_ids and labels shapes differ: "
                    f"{tuple(sample_input_ids.shape)} vs "
                    f"{tuple(sample_labels.shape)}."
                )

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        # EN: Stack primary images.
        primary_pixel_values = self._stack_pixel_tensors(
            [instance["pixel_values"] for instance in instances],
            field_name="pixel_values",
        )

        wrist_present = self._validate_consistent_presence(
            instances,
            "pixel_values_wrist",
        )

        if self.use_wrist_image and not wrist_present:
            raise KeyError(
                "use_wrist_image=True, but the batch has no pixel_values_wrist."
            )

        if not self.use_wrist_image and wrist_present:
            raise ValueError(
                "The batch contains pixel_values_wrist, but "
                "use_wrist_image=False."
            )

        if wrist_present:
            wrist_pixel_values = self._stack_pixel_tensors(
                [
                    instance["pixel_values_wrist"]
                    for instance in instances
                ],
                field_name="pixel_values_wrist",
            )

            if (
                primary_pixel_values.shape[0]
                != wrist_pixel_values.shape[0]
            ):
                raise RuntimeError(
                    "Primary and wrist image batch sizes differ."
                )

            if (
                primary_pixel_values.shape[2:]
                != wrist_pixel_values.shape[2:]
            ):
                raise ValueError(
                    "Primary and wrist image spatial sizes differ: "
                    f"{tuple(primary_pixel_values.shape[2:])} vs "
                    f"{tuple(wrist_pixel_values.shape[2:])}."
                )

            # EN: Merge all images into one channel-stacked tensor.
            #     Example for a fused backbone:
            #     primary [B, 6, H, W] + wrist [B, 6, H, W]
            #     -> pixel_values [B, 12, H, W].
            combined_pixel_values = torch.cat(
                (primary_pixel_values, wrist_pixel_values),
                dim=1,
            )
        else:
            combined_pixel_values = primary_pixel_values

        # EN: Convert and stack one-step continuous action targets.
        expected_action_shape = (
            self.expected_action_chunk_len,
            self.expected_action_dim,
        )
        action_tensors = []

        for sample_index, instance in enumerate(instances):
            actions = instance["actions"]

            if isinstance(actions, torch.Tensor):
                actions_tensor = actions.to(dtype=torch.float32)
            else:
                try:
                    actions_array = np.asarray(
                        actions,
                        dtype=np.float32,
                    )
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"Sample {sample_index} actions cannot be converted "
                        "to float32."
                    ) from error

                # EN: copy() avoids read-only or negative-stride NumPy buffers.
                actions_tensor = torch.from_numpy(actions_array.copy())

            if tuple(actions_tensor.shape) != expected_action_shape:
                raise ValueError(
                    f"Sample {sample_index} actions shape is "
                    f"{tuple(actions_tensor.shape)}, expected "
                    f"{expected_action_shape}."
                )

            if (
                self.validate_finite_values
                and not torch.isfinite(actions_tensor).all()
            ):
                raise ValueError(
                    f"Sample {sample_index} actions contains NaN or Inf."
                )

            action_tensors.append(actions_tensor)

        output: Dict[str, Any] = {
            "pixel_values": combined_pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": torch.stack(action_tensors, dim=0),
        }

        # EN: Preserve dataset names only when every sample provides one.
        dataset_name_present = self._validate_consistent_presence(
            instances,
            "dataset_name",
        )
        dataset_names_present = self._validate_consistent_presence(
            instances,
            "dataset_names",
        )

        if dataset_name_present and dataset_names_present:
            raise ValueError(
                "Samples contain both dataset_name and dataset_names."
            )

        if dataset_name_present:
            output["dataset_names"] = [
                instance["dataset_name"]
                for instance in instances
            ]
        elif dataset_names_present:
            output["dataset_names"] = [
                instance["dataset_names"]
                for instance in instances
            ]

        # EN: Convert and stack proprio vectors when enabled.
        proprio_present = self._validate_consistent_presence(
            instances,
            "proprio",
        )

        if self.use_proprio and not proprio_present:
            raise KeyError(
                "use_proprio=True, but the batch has no proprio."
            )

        if not self.use_proprio and proprio_present:
            raise ValueError(
                "The batch contains proprio, but use_proprio=False."
            )

        if proprio_present:
            proprio_tensors = []

            for sample_index, instance in enumerate(instances):
                proprio = instance["proprio"]

                if isinstance(proprio, torch.Tensor):
                    proprio_tensor = proprio.to(
                        dtype=torch.float32
                    ).reshape(-1)
                else:
                    try:
                        proprio_array = np.asarray(
                            proprio,
                            dtype=np.float32,
                        ).reshape(-1)
                    except (TypeError, ValueError) as error:
                        raise TypeError(
                            f"Sample {sample_index} proprio cannot be "
                            "converted to float32."
                        ) from error

                    proprio_tensor = torch.from_numpy(
                        proprio_array.copy()
                    )

                if proprio_tensor.numel() != self.expected_proprio_dim:
                    raise ValueError(
                        f"Sample {sample_index} proprio dimension is "
                        f"{proprio_tensor.numel()}, expected "
                        f"{self.expected_proprio_dim}."
                    )

                if (
                    self.validate_finite_values
                    and not torch.isfinite(proprio_tensor).all()
                ):
                    raise ValueError(
                        f"Sample {sample_index} proprio contains NaN or Inf."
                    )

                proprio_tensors.append(proprio_tensor)

            output["proprio"] = torch.stack(
                proprio_tensors,
                dim=0,
            )

        return output
