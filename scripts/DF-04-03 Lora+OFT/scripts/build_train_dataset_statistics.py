"""
build_train_dataset_statistics.py

Builds train-only normalization statistics for the four modified LIBERO RLDS
sub-datasets used by DF-04-03 LoRA+OFT.

The script reads the fixed seed-42 logical split JSON, selects only the
`train_episode_indices`, applies the same LIBERO action/proprio standardization
used by OpenVLA-OFT, and writes one provenance-rich JSON file:

    inputs/model_config/dataset_statistics_train_seed42.json

No validation or test episode is used when computing these statistics.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parent.parent

# EN: These paths stay inside the uploadable DF-04-03 folder.
# 中文：这些路径始终位于可整体上传的 DF-04-03 文件夹内部。
DEFAULT_SPLIT_JSON = BUNDLE_ROOT / "inputs" / "splits" / "libero_episode_split_seed42.json"
DEFAULT_OUTPUT_JSON = (
    BUNDLE_ROOT / "inputs" / "model_config" / "dataset_statistics_train_seed42.json"
)

# EN: The dataset itself remains external because it is too large to place in the bundle.
# 中文：数据集体积过大，因此仍放在 DF-04-03 文件夹外部。
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "OPENVLA_DATASET_ROOT",
        "/root/autodl-tmp/datasets/openvla_modified_libero_rlds",
    )
).expanduser()

# EN: These are the only four modified LIBERO sub-datasets expected for DF-04-03.
#     The statistics builder refuses missing, duplicate, or unexpected dataset names.
#     This prevents accidentally computing statistics on the wrong dataset mixture.
# 中文：这里固定了 DF-04-03 预期使用的四个 modified LIBERO 子数据集。
#     如果 split JSON 里缺少、多出、重复这些名字，脚本会直接报错。
#     这样可以防止误用错误的数据集组合来计算统计量。
EXPECTED_DATASETS: Tuple[str, ...] = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)
# EN: Expected total number of training episodes across the four sub-datasets.
#     Validation keeps 10 episodes per sub-dataset, test remains unchanged,
#     and all remaining episodes are assigned to training: train=1480.
# 中文：四个子数据集合起来的训练 episode 总数。
#     每个子数据集保留 10 个 validation episode，test 保持不变，
#     其余 episode 全部用于训练：train=1480。
EXPECTED_TOTAL_TRAIN_EPISODES = 1480
EXPECTED_SPLIT_SEED = 42
EXPECTED_SOURCE_SPLIT = "train"
# EN: LIBERO/OpenVLA action dimension.
#     The first six dimensions are continuous end-effector motion/control values.
#     The final dimension is the gripper command.
# 中文：LIBERO/OpenVLA 的动作维度。
#     前 6 维是连续的末端执行器运动/控制值。
#     最后一维是夹爪命令。
ACTION_DIM = 7
# EN: OFT proprioception input dimension.
#     It is constructed as 6 end-effector state values plus 2 gripper state values.
# 中文：OFT 使用的本体感知输入维度。
#     它由 6 维末端执行器状态 + 2 维夹爪状态拼接得到。
PROPRIO_DIM = 8

# EN: LIBERO's final action dimension is the gripper command and is not normalized.
# 中文：LIBERO 动作最后一维是夹爪命令，不参与连续动作归一化。
ACTION_NORMALIZATION_MASK: Tuple[bool, ...] = (
    True,
    True,
    True,
    True,
    True,
    True,
    False,
)

# EN: All eight OFT proprio dimensions are real state values.
# 中文：OFT 的 8 个 proprio 维度都是真实机器人状态值。
PROPRIO_NORMALIZATION_MASK: Tuple[bool, ...] = (True,) * PROPRIO_DIM


# EN: Custom exception type for this statistics-building pipeline.
#     Using a dedicated error class makes it clear that the failure is caused by
#     invalid split data, invalid dataset structure, or invalid generated statistics.
# 中文：这个脚本专用的异常类型。
#     使用专门的错误类，可以明确表示失败原因来自 split 数据、数据集结构、
#     或生成的统计量不合法。
class StatisticsBuildError(RuntimeError):
    """Raised when the split, dataset, or generated statistics are invalid."""


# EN: Parse command-line options for running the statistics builder.
#     This function does not touch the dataset and does not compute statistics.
#     It only defines how the user can override default paths and runtime behavior.
# 中文：解析运行统计量脚本时传入的命令行参数。
#     这个函数不会读取数据集，也不会计算统计量。
#     它只定义用户如何覆盖默认路径和运行行为。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute train-only action/proprio statistics for the four modified "
            "LIBERO RLDS datasets used by DF-04-03 LoRA+OFT."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root containing the four TFDS/RLDS sub-dataset directories. "
            f"Default: {DEFAULT_DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=DEFAULT_SPLIT_JSON,
        help=f"Fixed logical split JSON. Default: {DEFAULT_SPLIT_JSON}",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output statistics JSON. Default: {DEFAULT_OUTPUT_JSON}",
    )
    parser.add_argument(
        "--progress-interval-episodes",
        type=int,
        default=50,
        help="Print progress after this many selected episodes per sub-dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output JSON.",
    )
    return parser.parse_args()


# EN: Compute the SHA256 hash of a file by streaming it in 1 MB chunks.
#     This avoids loading a potentially large file fully into memory.
#     The hash is stored in metadata so the exact input/output file can be verified later.
# 中文：按 1MB 分块读取文件并计算 SHA256。
#     这样不会把较大的文件一次性全部读入内存。
#     哈希会写入 metadata，方便以后确认使用的是同一个输入/输出文件。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# EN: Compute a stable SHA256 hash for a JSON-serializable Python value.
#     sort_keys=True and compact separators make the serialized representation deterministic.
#     Here it is used to fingerprint each train_episode_indices list.
# 中文：为一个可 JSON 序列化的 Python 值计算稳定的 SHA256。
#     sort_keys=True 和紧凑分隔符能保证序列化结果稳定。
#     这里用于给每个 train_episode_indices 列表生成指纹。
def sha256_json_value(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


# EN: Load a JSON file and require its root to be a dictionary/object.
#     This is used for the split manifest. A list or scalar JSON would be structurally invalid.
# 中文：读取 JSON 文件，并要求 JSON 根节点必须是字典/对象。
#     这里主要用于读取 split 清单。如果根节点是列表或单个值，结构就不合法。
def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")

    return data


# EN: Strictly require a value to be an integer, while rejecting booleans.
#     In Python, bool is a subclass of int, so True/False must be rejected explicitly.
#     This prevents corrupted split metadata like train_count=true from passing validation.
# 中文：严格要求某个值必须是整数，同时明确拒绝布尔值。
#     在 Python 里 bool 是 int 的子类，所以 True/False 必须单独排除。
#     这样可以防止 train_count=true 这种错误 split 元数据通过检查。
def require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer; got {type(value).__name__}.")
    return value


# EN: Validate the fixed seed-42 logical split JSON before reading any heavy TFDS data.
#     This function confirms the split seed, source split, total train count, dataset names,
#     per-dataset train indices, duplicate indices, and out-of-range indices.
#     Its output is a normalized list of per-dataset records used by the actual statistics loop.
# 中文：在读取大型 TFDS 数据之前，先验证 seed=42 的逻辑 split JSON。
#     这个函数会检查 split seed、物理 source split、总训练数量、数据集名称、
#     每个数据集的训练索引、重复索引、越界索引。
#     它返回标准化后的每个子数据集记录，供后面的统计循环使用。
def validate_split_manifest(split_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Validate the fixed split and return normalized per-dataset records."""
    # EN: Check that the logical split was generated with the expected random seed.
    #     For this experiment, the official fixed split must use seed=42.
    # 中文：检查逻辑划分是否由预期随机种子生成。
    #     对这个实验来说，固定 split 必须使用 seed=42。
    seed = require_int(split_data.get("random_seed"), "random_seed")
    if seed != EXPECTED_SPLIT_SEED:
        raise StatisticsBuildError(
            f"Unexpected split seed: actual={seed}, expected={EXPECTED_SPLIT_SEED}."
        )

    # EN: The modified LIBERO TFDS datasets physically expose a source split named "train".
    source_split = split_data.get("source_split")
    if source_split != EXPECTED_SOURCE_SPLIT:
        raise StatisticsBuildError(
            "Unexpected physical TFDS source split: "
            f"actual={source_split!r}, expected={EXPECTED_SOURCE_SPLIT!r}."
        )

    totals = split_data.get("totals")
    if not isinstance(totals, dict):
        raise KeyError("Split JSON is missing the 'totals' object.")

    total_train = require_int(totals.get("train_count"), "totals.train_count")
    if total_train != EXPECTED_TOTAL_TRAIN_EPISODES:
        raise StatisticsBuildError(
            "Unexpected total number of train episodes: "
            f"actual={total_train}, expected={EXPECTED_TOTAL_TRAIN_EPISODES}."
        )

    raw_sub_datasets = split_data.get("sub_datasets")
    if not isinstance(raw_sub_datasets, list):
        raise KeyError("Split JSON is missing the 'sub_datasets' list.")

    # EN: Build a name -> split entry dictionary so each expected dataset can be looked up directly.
    #     This also allows duplicate dataset names to be detected immediately.
    # 中文：构建 name -> split entry 的字典，方便按数据集名称直接查找。
    #     这样也可以立刻发现重复的数据集名称。
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in raw_sub_datasets:
        if not isinstance(entry, dict):
            raise TypeError("Every sub_datasets entry must be an object.")
        name = entry.get("sub_dataset")
        if not isinstance(name, str) or not name:
            raise TypeError("Every sub_datasets entry must contain a non-empty sub_dataset string.")
        if name in by_name:
            raise StatisticsBuildError(f"Duplicate sub-dataset in split JSON: {name}")
        by_name[name] = entry

    missing = [name for name in EXPECTED_DATASETS if name not in by_name]
    unexpected = [name for name in by_name if name not in EXPECTED_DATASETS]
    if missing or unexpected:
        raise StatisticsBuildError(
            f"Split dataset mismatch; missing={missing}, unexpected={unexpected}."
        )

    # EN: normalized_entries is the clean version of the split information.
    #     It contains only validated fields needed by the statistics computation stage.
    # 中文：normalized_entries 是清洗后的 split 信息。
    #     它只保留后续统计阶段真正需要、且已经验证过的字段。
    normalized_entries: List[Dict[str, Any]] = []
    verified_total = 0

    for name in EXPECTED_DATASETS:
        entry = by_name[name]
        indices = entry.get("train_episode_indices")
        if not isinstance(indices, list) or not indices:
            raise StatisticsBuildError(f"{name}.train_episode_indices must be a non-empty list.")

        normalized_indices: List[int] = []
        for position, value in enumerate(indices):
            index = require_int(value, f"{name}.train_episode_indices[{position}]")
            if index < 0:
                raise StatisticsBuildError(f"Negative episode index for {name}: {index}")
            normalized_indices.append(index)

        if len(set(normalized_indices)) != len(normalized_indices):
            raise StatisticsBuildError(f"Duplicate train episode index found for {name}.")

        original_count = require_int(
            entry.get("original_episode_count"),
            f"{name}.original_episode_count",
        )
        declared_train_count = require_int(entry.get("train_count"), f"{name}.train_count")

        if len(normalized_indices) != declared_train_count:
            raise StatisticsBuildError(
                f"{name} train count mismatch: indices={len(normalized_indices)}, "
                f"declared={declared_train_count}."
            )

        out_of_range = [index for index in normalized_indices if index >= original_count]
        if out_of_range:
            raise StatisticsBuildError(
                f"{name} contains out-of-range train indices; first invalid={out_of_range[0]}, "
                f"episode_count={original_count}."
            )

        verified_total += declared_train_count
        normalized_entries.append(
            {
                "sub_dataset": name,
                "source_split": entry.get("source_split", EXPECTED_SOURCE_SPLIT),
                "original_episode_count": original_count,
                "train_count": declared_train_count,
                "train_episode_indices": normalized_indices,
                "train_episode_indices_sha256": sha256_json_value(normalized_indices),
            }
        )

    if verified_total != EXPECTED_TOTAL_TRAIN_EPISODES:
        raise StatisticsBuildError(
            f"Per-dataset train total is {verified_total}, expected {EXPECTED_TOTAL_TRAIN_EPISODES}."
        )

    return normalized_entries


# EN: Import TensorFlow and TensorFlow Datasets only when the script actually needs them.
#     This keeps --help, import checks, and syntax checks lightweight.
#     It also disables TensorFlow GPU visibility so TF does not reserve GPU memory.
# 中文：只有在脚本真正需要读取 TFDS 数据时才导入 TensorFlow 和 TensorFlow Datasets。
#     这样 --help、导入检查、语法检查会更轻量。
#     同时这里会禁用 TensorFlow 的 GPU 可见性，避免 TF 抢占 GPU 显存。
def import_tensorflow_modules() -> Tuple[Any, Any]:
    """Import TensorFlow lazily so --help and syntax checks stay lightweight."""
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as error:
        raise ImportError(
            "TensorFlow and tensorflow-datasets are required on the cloud runtime."
        ) from error

    # EN: Prevent TensorFlow from reserving GPU memory needed by PyTorch training.
    # 中文：禁止 TensorFlow 占用后续 PyTorch 训练所需的 GPU 显存。
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise StatisticsBuildError(
            "TensorFlow GPU visibility was configured too late. Run this script in a fresh process."
        ) from error

    return tf, tfds


# EN: Convert a TensorFlow tensor or array-like value into a flat float64 NumPy array.
#     The function also rejects empty arrays and non-finite values such as NaN or Inf.
#     This protects the normalization statistics from corrupted transitions.
# 中文：把 TensorFlow tensor 或类似数组的值转换成一维 float64 NumPy 数组。
#     同时拒绝空数组，以及 NaN/Inf 这种非有限数值。
#     这样可以防止损坏的 transition 污染归一化统计量。
def tensor_to_numpy(value: Any, field_name: str) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise StatisticsBuildError(f"Empty tensor encountered for {field_name}.")
    if not np.all(np.isfinite(array)):
        raise StatisticsBuildError(f"Non-finite value encountered for {field_name}.")
    return array


# EN: Convert raw LIBERO action into the exact action format expected by OpenVLA-OFT.
#     The first six action dimensions are kept unchanged.
#     The gripper dimension is remapped from raw LIBERO convention to OFT convention.
# 中文：把原始 LIBERO action 转换成 OpenVLA-OFT 期望的 action 格式。
#     前 6 个 action 维度保持不变。
#     夹爪维度会从 LIBERO 原始约定转换成 OFT 约定。
def standardize_libero_action(raw_action: np.ndarray) -> np.ndarray:
    """Apply the OpenVLA-OFT LIBERO action standardization."""
    if raw_action.shape != (ACTION_DIM,):
        raise StatisticsBuildError(
            f"LIBERO action must have shape ({ACTION_DIM},); got {raw_action.shape}."
        )

    # EN: Raw LIBERO uses -1=open and +1=close. OFT uses +1=open and 0=close.
    # 中文：原始 LIBERO 使用 -1=张开、+1=闭合；OFT 使用 +1=张开、0=闭合。
    gripper = 1.0 - np.clip(raw_action[-1], 0.0, 1.0)
    return np.concatenate((raw_action[:6], np.asarray([gripper], dtype=np.float64)))


# EN: Convert raw LIBERO observation.state into the 8-D proprio vector used by OFT.
#     The resulting vector is [state[:6], state[-2:]], i.e. six EEF values plus two gripper values.
# 中文：把原始 LIBERO observation.state 转成 OFT 使用的 8 维 proprio 向量。
#     最终向量是 [state[:6], state[-2:]]，也就是 6 个末端执行器状态 + 2 个夹爪状态。
def standardize_libero_proprio(raw_state: np.ndarray) -> np.ndarray:
    """Build the 8-D OFT proprio vector: EEF state (6) + gripper state (2)."""
    if raw_state.size < PROPRIO_DIM:
        raise StatisticsBuildError(
            f"LIBERO observation.state must contain at least {PROPRIO_DIM} values; "
            f"got {raw_state.size}."
        )

    proprio = np.concatenate((raw_state[:6], raw_state[-2:]))
    if proprio.shape != (PROPRIO_DIM,):
        raise StatisticsBuildError(
            f"Standardized proprio must have shape ({PROPRIO_DIM},); got {proprio.shape}."
        )
    return proprio


# EN: Retrieve the RLDS steps sequence from one episode.
#     Every RLDS episode must contain a "steps" field; otherwise the dataset is not usable here.
# 中文：从一个 episode 中取出 RLDS 的 steps 序列。
#     每个 RLDS episode 都必须包含 "steps" 字段，否则这里无法使用。
def get_episode_steps(episode: Mapping[str, Any], dataset_name: str, episode_index: int) -> Iterable[Any]:
    if "steps" not in episode:
        raise StatisticsBuildError(
            f"{dataset_name} episode {episode_index} is missing the RLDS 'steps' field."
        )
    return episode["steps"]


# EN: Extract one training transition's standardized action and proprio vectors.
#     This validates that the step contains action, observation, and observation.state.
#     It then converts tensors to NumPy and applies the same standardization as OpenVLA-OFT.
# 中文：提取一个训练 transition 的标准化 action 和 proprio 向量。
#     它会检查 step 是否包含 action、observation、observation.state。
#     然后把 tensor 转成 NumPy，并应用与 OpenVLA-OFT 一致的标准化。
def extract_action_and_proprio(
    step: Mapping[str, Any],
    dataset_name: str,
    episode_index: int,
    step_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if "action" not in step:
        raise StatisticsBuildError(
            f"Missing action at {dataset_name} episode={episode_index}, step={step_index}."
        )

    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        raise StatisticsBuildError(
            f"Missing observation object at {dataset_name} episode={episode_index}, step={step_index}."
        )
    if "state" not in observation:
        raise StatisticsBuildError(
            f"Missing observation.state at {dataset_name} episode={episode_index}, step={step_index}."
        )

    raw_action = tensor_to_numpy(
        step["action"],
        f"{dataset_name}[{episode_index}].steps[{step_index}].action",
    )
    raw_state = tensor_to_numpy(
        observation["state"],
        f"{dataset_name}[{episode_index}].steps[{step_index}].observation.state",
    )

    return standardize_libero_action(raw_action), standardize_libero_proprio(raw_state)


# EN: Compute per-dimension normalization statistics for a 2-D matrix of vectors.
#     Input shape must be [num_transitions, vector_dim].
#     The returned fields match the format expected by the downstream OpenVLA trainer.
# 中文：对一个二维向量矩阵计算逐维归一化统计量。
#     输入形状必须是 [transition 数量, 向量维度]。
#     返回字段与后续 OpenVLA trainer 期望的格式一致。
def compute_vector_statistics(values: np.ndarray, mask: Sequence[bool]) -> Dict[str, Any]:
    if values.ndim != 2:
        raise StatisticsBuildError(f"Statistics input must be 2-D; got shape={values.shape}.")
    if values.shape[0] == 0:
        raise StatisticsBuildError("Cannot compute statistics from zero transitions.")
    if values.shape[1] != len(mask):
        raise StatisticsBuildError(
            f"Statistics dimension {values.shape[1]} does not match mask length {len(mask)}."
        )

    # EN: Each statistic is computed along axis=0, meaning independently for each action/proprio dimension.
    #     For example, action mean produces seven values, one per action dimension.
    # 中文：每个统计量都沿 axis=0 计算，也就是对 action/proprio 的每个维度单独统计。
    #     例如 action mean 会得到 7 个值，每个 action 维度一个。
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "mask": [bool(item) for item in mask],
    }


# EN: Compute action and proprio statistics for one selected LIBERO sub-dataset.
#     This function loads the physical TFDS split, filters only the train_episode_indices,
#     extracts every transition, standardizes action/proprio, stacks them, and computes statistics.
# 中文：为一个指定的 LIBERO 子数据集计算 action 和 proprio 统计量。
#     该函数会加载物理 TFDS split，只筛选 train_episode_indices，
#     提取每个 transition，标准化 action/proprio，拼接后计算统计量。
def compute_sub_dataset_statistics(
    tfds: Any,
    dataset_root: Path,
    split_entry: Mapping[str, Any],
    progress_interval: int,
) -> Dict[str, Any]:
    dataset_name = str(split_entry["sub_dataset"])
    source_split = str(split_entry["source_split"])
    original_episode_count = int(split_entry["original_episode_count"])
    selected_indices = [int(index) for index in split_entry["train_episode_indices"]]
    # EN: Convert selected_indices to a set for O(1) membership checks while streaming episodes.
    #     The original selected_indices list is still kept for stable ordering and hashing.
    # 中文：把 selected_indices 转成 set，方便遍历 episode 时 O(1) 判断是否选中。
    #     原始 selected_indices 列表仍保留，用于稳定顺序和哈希记录。
    selected_set = set(selected_indices)

    print(
        f"[STATS] Loading {dataset_name}: source_split={source_split}, "
        f"selected_train_episodes={len(selected_indices)}"
    )

    # EN: Create a TFDS builder for the current sub-dataset.
    #     This checks that the named dataset can be found under dataset_root.
    # 中文：为当前子数据集创建 TFDS builder。
    #     这一步会检查 dataset_root 下能否找到这个数据集。
    try:
        builder = tfds.builder(dataset_name, data_dir=str(dataset_root))
    except Exception as error:
        raise StatisticsBuildError(
            f"Failed to create TFDS builder for {dataset_name} from {dataset_root}."
        ) from error

    if source_split not in builder.info.splits:
        raise StatisticsBuildError(
            f"TFDS dataset {dataset_name} does not contain split {source_split!r}."
        )

    physical_episode_count = int(builder.info.splits[source_split].num_examples)
    if physical_episode_count != original_episode_count:
        raise StatisticsBuildError(
            f"{dataset_name} physical episode count mismatch: dataset={physical_episode_count}, "
            f"split_json={original_episode_count}."
        )

    # EN: Load the physical TFDS split without file shuffling.
    #     shuffle_files=False is important because episode indices in the split JSON assume deterministic order.
    # 中文：加载物理 TFDS split，并关闭文件级 shuffle。
    #     shuffle_files=False 很重要，因为 split JSON 里的 episode 索引依赖确定性的顺序。
    try:
        dataset = builder.as_dataset(split=source_split, shuffle_files=False)
    except Exception as error:
        raise StatisticsBuildError(
            f"Failed to load TFDS dataset {dataset_name} from {dataset_root}."
        ) from error

    # EN: action_chunks and proprio_chunks collect per-episode arrays before final concatenation.
    #     This avoids repeatedly growing one large NumPy array inside the loop.
    # 中文：action_chunks 和 proprio_chunks 先按 episode 收集数组，最后再统一 concatenate。
    #     这样避免在循环里反复扩展一个巨大的 NumPy 数组。
    action_chunks: List[np.ndarray] = []
    proprio_chunks: List[np.ndarray] = []
    seen_selected_indices: set[int] = set()
    selected_episode_count = 0
    transition_count = 0

    # EN: Stream through every physical episode in deterministic TFDS order.
    #     Only episodes whose index is in selected_set are used for train-only statistics.
    # 中文：按 TFDS 的确定性顺序遍历每个物理 episode。
    #     只有索引在 selected_set 中的 episode 才会用于 train-only 统计量。
    for episode_index, episode in enumerate(dataset):
        if episode_index not in selected_set:
            continue

        if not isinstance(episode, Mapping):
            raise StatisticsBuildError(
                f"Unexpected episode type for {dataset_name}[{episode_index}]: {type(episode).__name__}."
            )

        episode_actions: List[np.ndarray] = []
        episode_proprios: List[np.ndarray] = []

        # EN: Each selected episode may contain many steps/transitions.
        #     Every step contributes one action vector and one proprio vector to the statistics pool.
        # 中文：每个选中的 episode 可能包含很多 step/transition。
        #     每个 step 都会贡献一个 action 向量和一个 proprio 向量到统计池。
        steps = get_episode_steps(episode, dataset_name, episode_index)
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise StatisticsBuildError(
                    f"Unexpected step type at {dataset_name}[{episode_index}][{step_index}]."
                )
            action, proprio = extract_action_and_proprio(
                step,
                dataset_name,
                episode_index,
                step_index,
            )
            episode_actions.append(action)
            episode_proprios.append(proprio)

        if not episode_actions:
            raise StatisticsBuildError(
                f"Selected episode contains zero steps: {dataset_name}[{episode_index}]."
            )

        # EN: Stack all vectors from this episode into matrices with shapes
        #     [episode_steps, ACTION_DIM] and [episode_steps, PROPRIO_DIM].
        # 中文：把当前 episode 的所有向量堆叠成矩阵，形状分别为
        #     [episode_steps, ACTION_DIM] 和 [episode_steps, PROPRIO_DIM]。
        actions_array = np.stack(episode_actions, axis=0)
        proprios_array = np.stack(episode_proprios, axis=0)

        action_chunks.append(actions_array)
        proprio_chunks.append(proprios_array)
        seen_selected_indices.add(episode_index)
        selected_episode_count += 1
        transition_count += int(actions_array.shape[0])

        if progress_interval > 0 and selected_episode_count % progress_interval == 0:
            print(
                f"[STATS] {dataset_name}: episodes={selected_episode_count}/{len(selected_indices)}, "
                f"transitions={transition_count}"
            )

    # EN: After streaming the dataset, verify that every requested train episode was actually seen.
    #     If not, the physical dataset order/count does not match the split JSON.
    # 中文：遍历完整个数据集后，确认每个请求的训练 episode 都真的被读到了。
    #     如果没有，说明物理数据集顺序/数量与 split JSON 不匹配。
    missing_indices = sorted(selected_set - seen_selected_indices)
    if missing_indices:
        raise StatisticsBuildError(
            f"{dataset_name} did not yield all requested train episodes; "
            f"missing_count={len(missing_indices)}, first_missing={missing_indices[0]}."
        )

    if selected_episode_count != len(selected_indices):
        raise StatisticsBuildError(
            f"{dataset_name} selected episode count mismatch: actual={selected_episode_count}, "
            f"expected={len(selected_indices)}."
        )

    # EN: Concatenate all selected episode chunks into full transition matrices.
    #     Statistics are computed over transitions, not merely over episodes.
    # 中文：把所有选中 episode 的块拼接成完整的 transition 矩阵。
    #     统计量是基于 transition 计算的，不只是基于 episode。
    all_actions = np.concatenate(action_chunks, axis=0)
    all_proprios = np.concatenate(proprio_chunks, axis=0)

    result = {
        "action": compute_vector_statistics(all_actions, ACTION_NORMALIZATION_MASK),
        "proprio": compute_vector_statistics(all_proprios, PROPRIO_NORMALIZATION_MASK),
        "num_transitions": int(transition_count),
        "num_trajectories": int(selected_episode_count),
        "source_split": source_split,
        "selected_episode_indices_sha256": split_entry["train_episode_indices_sha256"],
    }

    print(
        f"[STATS] Completed {dataset_name}: trajectories={selected_episode_count}, "
        f"transitions={transition_count}, action_dim={all_actions.shape[1]}, "
        f"proprio_dim={all_proprios.shape[1]}"
    )
    return result


# EN: Validate the final generated statistics before writing them to disk.
#     This is a second safety layer after split validation and extraction validation.
#     It checks dataset keys, trajectory counts, vector dimensions, finite values,
#     quantile ordering, and non-degenerate proprio statistics.
# 中文：在写入磁盘前验证最终生成的统计量。
#     这是 split 验证和提取验证之后的第二层安全检查。
#     它会检查数据集键、轨迹数量、向量维度、有限数值、分位数顺序、
#     以及 proprio 统计量是否退化为全零。
def validate_generated_statistics(
    datasets: Mapping[str, Any],
    expected_entries: Sequence[Mapping[str, Any]],
) -> None:
    # EN: Build expected per-dataset trajectory counts from the validated split entries.
    #     These counts must exactly match num_trajectories in the generated statistics.
    # 中文：从已经验证过的 split entries 中构建每个数据集的预期轨迹数量。
    #     这些数量必须与生成统计量中的 num_trajectories 完全一致。
    expected_counts = {
        str(entry["sub_dataset"]): int(entry["train_count"])
        for entry in expected_entries
    }

    if set(datasets) != set(EXPECTED_DATASETS):
        raise StatisticsBuildError(
            f"Generated dataset keys mismatch: actual={sorted(datasets)}, "
            f"expected={sorted(EXPECTED_DATASETS)}."
        )

    total_trajectories = 0
    for dataset_name in EXPECTED_DATASETS:
        stats = datasets[dataset_name]
        trajectories = int(stats["num_trajectories"])
        total_trajectories += trajectories

        if trajectories != expected_counts[dataset_name]:
            raise StatisticsBuildError(
                f"{dataset_name} trajectory count mismatch: actual={trajectories}, "
                f"expected={expected_counts[dataset_name]}."
            )

        # EN: Validate both action and proprio sections.
        #     Action must have 7 values per field; proprio must have 8 values per field.
        # 中文：同时验证 action 和 proprio 两个部分。
        #     action 每个字段必须有 7 个值；proprio 每个字段必须有 8 个值。
        for section_name, expected_dim in (("action", ACTION_DIM), ("proprio", PROPRIO_DIM)):
            section = stats.get(section_name)
            if not isinstance(section, dict):
                raise StatisticsBuildError(f"Missing {dataset_name}.{section_name} statistics.")

            for field_name in ("mean", "std", "max", "min", "q01", "q99", "mask"):
                values = section.get(field_name)
                if not isinstance(values, list) or len(values) != expected_dim:
                    raise StatisticsBuildError(
                        f"{dataset_name}.{section_name}.{field_name} must contain "
                        f"{expected_dim} values."
                    )

            numeric_fields = ("mean", "std", "max", "min", "q01", "q99")
            for field_name in numeric_fields:
                values = np.asarray(section[field_name], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise StatisticsBuildError(
                        f"Non-finite value in {dataset_name}.{section_name}.{field_name}."
                    )

            # EN: Enforce valid ordering: min <= q01 <= q99 <= max for every dimension.
            #     This catches corrupted quantiles or swapped fields.
            # 中文：强制检查每个维度都满足 min <= q01 <= q99 <= max。
            #     这样可以发现损坏的分位数或字段写反的问题。
            minimum = np.asarray(section["min"], dtype=np.float64)
            maximum = np.asarray(section["max"], dtype=np.float64)
            q01 = np.asarray(section["q01"], dtype=np.float64)
            q99 = np.asarray(section["q99"], dtype=np.float64)
            if np.any(minimum > q01) or np.any(q01 > q99) or np.any(q99 > maximum):
                raise StatisticsBuildError(
                    f"Invalid min/q01/q99/max ordering in {dataset_name}.{section_name}."
                )

        proprio_std = np.asarray(stats["proprio"]["std"], dtype=np.float64)
        if np.all(proprio_std == 0.0):
            raise StatisticsBuildError(
                f"{dataset_name} proprio statistics are all zero; refusing to mark them as formal statistics."
            )

    if total_trajectories != EXPECTED_TOTAL_TRAIN_EPISODES:
        raise StatisticsBuildError(
            f"Generated total trajectories={total_trajectories}, "
            f"expected={EXPECTED_TOTAL_TRAIN_EPISODES}."
        )


# EN: Write JSON atomically by first writing to a temporary file, then replacing the final path.
#     This prevents partially written/corrupted output if the process crashes during writing.
#     The overwrite flag protects existing formal statistics from accidental replacement.
# 中文：先写入临时文件，再原子替换最终路径，从而安全写 JSON。
#     如果写入过程中进程崩溃，可以避免留下半截损坏文件。
#     overwrite 参数用于防止正式统计量被意外覆盖。
def write_json_atomic(path: Path, data: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite only when replacement is intentional."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# EN: Main orchestration function for the full statistics-building pipeline.
#     The execution order is: parse args -> validate paths -> load split -> validate split ->
#     import TFDS -> compute per-dataset stats -> validate generated stats -> write output.
# 中文：整个统计量构建流程的主控函数。
#     执行顺序是：解析参数 -> 验证路径 -> 读取 split -> 验证 split ->
#     导入 TFDS -> 逐数据集计算统计量 -> 验证生成统计量 -> 写出文件。
def main() -> int:
    # EN: Read all user-provided command-line options and fill in defaults for omitted options.
    # 中文：读取用户传入的所有命令行参数，并为未提供的参数填入默认值。
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    split_json = args.split_json.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()

    if args.progress_interval_episodes < 0:
        raise ValueError("--progress-interval-episodes must be greater than or equal to zero.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    print(f"[STATS] Dataset root: {dataset_root}")
    print(f"[STATS] Split JSON:   {split_json}")
    print(f"[STATS] Output JSON:  {output_json}")

    # EN: Load and validate the logical split before touching the heavy dataset.
    #     If the split is wrong, the script fails early without expensive TFDS iteration.
    # 中文：先读取并验证逻辑 split，再接触大型数据集。
    #     如果 split 有问题，脚本会提前失败，不会浪费时间遍历 TFDS。
    split_data = load_json(split_json)
    split_entries = validate_split_manifest(split_data)
    split_sha256 = sha256_file(split_json)

    _, tfds = import_tensorflow_modules()

    # EN: Store the final statistics for each of the four LIBERO sub-datasets.
    #     The output JSON will place this dictionary under the top-level "datasets" field.
    # 中文：保存四个 LIBERO 子数据集各自的最终统计量。
    #     输出 JSON 会把这个字典放在顶层 "datasets" 字段下面。
    datasets: Dict[str, Any] = {}
    for split_entry in split_entries:
        dataset_name = str(split_entry["sub_dataset"])
        datasets[dataset_name] = compute_sub_dataset_statistics(
            tfds=tfds,
            dataset_root=dataset_root,
            split_entry=split_entry,
            progress_interval=args.progress_interval_episodes,
        )

    validate_generated_statistics(datasets, split_entries)

    total_transitions = sum(int(stats["num_transitions"]) for stats in datasets.values())
    total_trajectories = sum(int(stats["num_trajectories"]) for stats in datasets.values())

    # EN: Build the final provenance-rich statistics JSON.
    #     metadata explains exactly how the statistics were computed; datasets contains the actual numbers.
    # 中文：构建最终带完整来源信息的统计量 JSON。
    #     metadata 说明统计量到底如何生成；datasets 保存真正的数值。
    output_data: Dict[str, Any] = {
        "metadata": {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_debug_only": False,
            "stats_source": "train_only",
            "computed_from_split": "train",
            "episode_count": total_trajectories,
            "transition_count": total_transitions,
            "split_seed": EXPECTED_SPLIT_SEED,
            "split_json": str(split_json),
            "split_json_sha256": split_sha256,
            "dataset_root": str(dataset_root),
            "physical_source_split": EXPECTED_SOURCE_SPLIT,
            "dataset_names": list(EXPECTED_DATASETS),
            "action_dim": ACTION_DIM,
            "proprio_dim": PROPRIO_DIM,
            "action_standardization": (
                "first_6_raw; gripper=1-clip(raw_gripper,0,1); "
                "equivalent_to_OpenVLA_OFT_libero_dataset_transform"
            ),
            "proprio_standardization": (
                "concat(observation.state[:6], observation.state[-2:]); "
                "equivalent_to_OpenVLA_OFT_state_obs_keys_EEF_state_gripper_state"
            ),
        },
        "datasets": datasets,
    }

    write_json_atomic(output_json, output_data, overwrite=args.overwrite)
    output_sha256 = sha256_file(output_json)

    print(f"[STATS] Saved: {output_json}")
    print(f"[STATS] Output SHA256: {output_sha256}")
    print(
        f"[STATS] SUCCESS: train-only statistics built from "
        f"{total_trajectories} episodes and {total_transitions} transitions."
    )
    return 0


# EN: Standard Python entry point.
#     The script runs main() only when executed directly, not when imported by another module.
#     It converts normal completion and error cases into explicit process exit codes.
# 中文：标准 Python 入口。
#     只有直接运行这个脚本时才会执行 main()，被其他模块 import 时不会自动运行。
#     它会把正常完成和错误情况转换成明确的进程退出码。
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STATS] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"[STATS] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
