# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" TEFP8Callback """

from collections import OrderedDict
import os
import paddle
import numpy as np
import matplotlib.pyplot as plt
from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.trainer.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
)
from src.utils import logger


try:
    from transformer_engine.paddle.fp8_quantize_analysis_utils import (
        get_fp8_quantize_analysis_helper,
    )
except (ImportError, ModuleNotFoundError):
    get_fp8_quantize_analysis_helper = None


class TEFP8Callback(TrainerCallback):
    """
    Load and save Transformer Engine fp8 amax state hook
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """
        load fp8 amax state at the beginning of training.
        """
        model = kwargs.get("model", None)
        assert model is not None

        if args.enable_fp8_quantize_analysis:
            assert get_fp8_quantize_analysis_helper is not None
            name_mapping = {}
            for name, param in model.state_dict().items():
                name_mapping[param.name] = name
            get_fp8_quantize_analysis_helper().set_structure_name_mapping(name_mapping)
            get_fp8_quantize_analysis_helper().enable()

        if args.fp8_force_clear_state:
            logger.info("force clear Transformer Engine fp8 amax state dict.")
            return

        resume_from_checkpoint = (
            None if not args.resume_from_checkpoint else args.resume_from_checkpoint
        )
        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is None:
            return

        # if use distributed training
        if args.world_size > 1:
            process_index = args.process_index
            path = os.path.join(
                resume_from_checkpoint, f"te_fp8_state_{process_index}.pth"
            )
            if not os.path.isfile(path):
                logger.info(
                    f"Didn't find an Transformer Engine fp8 amax state file for process {process_index}, "
                    "if you are resuming a training that wasn't launched in a distributed fashion, "
                    "reproducibility is not guaranteed."
                )
                return
        else:
            path = os.path.join(resume_from_checkpoint, "te_fp8_state.pth")
            if not os.path.isfile(path):
                logger.info(
                    "Didn't find an Transformer Engine fp8 amax state file, if you are resuming a training that was "
                    "launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return

        logger.info(f"Loading Transformer Engine fp8 amax state to {path}")
        state_dict = paddle.load(path)

        for name, layer in model.named_sublayers(include_self=False):
            if hasattr(layer, "_set_fp8_state"):
                print("_set_fp8_state", name)
                layer._set_fp8_state(state_dict[f"{name}.fp8_state"])

        logger.info("load Transformer FP8 amax state dict success.")

    def on_step_begin(self, args, state, control, **kwargs):
        """set global_step"""
        if args.enable_fp8_quantize_analysis:
            assert get_fp8_quantize_analysis_helper is not None
            get_fp8_quantize_analysis_helper().set_step(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        """set global_step"""
        if (
            args.enable_fp8_quantize_analysis
            and args.enable_fp8_quantize_analysis
            and state.global_step % 5 == 0
        ):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            run_dir = args.output_dir
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            self.save_and_plot_analysis(output_dir)

    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint save.
        """
        model = kwargs.get("model", None)
        assert model is not None

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"

        run_dir = args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)

        os.makedirs(output_dir, exist_ok=True)

        if args.world_size > 1:
            # use global process_index to save
            process_index = args.process_index
            path = os.path.join(output_dir, f"te_fp8_state_{process_index}.pth")
        else:
            path = os.path.join(output_dir, "te_fp8_state_.pth")

        state_dict = OrderedDict()
        for name, layer in model.named_sublayers(include_self=False):
            if hasattr(layer, "_get_fp8_state"):
                state_dict[f"{name}.fp8_state"] = layer._get_fp8_state()

        logger.info(f"Saving Transformer Engine fp8 amax state to {path}")
        paddle.save(state_dict, path)

        if args.enable_fp8_quantize_analysis:
            self.save_and_plot_analysis(output_dir)

    def save_and_plot_analysis(self, output_dir):
        """doc"""
        os.makedirs(output_dir, exist_ok=True)

        fp8_quantize_analysis_helper = get_fp8_quantize_analysis_helper()

        global_state_list = []
        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        pp_group = hcg.get_pipe_parallel_group()
        paddle.distributed.all_gather_object(
            global_state_list, fp8_quantize_analysis_helper.state, pp_group
        )

        merged_state = {}
        seen_keys = set()

        for state in global_state_list:
            for key in state:
                if key in seen_keys:
                    raise ValueError(f"Duplicate key detected: {key}")
                seen_keys.add(key)
                merged_state[key] = state[key]

        if paddle.distributed.get_rank() == 0:
            path = os.path.join(output_dir, "te_fp8_quantize_analysis_state.pth")
            paddle.save(merged_state, path)

            state = merged_state
            # Extract and sort keys
            x_keys = sorted(
                [key for key in state.keys() if "#x" in key],
                key=lambda s: int(s.split(".")[2]),
            )
            w_keys = sorted(
                [key for key in state.keys() if "#w" in key],
                key=lambda s: int(s.split(".")[2]),
            )
            dy_keys = sorted(
                [key for key in state.keys() if "#dy" in key],
                key=lambda s: int(s.split(".")[2]),
            )

            # Function to extract data
            def extract_data(keys):
                (
                    underflow_ratio_list,
                    amax_list,
                    rmse_list,
                    kurtosis_list,
                    mean_list,
                    maxmin_list,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for key in keys:
                    # analysis_state["underflow_ratio"] = underflow_ratio
                    # analysis_state["rmse"] = rmse
                    # analysis_state["kurtosis"] = kurtosis
                    # analysis_state["amax"] = amax
                    # analysis_state["max"] = max
                    # analysis_state["min"] = min
                    # analysis_state["mean"] = mean.item()
                    # analysis_state["std"] = std.item()
                    # analysis_state["step"] = self.step
                    # analysis_state["layer_idx"] = layer_idx

                    underflow_ratio_list.append([item[0] for item in state[key]])
                    amax_list.append([item[3] for item in state[key]])
                    rmse_list.append([item[1] for item in state[key]])
                    kurtosis_list.append([item[2] for item in state[key]])
                    mean_list.append([item[6] for item in state[key]])
                    maxmin_list.append([item[4] - item[5] for item in state[key]])

                return (
                    np.array(amax_list).T,
                    np.array(underflow_ratio_list).T,
                    np.array(rmse_list).T,
                    np.array(kurtosis_list).T,
                    np.array(mean_list).T,
                    np.array(maxmin_list).T,
                )

            # Extract data for x, w, and dy
            x_data = extract_data(x_keys)
            w_data = extract_data(w_keys)
            dy_data = extract_data(dy_keys)

            # Create meshgrid for layers and iterations
            layers = [int(key.split(".")[2]) for key in x_keys]
            iterations = range(
                state[x_keys[0]][0][8], state[x_keys[0]][0][8] + len(state[x_keys[0]])
            )
            layers, iterations = np.meshgrid(layers, iterations)

            # Labels
            zlabel_list = [
                "AMax",
                "Underflow Ratio",
                "RMSE",
                "Kurtosis",
                "Mean",
                "Max-Min",
            ]
            row_labels = ["X", "W", "dY"]

            # Plot settings
            fig = plt.figure(figsize=(36, 18))
            data_list = [x_data, w_data, dy_data]

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            # Plotting
            for row, (data, row_label) in enumerate(zip(data_list, row_labels)):
                for col, plot_data in enumerate(data):
                    ax = fig.add_subplot(3, 6, row * 6 + col + 1, projection="3d")
                    surf = ax.plot_surface(
                        layers, iterations, plot_data, cmap="coolwarm", edgecolor="none"
                    )
                    ax.set_xlabel("Layers")
                    ax.set_ylabel("Iteration")
                    ax.set_title(zlabel_list[col])
                    fig.colorbar(surf, ax=ax, shrink=0.3, aspect=10)
                fig.text(
                    0.005,
                    1 - (row + 0.5) / 3,
                    row_label,
                    va="center",
                    ha="center",
                    fontsize=20,
                )

            plt.tight_layout()
            plt.show()
            plt.tight_layout(rect=[0.01, 0.01, 0.93, 0.93])
            plt.savefig(os.path.join(output_dir, "fp8_analysis"), dpi=300)

        paddle.distributed.barrier()
