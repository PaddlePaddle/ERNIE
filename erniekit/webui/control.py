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

"""
Event response mechanism handling
"""

import asyncio
import json
import os
import subprocess
import time

import aiohttp
import gradio as gr

import erniekit.webui.lang as la
from erniekit.webui import common
from erniekit.webui.alert import alert
from erniekit.webui.chatbot import chatbot as chat_generator
from erniekit.webui.common import config
from erniekit.webui.runner import CommandRunner

has_shown_info = False


def basic_reaction(manager):
    """
    Perform basic configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values

    """
    setup_model_name_or_path_update(manager)
    reflash_compute_type_by_fine_tuning(manager)


def eval_reaction(manager, runner, module):
    """
    Perform eval configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance
        module (str): Module name

    """
    setup_command_buttons(manager, runner, module)
    eval_specific_elem_change(manager)
    reflash_existed_dataset_path_prob(
        manager,
        section="eval",
        dataset_id="eval_dataset",
        path_id="eval_existed_dataset_path",
        prob_id="eval_existed_dataset_prob",
        type_id="eval_existed_dataset_type",
    )
    dataset_type_select_reflash(manager, "eval", "eval_customize")

    load_update_dataset_config(
        manager,
        module="eval",
        dataset_id="eval_dataset",
        path_id="eval_existed_dataset_path",
        prob_id="eval_existed_dataset_prob",
        type_id="eval_existed_dataset_type",
    )


def export_reaction(manager, runner, module):
    """
    Perform export configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance
        module (str): Module name

    """
    setup_command_buttons(manager, runner, module)


def chat_reaction(manager, runner):
    """
    Perform chat configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance

    """
    chat_load_model_button(manager, runner)
    setup_chatbot_response(manager)
    chat_upload_model_button(manager, runner)
    chat_status_button_handler(manager, CommandRunner())
    chat_update_max_new_len_max(manager)
    chat_role_setting_system_prompt_handler(manager)


def train_reaction(manager, runner, module):
    """
    Perform train configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance
        module (str): Module name

    """
    setup_command_buttons(manager, runner, module)
    setup_update_stage(manager)
    train_specific_elem_change(manager)
    train_epochs_change(manager)
    dataset_type_select_reflash(manager, "train", "train_customize")
    dataset_type_select_reflash(manager, "train", "eval_customize")

    reflash_existed_dataset_path_prob(
        manager,
        section="train",
        dataset_id="train_dataset",
        path_id="train_existed_dataset_path",
        prob_id="train_existed_dataset_prob",
        type_id="train_existed_dataset_type",
    )

    reflash_existed_dataset_path_prob(
        manager,
        section="train",
        dataset_id="eval_dataset",
        path_id="eval_existed_dataset_path",
        prob_id="eval_existed_dataset_prob",
        type_id="eval_existed_dataset_type",
    )

    load_update_dataset_config(
        manager,
        module="train",
        dataset_id="train_dataset",
        path_id="train_existed_dataset_path",
        prob_id="train_existed_dataset_prob",
        type_id="train_existed_dataset_type",
    )

    load_update_dataset_config(
        manager,
        module="train",
        dataset_id="eval_dataset",
        path_id="eval_existed_dataset_path",
        prob_id="eval_existed_dataset_prob",
        type_id="eval_existed_dataset_type",
    )


def dataset_type_select_reflash(manager, module, elem):
    """
    Refresh dataset configuration based on selected dataset type.

    Args:
        manager (object): Configuration manager
        module (str): Module name
        elem (str): Element name representing the dataset type
    """

    dataset_type = manager.get_elem_by_id(module, f"{elem}_dataset_type")
    select_dataset_type = manager.get_elem_by_id(module, f"{elem}_select_dataset_type")

    def add_selection(current_selection, current_text):
        if current_selection is not None:
            if current_text == "":
                new_text = current_selection
            else:
                new_text = current_text + "," + current_selection
        else:
            new_text = current_text

        return new_text, None

    select_dataset_type.change(
        fn=add_selection, inputs=[select_dataset_type, dataset_type], outputs=[dataset_type, select_dataset_type]
    )


def train_epochs_change(manager):
    """
    Handle interactions between training epochs and max steps parameters.

    Args:
        manager (manager): component manager
    """

    num_train_epochs = manager.get_elem_by_id("train", "num_train_epochs")
    max_steps = manager.get_elem_by_id("train", "max_steps")

    def on_num_train_epochs_change(max_steps_value):
        global has_shown_info
        if max_steps_value > 0 and not has_shown_info:
            gr.Info(alert.get("max_steps_notice", "info"))
            has_shown_info = True

        pass

    num_train_epochs.change(fn=on_num_train_epochs_change, inputs=[max_steps], outputs=[])


def reflash_existed_dataset_path_prob(manager, section, dataset_id, path_id, prob_id, type_id):
    """
    Refresh the probability configuration for existing dataset paths based on type changes.

    Args:
        manager (manager): Configuration manager for component values
        section (str): Configuration section containing dataset settings
        dataset_id (str): ID of the dataset element
        path_id (str): ID of the path configuration element
        prob_id (str): ID of the probability configuration element
        type_id (str): ID of the dataset type configuration element
    """

    dataset = manager.get_elem_by_id(section, dataset_id)
    path_elem = manager.get_elem_by_id(section, path_id)
    prob_elem = manager.get_elem_by_id(section, prob_id)
    type_elem = manager.get_elem_by_id(section, type_id)

    def update_path_prob(dataset_names):
        paths = []
        probs = []
        types = []

        for dataset_name in dataset_names:
            info = config.get_dataset_info_kwagrs(dataset_name)
            if info is not None:

                path = info.get("path")
                prob = info.get("prob")
                type = info.get("type")

                if path is None:
                    path_error_msg = alert.get("preview_data_non_path", "warning").format(dataset_name)
                    print(path_error_msg)
                    gr.Warning(path_error_msg)
                    continue
                if prob is None:
                    prob_error_msg = alert.get("preview_data_non_prob", "warning").format(dataset_name)
                    gr.Warning(prob_error_msg)
                    print(prob_error_msg)
                    continue
                if type is None:
                    type_error_msg = alert.get("preview_data_non_type", "warning").format(dataset_name)
                    gr.Warning(type_error_msg)
                    print(type_error_msg)
                    continue

                paths.append(path)
                probs.append(prob)
                types.append(type)

        path_str = ", ".join(paths)
        type_str = ", ".join(types)
        prob_str = ", ".join([str(p) for p in probs])

        return gr.update(value=path_str), gr.update(value=prob_str), gr.update(value=type_str)

    dataset.change(fn=update_path_prob, inputs=[dataset], outputs=[path_elem, prob_elem, type_elem])


def react_preview_dataset_button(manager, preview_button, module, elem_id):
    """
    Render dataset preview buttons with support for multiple datasets, pagination,
    and dataset switching.

    Args:
        manager (Manager): Component manager instance
        preview_button (Button): Gradio Button component for triggering preview
        module (str): Module identifier (e.g., "train", "eval")
        elem_id (str): Element identifier for the dataset configuration
    """

    language = manager.get_elem_by_id("basic", "language")

    current_page = gr.State(value=1)
    current_dataset_index = gr.State(value=0)

    with gr.Column(visible=False, elem_classes="modal-overlay") as overlay:
        pass

    with gr.Column(visible=False, elem_classes="modal-box") as popup:
        with gr.Row():
            with gr.Column(scale=1):
                dataset_preview_title = gr.Markdown(value="## 数据集预览")
            with gr.Column(scale=1):
                page_info = gr.Markdown(value="第1页/共1页", elem_classes="page-info")
            with gr.Row():
                dataset_info = gr.Markdown("")

        json_preview = gr.Code(language="json", label="数据内容", show_label=False)

        with gr.Row(elem_classes="pagination-controls"):
            prev_btn = gr.Button("上一页", size="lg", visible=False)
            manager.add_elem(module, "prev_btn", prev_btn)
            next_btn = gr.Button("下一页", size="lg", visible=False)
            close_button = gr.Button("关闭", size="lg")

        with gr.Row(elem_classes="pagination-controls"):
            prev_dataset_btn = gr.Button("上组数据集", size="lg", visible=False)
            next_dataset_btn = gr.Button("下组数据集", size="lg", visible=False)

    def get_data_paths():
        user_data_path_value = manager.get_component_value(module_id=module, elem_id=f"{elem_id}_dataset_path")
        default_data_path_value = manager.get_component_value(module_id=module, elem_id=f"{elem_id}_dataset")

        if default_data_path_value is not None and default_data_path_value:
            return [common.DEFAULT_DATASET_PATH]

        if user_data_path_value:
            data_paths = []
            for path in user_data_path_value.split(","):
                path = path.strip()
                if not path:
                    continue

                if os.path.isabs(path):
                    full_path = os.path.normpath(path)
                else:
                    full_path = os.path.normpath(os.path.join(common.ROOT_PATH, path))

                if os.path.exists(full_path):
                    data_paths.append(full_path)
                else:
                    preview_data_non_existent = alert.get("preview_data_non_existent", "error").format(path)
                    print(preview_data_non_existent)
                    gr.Warning(preview_data_non_existent)

            if data_paths:
                return data_paths

        return []

    def load_data(data_paths, dataset_index=0, page=1, page_size=10, max_pages=100):
        if not data_paths or dataset_index >= len(data_paths):
            return [], 0, 0, dataset_index, len(data_paths), ""

        path = data_paths[dataset_index]
        all_data = []

        try:
            if path.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_data.append(json.loads(line))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_data.extend(data)
        except FileNotFoundError as fe:
            preview_data_error = alert.get("preview_data_error", "error").format(path, fe)
            print(preview_data_error)
            raise gr.Error(preview_data_error)
        except json.JSONDecodeError:
            preview_data_non_json = alert.get("preview_data_non_json", "error").format(path)
            print(preview_data_non_json)
            gr.Warning(preview_data_non_json)
        except Exception as e:
            preview_data_error = alert.get("preview_data_error", "error").format(path, e)
            print(preview_data_error)
            raise gr.Error(preview_data_error)

        total = len(all_data)
        cutoff_page = False
        if total == 0:
            return [], 0, 0, dataset_index, len(data_paths), path

        total_pages = (total + page_size - 1) // page_size
        effective_max_page = min(max_pages, total_pages)

        if total_pages > max_pages:
            cutoff_page = True

        safe_page = min(max(1, page), effective_max_page)

        start_idx = (safe_page - 1) * page_size
        end_idx = start_idx + page_size

        return all_data[start_idx:end_idx], total, cutoff_page, dataset_index, len(data_paths), path

    def keep_last_five_levels(path_str):
        parts = path_str.strip('/').split('/')
        if len(parts) <= 5:
            return '/' + path_str.strip('/')
        else:
            return '/' + '/'.join(parts[-5:])

    def show_popup(language, page_number, dataset_index):
        page = int(page_number) if isinstance(page_number, str) else page_number
        data_paths = get_data_paths()
        data, total_records, cutoff_page, current_dataset, total_datasets, current_path = load_data(
            data_paths, dataset_index=dataset_index, page=page, page_size=10
        )

        total_pages = max(1, (total_records + 9) // 10)

        dataset_text = la.get("dataset_info", language, "value").format(
            current_dataset + 1, total_datasets, keep_last_five_levels(current_path)
        )

        has_multiple_datasets = total_datasets > 1
        show_prev_dataset = has_multiple_datasets and current_dataset > 0
        show_next_dataset = has_multiple_datasets and current_dataset < total_datasets - 1
        page_info_text = la.get("page_info", language, "value").format(page, total_pages)

        return (
            gr.Column(visible=True),
            gr.Column(visible=True),
            json.dumps(data, ensure_ascii=False, indent=2),
            str(page),
            page_info_text,
            dataset_text,
            gr.Button(visible=page > 1),
            gr.Button(visible=page < total_pages),
            gr.Button(visible=show_prev_dataset),
            gr.Button(visible=show_next_dataset),
            current_dataset,
        )

    def hide_popup():
        return (
            gr.Column(visible=False),
            gr.Column(visible=False),
            None,
            "1",
            "第 1 页，共 1 页",
            "",
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            0,
        )

    def update_page(language, page_num, dataset_index):
        return show_popup(language, page_num, dataset_index)

    def on_preview_check(language, current_page, current_dataset_index):
        dataset_path_value = manager.get_component_value(module_id=module, elem_id=f"{elem_id}_dataset_path")
        if dataset_path_value == "" or dataset_path_value is None:
            gr.Warning(alert.get("preview_data_path_none", "warning"))
            return hide_popup()
        else:
            data_paths = get_data_paths()
            if not data_paths:
                return hide_popup()
            return show_popup(language, current_page, current_dataset_index)

    def navigate_dataset(dataset_index, direction):
        data_paths = get_data_paths()
        if not data_paths:
            return 0, gr.Button(visible=False), gr.Button(visible=False)

        new_index = dataset_index + direction
        new_index = max(0, min(new_index, len(data_paths) - 1))

        show_prev = len(data_paths) > 1 and new_index > 0
        show_next = len(data_paths) > 1 and new_index < len(data_paths) - 1

        return new_index, gr.Button(visible=show_prev), gr.Button(visible=show_next)

    def go_to_prev_dataset(dataset_index):
        new_index, prev_btn_vis, next_btn_vis = navigate_dataset(dataset_index, -1)
        return (
            new_index,
            prev_btn_vis,
            next_btn_vis,
            "1",
        )

    def go_to_next_dataset(dataset_index):
        new_index, prev_btn_vis, next_btn_vis = navigate_dataset(dataset_index, 1)
        return (
            new_index,
            prev_btn_vis,
            next_btn_vis,
            "1",
        )

    preview_button.click(
        on_preview_check,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    prev_btn.click(
        lambda page: int(page) - 1 if isinstance(page, str) else page - 1,
        inputs=[current_page],
        outputs=[current_page],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    next_btn.click(
        lambda page: int(page) + 1 if isinstance(page, str) else page + 1,
        inputs=[current_page],
        outputs=[current_page],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    close_button.click(
        hide_popup,
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    prev_dataset_btn.click(
        go_to_prev_dataset,
        inputs=[current_dataset_index],
        outputs=[
            current_dataset_index,
            prev_dataset_btn,
            next_dataset_btn,
            current_page,
        ],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    next_dataset_btn.click(
        go_to_next_dataset,
        inputs=[current_dataset_index],
        outputs=[
            current_dataset_index,
            prev_dataset_btn,
            next_dataset_btn,
            current_page,
        ],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    def update_language(language):
        dataset_preview_title_la = la.get(key="dataset_preview_title", lang=language, prop="value")
        page_info_la = la.get(key="page_info", lang=language, prop="value")
        dataset_info_la = la.get(key="dataset_info", lang=language, prop="value")
        next_btn_la = la.get(key="next_btn", lang=language, prop="value")
        prev_btn_la = la.get(key="prev_btn", lang=language, prop="value")
        close_button_la = la.get(key="close_button", lang=language, prop="value")
        prev_dataset_btn_la = la.get(key="prev_dataset_btn", lang=language, prop="value")
        next_dataset_btn_la = la.get(key="next_dataset_btn", lang=language, prop="value")

        return (
            dataset_preview_title_la,
            page_info_la,
            dataset_info_la,
            next_btn_la,
            prev_btn_la,
            close_button_la,
            prev_dataset_btn_la,
            next_dataset_btn_la,
        )

    language.change(
        fn=update_language,
        inputs=[language],
        outputs=[
            dataset_preview_title,
            page_info,
            dataset_info,
            next_btn,
            prev_btn,
            close_button,
            prev_dataset_btn,
            next_dataset_btn,
        ],
    )


def setup_command_buttons(manager, runner, module):
    """
    Configure command execution buttons for a specific module.

    Args:
        manager (manager): the component manager
        runner (runner): the execution runner
        module (str): Name of the target module (e.g., "training", "inference")

    Returns:
        None
    """
    setup_preview_button(manager, module)
    setup_start_button(manager, runner, module)
    setup_stop_button(manager, runner, module)
    setup_clean_button(manager, runner, module)


def train_specific_elem_change(manager):
    """
    Evaluate changes to a specific UI element and trigger dependent updates.

    Args:
        manager (manager): Manager instance responsible for component state
    """
    language = manager.get_elem_by_id("basic", "language")
    train_dataset = manager.get_specific_elem_by_id("train", "train_dataset")
    eval_dataset = manager.get_specific_elem_by_id("train", "eval_dataset")

    def update_specific_elem_dropdown(language):
        train_dataset_label = la.get(key="train_dataset", lang=language, prop="label")
        eval_dataset_label = la.get(key="eval_dataset", lang=language, prop="label")

        train_dataset_info = la.get(key="train_dataset", lang=language, prop="info")
        eval_dataset_info = la.get(key="eval_dataset", lang=language, prop="info")

        return (
            gr.update(label=train_dataset_label, info=train_dataset_info),
            gr.update(label=eval_dataset_label, info=eval_dataset_info),
        )

    def update_train_dataset_data(inputs):
        manager.set_specific_component_value("train", "train_dataset", inputs)
        return gr.update(value=inputs)

    def update_eval_dataset_data(inputs):
        manager.set_specific_component_value("train", "eval_dataset", inputs)
        return gr.update(value=inputs)

    train_dataset.change(fn=update_train_dataset_data, inputs=[train_dataset], outputs=[train_dataset])

    eval_dataset.change(fn=update_eval_dataset_data, inputs=[eval_dataset], outputs=[eval_dataset])

    language.change(
        fn=update_specific_elem_dropdown,
        inputs=[language],
        outputs=[train_dataset, eval_dataset],
    )


def eval_specific_elem_change(manager):
    """
    Evaluates specific element changes.

    Args:
        manager (manager): Manages component states and updates.
    """
    language = manager.get_elem_by_id("basic", "language")
    eval_dataset = manager.get_specific_elem_by_id("eval", "eval_dataset")

    def update_specific_elem_dropdown(language):
        eval_dataset_label = la.get(key="eval_dataset", lang=language, prop="label")
        eval_dataset_info = la.get(key="eval_dataset", lang=language, prop="info")
        return gr.update(label=eval_dataset_label, info=eval_dataset_info)

    def update_eval_dataset_data(inputs):
        manager.set_specific_component_value("eval", "eval_dataset", inputs)
        return gr.update(value=inputs)

    eval_dataset.change(fn=update_eval_dataset_data, inputs=[eval_dataset], outputs=[eval_dataset])

    language.change(fn=update_specific_elem_dropdown, inputs=[language], outputs=[eval_dataset])


def setup_model_name_or_path_update(manager):
    """
    Configure model name or path update functionality.

    Args:
        manager (object): Manager instance handling model configurations
    """

    model_name = manager.get_elem_by_id("basic", "model_name")
    model_name_or_path = manager.get_elem_by_id("basic", "model_name_or_path")
    model_source = manager.get_elem_by_id("basic", "model_source")

    def update_path(selected_model, select):

        if selected_model == "Customization":
            gr.Info(alert.get("custom_model_notice", "info"))
            return (
                gr.update(interactive=True, value=""),
                gr.update(value="Local", choices=config.get_choices_kwargs("model_source_custom")),
            )

        if config.is_thought_model(selected_model):
            gr.Info(alert.get("thought_model_notice", "info"))

        path = config.get_model_name_or_path(selected_model, select.replace(" ", "_").lower())

        manager._component_values["basic"]["model_name_or_path"] = path
        return path, gr.update(value="Local", choices=config.get_choices_kwargs("model_source_ernie"))

    def update_path_selector(selected_model, model_source_value):

        path = config.get_model_name_or_path(selected_model, model_source_value.replace(" ", "_").lower())

        manager._component_values["basic"]["model_name_or_path"] = path
        return path

    model_name.change(
        fn=update_path,
        inputs=[model_name, model_source],
        outputs=[model_name_or_path, model_source],
    )

    model_source.change(fn=update_path_selector, inputs=[model_name, model_source], outputs=[model_name_or_path])


def reflash_compute_type_by_fine_tuning(manager):
    """
    Refresh compute type based on fine-tuning configuration.

    Args:
        manager (object): Manager instance handling compute resources

    """

    fine_tuning = manager.get_elem_by_id("basic", "fine_tuning")
    compute_type = manager.get_elem_by_id("basic", "compute_type")

    last_update_time = [0]

    def update_choices(fine_tuning_value):

        current_time = time.time()
        last_update_time[0] = current_time

        time.sleep(0.1)

        if current_time != last_update_time[0]:
            return gr.update()

        try:
            if fine_tuning_value == "LoRA":
                choices = config.get_choices_kwargs("compute_type_LoRA")
            else:
                choices = config.get_choices_kwargs("compute_type_Full")

            return gr.update(choices=choices, value="bf16")
        except Exception as e:
            print(f"update fail: {e}")
            return gr.update()

    def alert_compute_type_fp8(compute_type_value):
        if compute_type_value == "fp8":
            gr.Warning(alert.get("compute_type_fp8_notice", "warning"))

    fine_tuning.change(fn=update_choices, inputs=[fine_tuning], outputs=[compute_type])
    compute_type.change(fn=alert_compute_type_fp8, inputs=[compute_type], outputs=[])


def setup_preview_button(manager, module):
    """
    Initialize preview button configuration.

    Args:
        manager (object): Component manager instance
        module (str): Target module identifier

    """

    preview_command_btn = manager.get_elem_by_id(module, "preview_command_btn")
    command_preview = manager.get_elem_by_id(module, "command_preview")
    output_text = manager.get_elem_by_id(module, "output_text")
    output_container = manager.get_elem_by_id(module, "output_container")
    stage = manager.get_elem_by_id("train", "stage")

    def show_command_preview():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
        )

    def module_command_preview(stage):

        if module == "train":
            execute_path = f"train_{stage.lower()}_yaml_path"
        else:
            execute_path = module + "_yaml_path"

        update_config_yaml(manager, execute_path, module, True)

        return common.yaml_to_args(config.get_execute_yaml_path(execute_path), config.get_commands_cli(module))

    if preview_command_btn and command_preview and output_text and output_container:
        preview_command_btn.click(
            fn=show_command_preview,
            inputs=[],
            outputs=[command_preview, output_text, output_container, command_preview],
        ).then(fn=module_command_preview, inputs=[stage], outputs=command_preview)


async def execute_command(runner, command):
    """
    Asynchronously execute a shell command.

    Args:
        runner (object): Execution context or runner instance
        command (str): Command string to be executed
    """
    async for output, _, _ in runner.execute(command):
        yield output


def chat_load_model_button(manager, runner):
    """
    Configure the chat interface's model loading button.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Execution handler for model loading

    """
    load_model_btn = manager.get_elem_by_id("chat", "load_model_btn")
    output_text = manager.get_elem_by_id("chat", "output_text")
    port = manager.get_elem_by_id("chat", "port")
    save_port = manager.get_elem_by_id("chat", "save_port")

    async def chat_start_execution(port):

        update_config_yaml(manager, "chat_yaml_path", "chat")

        command = config.get_execute_command("chat")
        async for output in execute_command(runner, command):
            yield output, gr.update(value=port)

    load_model_btn.click(fn=chat_start_execution, inputs=[port], outputs=[output_text, save_port])


def chat_upload_model_button(manager, runner):
    """
    Initialize the chat interface's model upload button.

    Args:
        manager (object): Manager for handling UI component states
        runner (object): Handler for model upload execution
    """
    unload_model_btn = manager.get_elem_by_id("chat", "unload_model_btn")
    output_text = manager.get_elem_by_id("chat", "output_text")

    async def chat_stop_current_process():
        result = await runner.stop()
        return result

    if unload_model_btn and output_text:
        unload_model_btn.click(fn=chat_stop_current_process, inputs=[], outputs=output_text)


def setup_start_button(manager, runner, module):
    """
    Configure the start button for a specific module.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Execution handler for module operations
        module (str): target module
    """
    start_btn = manager.get_elem_by_id(module, "start_btn")
    command_preview = manager.get_elem_by_id(module, "command_preview")
    output_text = manager.get_elem_by_id(module, "output_text")
    output_container = manager.get_elem_by_id(module, "output_container")
    start_merge_btn = manager.get_elem_by_id("export", "start_merge_btn")
    start_split_btn = manager.get_elem_by_id("export", "start_split_btn")
    stage = manager.get_elem_by_id("train", "stage")

    def show_output_text():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            "",
        )

    async def start_execution(stage):
        if module == "train":
            execute_path = f"train_{stage.lower()}_yaml_path"
            command_name = f"train_{stage.lower()}"
        else:
            execute_path = module + "_yaml_path"
            command_name = module

        update_config_yaml(manager, execute_path, module)
        command = config.get_execute_command(command_name)

        async for output in execute_command(runner, command):
            yield output

    async def start_export_merge_execution():
        execute_path = "export_yaml_path"
        update_config_yaml(manager, execute_path, "export")
        command = config.get_execute_command("export")

        async for output in execute_command(runner, command):
            yield output

    async def start_export_split_execution():
        execute_path = "export_yaml_path"
        update_config_yaml(manager, execute_path, "export")
        command = config.get_execute_command("split")

        async for output in execute_command(runner, command):
            yield output

    if start_btn and output_text:
        start_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_execution, inputs=[stage], outputs=output_text)

    if start_merge_btn and start_split_btn:
        start_merge_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_export_merge_execution, inputs=[], outputs=output_text)

        start_split_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_export_split_execution, inputs=[], outputs=output_text)


def setup_stop_button(manager, runner, module):
    """
    Configure the stop button for a specific module.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Execution handler for module operations
        module (str): Name of the target module
    """
    stop_btn = manager.get_elem_by_id(module, "stop_btn")
    output_text = manager.get_elem_by_id(module, "output_text")

    async def stop_current_process():
        result = await runner.stop()
        return result

    if stop_btn and output_text:
        stop_btn.click(fn=stop_current_process, inputs=[], outputs=output_text)


def setup_clean_button(manager, runner, module):
    """
    Configure the clean button for resource cleanup.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Handler for cleanup operations
        module (str):   Name of the target module
    """
    clean_btn = manager.get_elem_by_id(module, "clean_btn")
    output_text = manager.get_elem_by_id(module, "output_text")

    def clear_output():
        return runner.clear_output()

    if clean_btn and output_text:
        clean_btn.click(fn=clear_output, inputs=[], outputs=output_text)


def model_update_callback(manager, stage):
    """
    Handle model update events at specified stages.

    Args:
        manager (object): Manager for model lifecycle
        stage (str): Update stage
    """
    sft_params = config.get_default_dict_module("train_sft")
    dpo_params = config.get_default_dict_module("train_dpo")

    updates = {}

    if stage == "SFT":
        params = sft_params
    elif stage == "DPO":
        params = dpo_params
    else:
        return updates

    train_componet_elem_list = manager.get_dependencies("basic.best_config.train")
    basic_componet_elem_list = manager.get_dependencies("basic.best_config.basic")

    for full_id in train_componet_elem_list["dependent_ids"]:
        if full_id.startswith("train."):
            elem_id = full_id[len("train") + 1 :]
            if elem_id in params:
                param_value = params[elem_id]

                try:
                    component = manager.get_elem_by_id("train", elem_id)
                    fixed_value = fix_component_value(component, param_value)

                    if param_value is None:
                        fixed_value = get_default_value_for_component(component)

                    updates[full_id] = gr.update(value=fixed_value)

                except Exception as e:
                    print(f"Error processing component {full_id}: {e}")
                    updates[full_id] = gr.update(value=param_value)

    for full_id in basic_componet_elem_list["dependent_ids"]:
        if full_id.startswith("basic."):
            elem_id = full_id[len("basic") + 1 :]
            if elem_id in params:
                param_value = params[elem_id]

                try:
                    component = manager.get_elem_by_id("basic", elem_id)

                    fixed_value = fix_component_value(component, param_value)

                    if param_value is None:
                        fixed_value = get_default_value_for_component(component)

                    updates[full_id] = gr.update(value=fixed_value)

                except Exception as e:
                    print(f"Error processing component {full_id}: {e}")
                    updates[full_id] = gr.update(value=param_value)

    return updates


def get_default_value_for_component(component):
    """
    Return appropriate default value based on component type.

    Args:
        component (str): Component identifier or type

    Returns:
        Any: Default value corresponding to the component type
    """
    component_type = type(component).__name__

    if hasattr(component, 'value') and component.value is not None:
        return component.value

    if component_type in ['Textbox', 'TextArea']:
        return ""
    elif component_type in ['Number', 'Slider']:
        return 0
    elif component_type == 'Checkbox':
        return False
    elif component_type == 'Dropdown':
        if hasattr(component, 'multiselect') and component.multiselect:
            return []
        else:
            return None
    elif component_type == 'Radio':
        return None
    elif component_type == 'CheckboxGroup':
        return []
    else:
        return ""


def fix_component_value(component, value):
    """
    Validate and adjust component value to acceptable range.

    Args:
        component (str): Component identifier or type
        value (Any): Input value to be validated
    """

    component_type = type(component).__name__

    if component_type == "Number":
        if isinstance(value, str):
            try:
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                return float_val
            except (ValueError, TypeError):
                print(f"Warning: Cannot convert '{value}' to number, using 0")
                return 0
        elif isinstance(value, (int, float)):
            return value
        else:
            print(f"Warning: Unexpected type {type(value)} for Number component, using 0")
            return 0

    elif component_type == "Dropdown":
        if hasattr(component, "choices") and component.choices:
            choices = component.choices
            choices_str = [str(choice) for choice in choices]

            if set(choices_str) == {"True", "False"}:
                if isinstance(value, bool):
                    return str(value)
                elif isinstance(value, str):
                    if value.lower() in ["true", "1", "yes"]:
                        return "True"
                    elif value.lower() in ["false", "0", "no"]:
                        return "False"
                    return value
                elif isinstance(value, (int, float)):
                    return str(bool(value))

            elif all(isinstance(c, (int, float)) for c in choices):
                try:
                    return type(choices[0])(value)
                except (ValueError, TypeError):
                    return choices[0] if choices else value
            elif isinstance(value, list):
                return value
            else:
                return str(value)
        return value

    elif component_type == "Slider":
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0

    elif component_type == "Textbox":
        return str(value)

    elif component_type == "Checkbox":
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "on"]
        elif isinstance(value, (int, float)):
            return bool(value)
        elif isinstance(value, bool):
            return value
        else:
            return False

    return value


def setup_update_stage(manager):
    """
    Configure the update process lifecycle stages.

    Args:
        manager (object): Manager for update process coordination
    """
    best_config_elem = manager.get_elem_by_id("basic", "best_config")

    train_componet_elem_list = manager.get_dependencies("basic.best_config.train")
    basic_componet_elem_list = manager.get_dependencies("basic.best_config.basic")
    train_components = []
    basic_components = []

    for full_id in train_componet_elem_list["dependent_ids"]:
        if full_id.startswith("train."):
            elem_id = full_id[len("train") + 1 :]
            component = manager.get_elem_by_id("train", elem_id)
            train_components.append((full_id, component))

    for full_id in basic_componet_elem_list["dependent_ids"]:
        if full_id.startswith("basic."):
            elem_id = full_id[len("basic") + 1 :]
            component = manager.get_elem_by_id("basic", elem_id)
            basic_components.append((full_id, component))

    all_components = train_components + basic_components

    def on_component_value_change(value):
        manager._update_component_value("basic", "best_config", value)

        updates = model_update_callback(manager, value)

        output_list = []
        for full_id, component in all_components:
            if full_id in updates:
                output_list.append(updates[full_id])
            else:
                output_list.append(gr.update())

        return output_list

    best_config_elem.change(
        fn=on_component_value_change,
        inputs=[best_config_elem],
        outputs=[component for _, component in all_components],
    )


def setup_chatbot_response(manager):
    """
    Configure chatbot response logic, including event bindings for submit and stop buttons.

    Args:
        manager (object): Manager for chatbot UI and state

    """
    chat_input = manager.get_elem_by_id("chat", "chat_input")
    chatbot = manager.get_elem_by_id("chat", "chatbot")
    role_setting = manager.get_elem_by_id("chat", "role_setting")
    system_prompt = manager.get_elem_by_id("chat", "system_prompt")
    max_new_tokens = manager.get_elem_by_id("chat", "max_new_tokens")
    top_p = manager.get_elem_by_id("chat", "top_p")
    temperature = manager.get_elem_by_id("chat", "temperature")
    submit_btn = manager.get_elem_by_id("chat", "submit_btn")
    port = manager.get_elem_by_id("chat", "save_port")
    model_name = manager.get_elem_by_id("basic", "model_name")
    stop_btn = manager.get_elem_by_id("chat", "stop_btn")
    clear_btn = manager.get_elem_by_id("chat", "clear_btn")

    async def on_submit(message, history, role, system_prompt, max_new_tokens, top_p, temperature, port, model_name):
        update_config_yaml(manager, "chat_yaml_path", "chat")

        use_thought_model = chat_generator.check_thought_model(model_name=model_name)

        if use_thought_model:
            async for result in chat_generator.thought_response(
                message,
                history,
                role,
                system_prompt,
                max_new_tokens,
                top_p,
                temperature,
                port,
            ):
                yield result
        else:
            async for result in chat_generator.mm_response(
                message,
                history,
                role,
                system_prompt,
                max_new_tokens,
                top_p,
                temperature,
                port,
            ):
                yield result

    def on_stop():
        chat_generator.stop()

    def on_clear():
        return [], ""

    submit_btn.click(
        fn=on_submit,
        inputs=[
            chat_input,
            chatbot,
            role_setting,
            system_prompt,
            max_new_tokens,
            top_p,
            temperature,
            port,
            model_name,
        ],
        outputs=[
            chatbot,
            chat_input,
        ],
        api_name=None,
    )

    chat_input.submit(
        fn=on_submit,
        inputs=[
            chat_input,
            chatbot,
            role_setting,
            system_prompt,
            max_new_tokens,
            top_p,
            temperature,
            port,
            model_name,
        ],
        outputs=[
            chatbot,
            chat_input,
        ],
        api_name=None,
    )

    stop_btn.click(fn=on_stop, inputs=[], outputs=[])

    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[
            chatbot,
            chat_input,
        ],
    )


def chat_status_button_handler(manager, runner):
    """
    Handle chat status button interactions (e.g., start/stop/pause).

    Args:
        manager (object): Manager for chat UI components
        runner (object): Execution handler for chat operations
    """

    button = manager.get_elem_by_id("chat", "status_button")
    save_port = manager.get_elem_by_id("chat", "save_port")

    async def update_status(port_value):
        url = f"http://0.0.0.0:{port_value}/health"

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        success = alert.get(
                            "chat_check_load_model",
                            "tips",
                        )
                        status_message = f"{success}: " + f"http://0.0.0.0:{port_value}"
                        gr.Success(status_message)
                    else:
                        status_message = alert.get(
                            "chat_check_load_model",
                            "error",
                        ).format(port_value)
                        gr.Warning(f"{status_message} - Status: {response.status}")

        except asyncio.TimeoutError:
            error_message = alert.get("chat_check_load_model", "error").format(port_value)
            gr.Warning(f"{error_message} - Timeout")

        except Exception as e:
            error_message = alert.get("chat_check_load_model", "error").format(port_value)
            gr.Warning(f"{error_message} - Exception: {e!s}")

    button.click(fn=update_status, inputs=[save_port], outputs=[])


def chat_role_setting_system_prompt_handler(manager):
    """
    Handle system prompt configuration for chat role settings.

    Args:
        manager (object): Manager for chat role configuration
    """
    role_setting = manager.get_elem_by_id("chat", "role_setting")
    system_prompt = manager.get_elem_by_id("chat", "system_prompt")
    model_name = manager.get_elem_by_id("basic", "model_name")

    def update_role_setting(input):
        if config.is_thought_model(input):
            gr.Warning(alert.get("chat_role_setting", "warning"))

    def update_system_prompt(input):
        if config.is_thought_model(input):
            gr.Warning(alert.get("chat_system_prompt", "warning"))

    role_setting.change(
        fn=update_role_setting,
        inputs=[model_name],
        outputs=[],
    )

    system_prompt.change(
        fn=update_system_prompt,
        inputs=[model_name],
        outputs=[],
    )


def chat_update_max_new_len_max(manager):
    """
    Update the maximum value for the 'max_new_tokens' parameter.

    Args:
        manager (object): Manager for chat configuration settings
    """
    max_model_len = manager.get_elem_by_id("chat", "max_model_len")
    max_new_tokens = manager.get_elem_by_id("chat", "max_new_tokens")

    def update_max_new_tokens(input):
        return gr.update(maximum=input - 1)

    max_model_len.change(
        fn=update_max_new_tokens,
        inputs=[max_model_len],
        outputs=[max_new_tokens],
    )


def update_config_yaml(manager, execute_path, module, is_preview=False):
    """
    Update configuration in YAML file for specified module.

    Args:
        manager (object): Configuration manager
        execute_path (str): Path to execution directory
        module (str): Target module identifier
    """

    config_dict = manager.get_all_component_values()
    common.merge_dict_to_yaml(
        manager,
        config_dict,
        config.get_execute_yaml_path(execute_path),
        ["basic", module],
        [
            "preview_command_btn",
            "start_btn",
            "language",
            "stop_btn",
            "best_config",
            "command_preview",
            "output_text",
            "output_container",
            "clean_btn",
            "train_preview_btn",
            "start_merge_btn",
            "start_split_btn",
            "chat_input",
            "chatbot",
            "load_model_btn",
            "response_display",
            "status_button",
            "submit_btn",
            "thinking_display",
            "unload_model_btn",
            "role_setting",
            "system_prompt",
            "eval_preview_btn",
            "model_output_tab",
            "train_tab",
            "chat_tab",
            "clear_btn",
            "eval_tab",
            "start_split_btn",
            "start_merge_btn",
            "save_port",
            "export_tab",
            "gpu_num",
            "dataloader_parameters_tab",
            "distributed_parameters_tab",
            "optimizer_parameters_tab",
            "other_parameters_tab",
            "train_existed_dataset_path",
            "train_existed_dataset_prob",
            "train_customize_dataset_path",
            "train_customize_dataset_prob",
            "eval_existed_dataset_path",
            "eval_existed_dataset_prob",
            "eval_customize_dataset_path",
            "eval_customize_dataset_prob",
            "train_dataset_setting_tab",
            "eval_dataset_setting_tab",
            "eval_customize_preview_btn",
            "eval_existed_preview_btn",
            "train_customize_preview_btn",
            "train_existed_preview_btn",
            "stop_btn",
            "train_customize_select_dataset_type",
            "eval_customize_select_dataset_type",
            "output_dir_view",
            "prev_btn",
            "train_customize_dataset_type",
            "train_existed_dataset_type",
            "eval_customize_dataset_type",
            "eval_existed_dataset_type",
            "builtin_dataset_tab",
            "customize_dataset_tab",
            "eval_builtin_dataset_tab",
            "eval_customize_dataset_tab",
            "train_builtin_dataset_tab",
            "train_customize_dataset_tab",
            "model_source",
            "chat_info",
            "eval_dataset",
            "train_dataset",
            "model_name",
        ],
        is_preview,
    )


async def split_large_safetensors(folder_path, runner, max_shard_size=10.0):
    """
    Check .safetensors files in the specified folder and split any files
    exceeding max_shard_size (GB) using the erniekit split command.

    Args:
        folder_path (str): Path to the folder to check
        runner (object): Execution handler for splitting operations
        max_shard_size (float): Maximum file size threshold in GB (default: 10GB)

    Returns:
        dict: Processing results in the format {filename: processing status}
    """
    max_size_bytes = max_shard_size * 1024**3
    results = {}

    if not os.path.exists(folder_path):
        error_msg_non_existent = alert.get("export_split_non_existent", "error")
        print("❌ " + error_msg_non_existent)
        gr.Warning(error_msg_non_existent)
        yield "❌ " + error_msg_non_existent
        return

    safetensors_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".safetensors") and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not safetensors_files:
        export_split_success = alert.get("export_split_success", "info")
        print("✅ " + export_split_success)
        yield results
        return

    for filename in safetensors_files:
        file_path = os.path.join(folder_path, filename)
        file_size = os.path.getsize(file_path)

        if file_size > max_size_bytes:
            file_size_gb = file_size / (1024**3)

            split_text = alert.get("export_split_find_exceed_file", "info").format(
                filename, file_size_gb, max_shard_size
            )
            print(split_text)
            yield split_text

            try:
                command = config.get_execute_command("split")
                async for output in execute_command(runner, command):
                    yield output

                export_split_success = alert.get("export_split_success", "info")
                print("✅ " + export_split_success)
                gr.Info(export_split_success)
                yield "✅ " + export_split_success
            except subprocess.CalledProcessError as e:
                export_split_fail = alert.get("export_split_fail", "error")
                print("❌ " + export_split_fail)
                gr.Warning(export_split_fail)
                yield "❌ " + export_split_fail
                yield e.output


def load_update_dataset_config(manager, module, dataset_id, path_id, prob_id, type_id):
    """
    Load and update dataset configuration with specified parameters.

    Args:
        manager (object): Configuration manager
        module (str): Target module identifier
        dataset_id (str): Unique identifier for the dataset
        path_id (str): Identifier for dataset path configuration
        prob_id (str): Identifier for probability distribution configuration
        type_id (str): Identifier for dataset type configuration

    Returns:
        dict: Updated dataset configuration
    """

    dataset = manager.get_elem_by_id(module, dataset_id)
    path_elem = manager.get_elem_by_id(module, path_id)
    prob_elem = manager.get_elem_by_id(module, prob_id)
    type_elem = manager.get_elem_by_id(module, type_id)

    def update_path_prob(dataset_names):
        paths = []
        probs = []
        types = []

        for dataset_name in dataset_names:
            info = config.get_dataset_info_kwagrs(dataset_name)
            if info is not None:

                path = info.get("path")
                prob = info.get("prob")
                type = info.get("type")

                if path is None:
                    path_error_msg = alert.get("preview_data_non_path", "warning").format(dataset_name)
                    print(path_error_msg)
                    gr.Warning(path_error_msg)
                    continue
                if prob is None:
                    prob_error_msg = alert.get("preview_data_non_prob", "warning").format(dataset_name)
                    gr.Warning(prob_error_msg)
                    print(prob_error_msg)
                    continue
                if type is None:
                    type_error_msg = alert.get("preview_data_non_type", "warning").format(dataset_name)
                    gr.Warning(type_error_msg)
                    print(type_error_msg)
                    continue

                paths.append(path)
                probs.append(prob)
                types.append(type)

        path_str = ", ".join(paths)
        type_str = ", ".join(types)
        prob_str = ", ".join([str(p) for p in probs])

        return gr.update(value=path_str), gr.update(value=prob_str), gr.update(value=type_str)

    manager.demo.load(fn=update_path_prob, inputs=dataset, outputs=[path_elem, prob_elem, type_elem])
