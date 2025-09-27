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
Unified handling of Gradio components
"""

import gradio as gr
import lang as la


class Manager:

    def __init__(self):
        self._id_to_elem = {}
        self.specific_id_to_elem = {}
        self.specific_component_value = {}
        self._locale_data = la.LOCALES
        self._current_lang = "zh"
        self._component_values = {}
        self._debug = False
        self._dependencies = {}
        self.demo = None

    def add_specific_elem_by_id(self, module_id, elem_id, elem, initial_value=None):
        """
        Add a specific element to the module by IDs with optional initial value.

        Args:
            self: Instance reference
            module_id (str): Identifier for the target module
            elem_id (str): Unique identifier for the element
            elem (Any): Element to be added
            initial_value (Any, optional): Initial value for the element (default: None)
        """
        full_id = f"{module_id}.{elem_id}"
        self.specific_id_to_elem[full_id] = elem
        self.specific_component_value[full_id] = initial_value

    def get_specific_elem_by_id(self, module_id, elem_id):
        """
        Retrieve a specific element by its module and element IDs.

        Args:
            self: Instance reference
            module_id (str): Identifier for the target module
            elem_id (str): Unique identifier for the element

        Returns:
            Any: Element associated with the given IDs, or None if not found
        """

        full_id = f"{module_id}.{elem_id}"
        return self.specific_id_to_elem.get(full_id)

    def set_specific_component_value(self, module_id, elem_id, value):
        """
        Update the value of a specific component within a module.

        Args:
            self: Instance reference
            module_id (str): Identifier for the target module
            elem_id (str): Unique identifier for the component
            value (Any): New value to be set
        """
        full_id = f"{module_id}.{elem_id}"
        self.specific_component_value[full_id] = value

    def get_specific_component_value(self, module_id, elem_id):
        """
        Retrieve the current value of a specific component.

        Args:
            self: Instance reference
            module_id (str): Identifier for the target module
            elem_id (str): Unique identifier for the component
        """
        full_id = f"{module_id}.{elem_id}"
        if full_id in self.specific_component_value:
            return self.specific_component_value[module_id].get(elem_id)
        return None

    def get_all_specific_component_values(self):
        """
        Retrieve all component values organized by module.

        Returns:
            dict: Nested dictionary structured as {"specific_" + module_id: {elem_id: value}}
        """
        result = {}
        for full_id, value in self.specific_component_value.items():
            module_id, elem_id = full_id.split(".", 1)  # 分割完整ID为模块和元素ID
            specific_key = f"specific_{module_id}"
            if specific_key not in result:
                result[specific_key] = {}
            result[specific_key][elem_id] = value
        return result

    def add_elem(self, module_id, elem_id, elem, initial_value=None):
        """
        Register a component with initial value and organize by module.

        Args:
            self: Instance reference
            module_id (str): Identifier for the module
            elem_id (str): Unique identifier for the component
            elem (Any): Component object to register
            initial_value (Any, optional): Initial value for the component (default: None)
        """
        full_id = f"{module_id}.{elem_id}"
        self._id_to_elem[full_id] = elem

        if module_id not in self._component_values:
            self._component_values[module_id] = {}

        self._component_values[module_id][elem_id] = initial_value

        if self._debug:
            print(f"[Manager] 注册组件: {full_id} ({type(elem).__name__}), 初始值: {initial_value}")

    def get_elem_by_id(self, module_id, elem_id):
        """
        Retrieve a component by its module and element IDs.

        Args:
            self: Instance reference
            module_id (str): Identifier for the module
            elem_id (str): Unique identifier for the component
        """
        full_id = f"{module_id}.{elem_id}"
        return self._id_to_elem.get(full_id)

    def get_component_value(self, module_id, elem_id):
        """
        Retrieve the current value of a component.

        Args:
            self: Instance reference
            module_id (str): Identifier for the module
            elem_id (str): Unique identifier for the component

        Returns:
            Any: Current value of the component if found, None otherwise
        """
        if module_id in self._component_values:
            return self._component_values[module_id].get(elem_id)
        return None

    def get_module_values(self, module_id):
        """
        Retrieve all component values within a specified module.

        Args:
            self: Instance reference
            module_id (str): Identifier for the module

        Returns:
            dict: Dictionary of component values (empty if module not found)
        """
        return self._component_values.get(module_id, {})

    def get_all_component_values(self):
        """
        Retrieve all component values across all modules.

        Returns:
            dict: Nested dictionary of all components structured as {module_id: {elem_id: value}}
        """
        return self._component_values

    def change_lang(self, lang):
        """
        Switch the application language and update UI components.

        Args:
            self: Instance reference
            lang (str): Language code (e.g., 'en', 'zh')
        """
        if lang not in ["zh", "en"]:
            return {}

        updates = {}
        for full_id, elem in self._id_to_elem.items():
            parts = full_id.split(".")
            elem_name = parts[-1]

            if elem_name not in self._locale_data:
                continue

            lang_config = self._locale_data[elem_name].get(lang, {})
            if not lang_config:
                continue

            update_kwargs = {}

            if isinstance(elem, gr.Button):
                if "value" in lang_config:
                    update_kwargs["value"] = lang_config["value"]
            elif isinstance(elem, gr.Markdown):
                if "value" in lang_config:
                    update_kwargs["value"] = lang_config["value"]
            elif isinstance(elem, gr.Tab):
                if "label" in lang_config:
                    update_kwargs["label"] = lang_config["label"]
            elif isinstance(elem, gr.HTML):
                if "value" in lang_config:
                    update_kwargs["value"] = lang_config["value"]
            else:
                if "label" in lang_config:
                    update_kwargs["label"] = lang_config["label"]
                if "info" in lang_config:
                    update_kwargs["info"] = lang_config["info"]
                if "placeholder" in lang_config:
                    update_kwargs["placeholder"] = lang_config["placeholder"]

            if update_kwargs:
                updates[elem] = gr.update(**update_kwargs)

        return updates

    def setup_language_switching(self, language, demo, alert):
        """
        Configure language switching event handlers and initial state.

        Args:
            self: Instance reference
            language (str): Initial language code
            demo (object): Demo component to update
            alert (object): Alert component for notifications

        """
        all_components = list(self._id_to_elem.values())
        input_components = [
            comp
            for comp in all_components
            if isinstance(
                comp,
                (
                    gr.Textbox,
                    gr.Dropdown,
                    gr.Slider,
                    gr.Checkbox,
                    gr.CheckboxGroup,
                    gr.Radio,
                    gr.Chatbot,
                    gr.Button,
                    gr.HTML,
                ),
            )
        ]

        if self._debug:
            print(f"[语言切换初始化] 总组件数: {len(all_components)}")
            print(f"[语言切换初始化] 输入组件数: {len(input_components)}")

        def update_fn(lang, *values):
            for comp in input_components:
                for full_id, elem in self._id_to_elem.items():
                    if elem == comp:
                        parts = full_id.split(".")
                        if len(parts) >= 2:
                            module_id, elem_id = parts[0], ".".join(parts[1:])
                            if isinstance(comp, gr.Chatbot):
                                if values and input_components.index(comp) < len(values):
                                    self._component_values[module_id][elem_id] = values[input_components.index(comp)]
                            else:
                                if values and input_components.index(comp) < len(values):
                                    self._component_values[module_id][elem_id] = values[input_components.index(comp)]
                        break
            updates = self.change_lang(lang)
            return [updates.get(comp, comp) for comp in all_components]

        language.change(fn=update_fn, inputs=[language] + input_components, outputs=all_components)

        if self.demo:
            initial_values = []
            for comp in input_components:
                for full_id, elem in self._id_to_elem.items():
                    if elem == comp:
                        parts = full_id.split(".")
                        if len(parts) >= 2:
                            module_id, elem_id = parts[0], ".".join(parts[1:])
                            initial_values.append(self._component_values[module_id].get(elem_id, None))
                        break

            demo.load(
                fn=lambda: update_fn(self._current_lang, *initial_values),
                outputs=all_components,
            )

    def add_dependency(
        self,
        source_module_id,
        source_elem_id,
        dependent_module_ids,
        dependent_elem_ids,
        update_callback,
    ):
        """
        Register a dependency between a source component and multiple dependent components.

        Args:
            self: Instance reference
            source_module_id (str): Module ID of the source component
            source_elem_id (str): ID of the source component
            dependent_module_ids (list): List of module IDs containing dependent components
            dependent_elem_ids (list): List of dependent component IDs (parallel to module IDs)
            update_callback (callable): Function to compute updated values for dependents,
                                        takes source value and returns list of dicts

        Returns:
            None
        """
        source_full_id = f"{source_module_id}.{source_elem_id}"
        dependent_full_ids = [
            f"{mod_id}.{elem_id}" for mod_id, elem_id in zip(dependent_module_ids, dependent_elem_ids)
        ]

        self._dependencies[source_full_id] = {
            "dependent_ids": dependent_full_ids,
            "callback": update_callback,
        }

        if self._debug:
            print(f"[Manager] 注册依赖关系: {source_full_id} -> {dependent_full_ids}")

    def add_module_dependency(
        self,
        source_module_id,
        source_elem_id,
        update_module_id,
        update_callback,
        exclude_components=None,
    ):
        """
        Add a module-level dependency to trigger updates for an entire module
        when a source component's value changes.

        Args:
            self: Instance reference
            source_module_id (str): Module ID of the source component
            source_elem_id (str): ID of the source component
            update_module_id (str): Module ID to update when source changes
            update_callback (callable): Function to generate update values,
                                        takes source value and returns dict
            exclude_components (list, optional): List of component IDs (without module prefix)
                                                 to exclude from updates (default: None)
        """
        source_full_id = f"{source_module_id}.{source_elem_id}.{update_module_id}"

        module_components = []
        exclude_components = exclude_components or []

        for full_id in self._id_to_elem.keys():
            if full_id.startswith(f"{update_module_id}."):
                elem_id = full_id[len(update_module_id) + 1 :]

                if elem_id not in exclude_components:
                    module_components.append(full_id)

        self._dependencies[source_full_id] = {
            "dependent_ids": module_components,
            "callback": update_callback,
        }

    def get_dependencies(self, source_id):
        """
        Retrieve all dependencies registered for a given source component.

        Args:
            self: Instance reference
            source_id (str): Identifier of the source component

        Returns:
            list: List of dependency configurations for the source component
        """
        return self._dependencies[source_id]

    def setup_component_tracking(self, demo):
        """
        Configure value tracking and initialize default values for all components.

        Args:
            self: Instance reference
            demo (object): Demo environment containing components to track
        """
        for full_id, elem in self._id_to_elem.items():
            parts = full_id.split(".")
            if len(parts) < 2:
                continue

            module_id, elem_id = parts[0], ".".join(parts[1:])

            if isinstance(
                elem,
                (gr.Textbox, gr.Dropdown, gr.Slider, gr.Checkbox, gr.CheckboxGroup, gr.Radio, gr.Number, gr.HTML),
            ):
                elem.change(
                    fn=lambda value, mid=module_id, eid=elem_id: self._update_component_value(mid, eid, value),
                    inputs=[elem],
                    outputs=[],
                )

                initial_value = self._component_values[module_id].get(elem_id)
                if initial_value is not None:
                    if isinstance(elem, gr.Textbox):
                        elem.value = initial_value
                    elif isinstance(elem, gr.Dropdown):
                        elem.value = initial_value
                    elif isinstance(elem, gr.Slider):
                        elem.value = initial_value
                    elif isinstance(elem, gr.Checkbox):
                        elem.value = initial_value
                    elif isinstance(elem, gr.CheckboxGroup):
                        elem.value = initial_value
                    elif isinstance(elem, gr.Radio):
                        elem.value = initial_value
                    elif isinstance(elem, gr.Number):
                        elem.value = initial_value
                    elif isinstance(elem, gr.HTML):
                        elem.value = initial_value

    def _update_component_value(self, module_id, elem_id, value):
        """
        Update a component's value and log debug information.

        Args:
            self: Instance reference
            module_id (str): Identifier for the module
            elem_id (str): Unique identifier for the component
            value (Any): New value to set
        """
        if module_id in self._component_values and elem_id in self._component_values[module_id]:
            old_value = self._component_values[module_id][elem_id]
            self._component_values[module_id][elem_id] = value
            if self._debug and old_value != value:
                print(f"[Manager] Value updated: {module_id}.{elem_id} = {old_value} → {value}")
        else:
            print(f"[Manager] Error: Unregistered component {module_id}.{elem_id}")

    def setup_dropdown(self, module_id, dropdown_id):
        """
        Configure a dropdown component with dynamic options and event handling.

        Args:
            self: Instance reference
            module_id (str): Identifier for the module containing the dropdown
            dropdown_id (str): Unique identifier for the dropdown component
        """
        full_id = f"{module_id}.{dropdown_id}"
        if full_id not in self._dependencies:
            print(f"[Manager] Warning: {full_id} has no registered dependencies")
            return

        source_elem = self.get_elem_by_id(module_id, dropdown_id)
        if not source_elem:
            print(f"[Manager] Error: Component {full_id} not found")
            return

        dependency_info = self._dependencies[full_id]
        dependent_ids = dependency_info["dependent_ids"]
        update_callback = dependency_info["callback"]

        all_components = [self.get_elem_by_id(*id.split(".", 1)) for id in [full_id] + dependent_ids]
        all_components = [c for c in all_components if c is not None]

        def dropdown_change_handler(selected_value):
            self._component_values[module_id][dropdown_id] = selected_value
            updates = update_callback(selected_value)
            output_updates = [updates.get(comp, comp) for comp in all_components]
            return output_updates

        source_elem.change(fn=dropdown_change_handler, inputs=[source_elem], outputs=all_components)

        if self.demo:
            initial_value = self._component_values[module_id].get(dropdown_id)
            self.demo.load(
                fn=lambda: dropdown_change_handler(initial_value),
                outputs=all_components,
            )


manager = Manager()
