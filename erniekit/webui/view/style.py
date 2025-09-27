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

Customize CSS styles to achieve the floating effect of the modal box.

"""

CSS = """

.hide-copy-code .hide-top-corner > button {
    display: none !important;
}

.large-tabs .tab-container,
.large-tabs .tab-wrapper {
    height: var(--size-16) !important;
}
.large-tabs .tab-container button{
    font-size: 20px !important;
    font-weight: 600 !important;
}

.basic-info-row-accordion,
.basic-info-row .form {
    border-color: #d3d3d3 !important;
}

:root {
      --primary-500: #87a9ff!important;
      --primary-100: #87a9ff!important;
      --primary-200: #87a9ff!important;
      --primary-300: #87a9ff!important;
      --primary-400: #87a9ff!important;
      --primary-500: #87a9ff!important;
      --primary-600: #87a9ff!important;
      --primary-700: #87a9ff!important;
      --primary-800: #87a9ff!important;
      --primary-900: #87a9ff!important;
      --primary-950: #87a9ff!important;
      --border-color-accent-subdued: #fff7ed!important;
}

/* 居中按钮 */
.center-button .gr-button {
    margin: 0 auto;  /* 水平居中 */
    display: block;  /* 确保 margin 生效 */
}

/* 固定高度的命令输出框 */
.export-height-output textarea {
    height: 350px !important;
    max-height: 350px !important;
    overflow-y: auto !important;
    resize: none !important;
    font-family: monospace !important;
    white-space: pre !important;
    font-size: 14px !important;
}


.dataset-text-height-output textarea {
    /* 固定显示2行文本的高度 */
    height: 4em !important; /* 1em约等于1行文字高度，3em大致等于2行（含行间距） */
    max-height: 4em !important; /* 防止内容撑开高度 */

    /* 超出内容处理 */
    overflow: auto !important; /* 超出时显示滚动条 */
    resize: none !important; /* 禁止用户调整大小 */

    /* 其他样式 */
    font-family: monospace !important;
    white-space: pre-wrap !important; /* 保留换行符，自动换行 */
    font-size: 14px !important;
    line-height: 1.5 !important; /* 行高 */
}


.chat-height-output textarea {
    /* 固定显示2行文本的高度 */
    height: 14em !important; /* 1em约等于1行文字高度，3em大致等于2行（含行间距） */
    max-height: 14em !important; /* 防止内容撑开高度 */

    /* 超出内容处理 */
    overflow: auto !important; /* 超出时显示滚动条 */
    resize: none !important; /* 禁止用户调整大小 */

    /* 其他样式 */
    font-family: monospace !important;
    white-space: pre-wrap !important; /* 保留换行符，自动换行 */
    font-size: 14px !important;
    line-height: 1.5 !important; /* 行高 */
}

.general-height-output textarea {
    /* 固定显示2行文本的高度 */
    height: 30em !important; /* 1em约等于1行文字高度，3em大致等于2行（含行间距） */
    max-height: 30em !important; /* 防止内容撑开高度 */

    /* 超出内容处理 */
    overflow: auto !important; /* 超出时显示滚动条 */
    resize: none !important; /* 禁止用户调整大小 */

    /* 其他样式 */
    font-family: monospace !important;
    white-space: pre-wrap !important; /* 保留换行符，自动换行 */
    font-size: 14px !important;
    line-height: 1.5 !important; /* 行高 */
}

/* 模态框容器样式 */
.modal-box {
    position: fixed!important;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    z-index: 1000; /* 确保模态框显示在最上层 */
    width: 800px!important; /* 设置固定宽度 */
    max-height: 80vh; /* 设置最大高度为视窗高度的80% */
    overflow-y: auto; /* 内容超出时显示滚动条 */
}

.page-info {
    overflow: hidden!important;
    text-align: right;
    margin: 10px 0;
    color: #666;
    padding-top: 8px; /* 为了和标题垂直对齐 */
}

.pagination-controls {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 15px 0;
}

/* 遮罩层样式 */
.modal-overlay {
    position: fixed!important;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 999; /* 遮罩层在模态框下方 */
}

"""

html_log = """
<div style="display: flex; justify-content: center; width: 100%; margin: 0; padding: 0;">
    <img src="data:image/png;base64,{}"
         style="width: 520px; height: auto; border: none; box-shadow: none;">
</div>
"""
