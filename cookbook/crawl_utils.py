# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CrawlUtils is a class that provides utility methods for web crawling and processing.
"""

import logging
import re

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)


class CrawlUtils:
    """
    Provides web crawling and content extraction utilities with intelligent filtering.
    Features include asynchronous crawling, content pruning, markdown generation,
    and configurable link/media filtering. Uses crawl4ai library for core functionality.
    """

    def __init__(self):
        """Initialize the CrawlUtils instance."""
        self.logger = logging.getLogger(__name__)

        # Configure content filter - uses pruning algorithm to filter page content
        content_filter = PruningContentFilter(threshold=0.48, threshold_type="fixed")
        # Configure markdown generator, apply the above content filter to generate "fit_markdown"
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        # Configure crawler run parameters
        self.run_config = CrawlerRunConfig(
            # 20 seconds page timeout
            page_timeout=20000,
            # Filtering
            word_count_threshold=10,
            excluded_tags=[
                "nav",
                "footer",
                "aside",
                "header",
                "script",
                "style",
                "iframe",
                "meta",
            ],
            exclude_external_links=True,
            exclude_internal_links=True,
            exclude_social_media_links=True,
            exclude_external_images=True,
            only_text=True,
            # Markdown generation
            markdown_generator=md_generator,
            # Cache
            cache_mode=CacheMode.BYPASS,
        )

    async def get_webpage_text(self, url: str) -> str:
        """
        Asynchronously fetches and cleans webpage content from given URL using configured crawler.
        Applies content filtering, markdown conversion, and text cleaning (removing undefined,
        excess whitespace, tabs). Returns None if error occurs.

        Args:
            url (str): The URL to retrieve the text from.

        Returns:
            str: The plain text retrieved from the specified URL.
        """
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=self.run_config)

            webpage_text = result.markdown.fit_markdown
            self.logger.info(f"Webpage Text: \n{webpage_text}")

            # Clean up the text
            cleaned_text = webpage_text.replace("undefined", "")
            cleaned_text = re.sub(r"(\n\s*){3,}", "\n\n", cleaned_text)
            cleaned_text = re.sub(r"[\r\t]", "", cleaned_text)
            cleaned_text = re.sub(r" +", " ", cleaned_text)
            cleaned_text = re.sub(r"^\s+|\s+$", "", cleaned_text, flags=re.MULTILINE)
            return cleaned_text.strip()

        except Exception as e:
            self.logger.info(f"Error: {e}")
            return None
