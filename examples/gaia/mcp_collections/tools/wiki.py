"""
Wikipedia MCP Server

This module provides MCP server functionality for interacting with Wikipedia.
It supports searching Wikipedia, retrieving article content, and getting summaries.

Key features:
- Search Wikipedia for articles
- Retrieve full article content
- Get article summaries
- Fetch random articles
- Get article categories and links
- Access historical versions of articles

Main functions:
- mcp_search_wikipedia: Searches Wikipedia for articles matching a query
- mcp_get_article_content: Retrieves the full content of a Wikipedia article
- mcp_get_article_summary: Gets a summary of a Wikipedia article
- mcp_get_article_categories: Gets categories for a Wikipedia article
- mcp_get_article_links: Gets links from a Wikipedia article
- mcp_get_article_history: Gets historical version of a Wikipedia page closest to the specified date
- mcp_get_wikipedia_capabilities: Gets information about Wikipedia service capabilities
"""

import calendar
import json
import os
import time
import traceback
from datetime import datetime

import requests
import wikipedia
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from aworld.logs.util import Color
from examples.gaia.mcp_collections.base import ActionArguments, ActionCollection, ActionResponse


class WikipediaSearchResult(BaseModel):
    """Model representing a Wikipedia search result."""

    title: str
    snippet: str | None = None
    url: str | None = None


class WikipediaArticle(BaseModel):
    """Model representing a Wikipedia article."""

    title: str
    pageid: int | None = None
    url: str
    content: str
    summary: str
    images: list[str] | None = None
    categories: list[str] | None = None
    links: list[str] | None = None
    references: list[str] | None = None
    sections: list[dict[str, str]] | None = None
    # History-specific fields
    original_query: str | None = None
    requested_date: str | None = None
    actual_date: str | None = None
    is_exact_date: bool | None = None
    is_redirect: bool | None = None
    editor: str | None = None
    edit_comment: str | None = None


class WikipediaMetadata(BaseModel):
    """Metadata for Wikipedia operation results."""

    query: str
    language: str
    count: int
    operation_type: str
    execution_time: float | None = None
    error_type: str | None = None
    article_id: int | None = None
    is_redirect: bool | None = None
    requested_date: str | None = None
    actual_date: str | None = None


class WikipediaCollection(ActionCollection):
    """MCP service for Wikipedia information retrieval.

    Provides Wikipedia interaction capabilities including:
    - Article search
    - Content retrieval
    - Summary generation
    - Random article fetching
    - Category and link extraction
    - Historical version access
    - LLM-friendly result formatting
    """

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Initialize configuration
        self.default_language = "en"
        self.max_search_results = 20
        self.max_random_articles = 10
        self.default_summary_sentences = 5

        wikipedia.set_lang(self.default_language)

        self._color_log("Wikipedia service initialized", Color.green, "debug")

    def _format_search_results(self, results: list[WikipediaSearchResult], output_format: str = "markdown") -> str:
        """Format search results for LLM consumption.

        Args:
            results: List of search results
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps([result.model_dump() for result in results], indent=2)

        elif output_format == "text":
            if not results:
                return "No results found."

            output_parts = [f"Found {len(results)} results:"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"{i}. {result.title}")
                if result.snippet:
                    output_parts.append(f"   {result.snippet}")
                if result.url:
                    output_parts.append(f"   URL: {result.url}")

            return "\n".join(output_parts)

        else:  # markdown (default)
            if not results:
                return "No Wikipedia search results found."

            output_parts = [f"# Wikipedia Search Results\n\nFound {len(results)} results:\n"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"## {i}. [{result.title}]({result.url})")
                if result.snippet:
                    output_parts.append(f"{result.snippet}\n")

            return "\n".join(output_parts)

    def _format_article(
        self, article: WikipediaArticle, output_format: str = "markdown", include_full_content: bool = False
    ) -> str:
        """Format article for LLM consumption.

        Args:
            article: Wikipedia article
            output_format: Format type ('markdown', 'json', 'text')
            include_full_content: Whether to include the full article content

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps(article.model_dump(), indent=2)

        elif output_format == "text":
            output_parts = [f"Title: {article.title}"]

            if article.url:
                output_parts.append(f"URL: {article.url}")

            if article.summary:
                output_parts.append(f"\nSummary:\n{article.summary}")

            if include_full_content and article.content:
                output_parts.append(f"\nContent:\n{article.content}")

            if article.categories:
                output_parts.append(f"\nCategories: {', '.join(article.categories)}")

            if article.requested_date:
                output_parts.append("\nHistorical Version:")
                output_parts.append(f"Requested Date: {article.requested_date}")
                output_parts.append(f"Actual Date: {article.actual_date}")

            if article.links and len(article.links) > 0:
                output_parts.append(f"\nRelated Links: {', '.join(article.links[:10])}")
                if len(article.links) > 10:
                    output_parts.append(f"... and {len(article.links) - 10} more links")

            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [f"# {article.title}"]

            if article.url:
                output_parts.append(f"**Wikipedia:** [{article.title}]({article.url})")

            if article.summary:
                output_parts.append(f"\n## Summary\n{article.summary}")

            if include_full_content and article.content:
                output_parts.append(f"\n## Content\n{article.content}")

            if article.categories and len(article.categories) > 0:
                output_parts.append(f"\n## Categories\n{', '.join(article.categories)}")

            if article.links and len(article.links) > 0:
                output_parts.append("\n## Related Links\n")
                # Limit to first 20 links to avoid overwhelming output
                for i, link in enumerate(article.links[:20], 1):
                    output_parts.append(f"{i}. {link}")
                if len(article.links) > 20:
                    output_parts.append(f"\n... and {len(article.links) - 20} more links")

            if article.requested_date:
                output_parts.append("\n## Historical Version Information")
                output_parts.append(f"**Requested Date:** {article.requested_date}")
                output_parts.append(f"**Actual Date:** {article.actual_date}")
                if article.editor:
                    output_parts.append(f"**Editor:** {article.editor}")
                if article.edit_comment:
                    output_parts.append(f"**Edit Comment:** {article.edit_comment}")

            return "\n".join(output_parts)

    def mcp_search_wikipedia(
        self,
        query: str = Field(..., description="The search query string"),
        limit: int = Field(10, description="Maximum number of results to return"),
        language: str = Field("en", description="Language code for Wikipedia (e.g., 'en', 'es', 'fr')"),
        output_format: str = Field("markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Search Wikipedia for articles matching the query.

        This tool provides Wikipedia search capabilities with:
        - Configurable result limits
        - Multi-language support
        - LLM-optimized result formatting
        - Error handling

        Args:
            query: Search query string
            limit: Maximum number of results to return
            language: Language code for Wikipedia
            output_format: Format for the response output

        Returns:
            ActionResponse with search results and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(query, FieldInfo):
                query = query.default
            if isinstance(limit, FieldInfo):
                limit = limit.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default

            # Validate parameters
            if not query:
                return ActionResponse(
                    success=False,
                    message="Search query cannot be empty",
                    metadata=WikipediaMetadata(
                        query=query,
                        language=language,
                        count=0,
                        operation_type="search",
                        error_type="invalid_parameters",
                    ).model_dump(),
                )

            # Limit the number of results to prevent excessive API calls
            if limit > self.max_search_results:
                limit = self.max_search_results

            self._color_log(f"🔍 Searching Wikipedia for: {query} (language: {language})", Color.cyan)

            # Search Wikipedia
            search_results = wikipedia.search(query, results=limit)

            # Format results
            formatted_results = []
            for title in search_results:
                try:
                    # Get a summary to use as a snippet
                    summary = wikipedia.summary(title, sentences=1, auto_suggest=False)
                    # Create URL
                    url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"

                    result = WikipediaSearchResult(title=title, snippet=summary, url=url)
                    formatted_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error getting details for '{title}': {str(e)}")
                    # Still include the result, but without a snippet
                    url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    result = WikipediaSearchResult(title=title, url=url)
                    formatted_results.append(result)

            # Format output for LLM
            formatted_output = self._format_search_results(formatted_results, output_format)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create metadata
            metadata = WikipediaMetadata(
                query=query,
                language=language,
                count=len(formatted_results),
                operation_type="search",
                execution_time=execution_time,
            )

            self._color_log(f"✅ Found {len(formatted_results)} results for query: {query}", Color.green)

            return ActionResponse(
                success=True,
                message=formatted_output,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to search Wikipedia: {str(e)}"
            self.logger.error(f"Wikipedia search error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=query,
                    language=language,
                    count=0,
                    operation_type="search",
                    error_type="search_error",
                ).model_dump(),
            )

    def mcp_get_article_content(
        self,
        title: str = Field(..., description="Title of the Wikipedia article"),
        auto_suggest: bool = Field(False, description="Whether to use Wikipedia's auto-suggest feature"),
        redirect: bool = Field(True, description="Whether to follow redirects"),
        language: str = Field("en", description="Language code for Wikipedia (e.g., 'en', 'es', 'fr')"),
        output_format: str = Field("markdown", description="Output format: 'markdown', 'json', or 'text'"),
        include_full_content: bool = Field(True, description="Whether to include the full article content"),
    ) -> ActionResponse:
        """Retrieve the full content of a Wikipedia article.

        This tool provides Wikipedia article retrieval with:
        - Auto-suggestion support
        - Redirect handling
        - Multi-language support
        - LLM-optimized result formatting
        - Error handling

        Args:
            title: Title of the Wikipedia article
            auto_suggest: Whether to use Wikipedia's auto-suggest feature
            redirect: Whether to follow redirects
            language: Language code for Wikipedia
            output_format: Format for the response output
            include_full_content: Whether to include the full article content

        Returns:
            ActionResponse with article content and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(title, FieldInfo):
                title = title.default
            if isinstance(auto_suggest, FieldInfo):
                auto_suggest = auto_suggest.default
            if isinstance(redirect, FieldInfo):
                redirect = redirect.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default
            if isinstance(include_full_content, FieldInfo):
                include_full_content = include_full_content.default

            # Validate parameters
            if not title:
                return ActionResponse(
                    success=False,
                    message="Article title cannot be empty",
                    metadata=WikipediaMetadata(
                        query=title,
                        language=language,
                        count=0,
                        operation_type="content_retrieval",
                        error_type="invalid_parameters",
                    ).model_dump(),
                )

            self._color_log(f"📖 Retrieving Wikipedia article: {title} (language: {language})", Color.cyan)

            # Get the page
            page = wikipedia.page(title, auto_suggest=auto_suggest, redirect=redirect)

            # Create article object
            article = WikipediaArticle(
                title=page.title,
                pageid=page.pageid,
                url=page.url,
                content=page.content,
                summary=page.summary,
                images=page.images,
                categories=page.categories,
                links=page.links,
                references=page.references,
                sections=[{"title": section, "content": page.section(section)} for section in page.sections],
            )

            # Format output for LLM
            formatted_output = self._format_article(article, output_format, include_full_content)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create metadata
            metadata = WikipediaMetadata(
                query=title,
                language=language,
                count=1,
                operation_type="content_retrieval",
                execution_time=execution_time,
                article_id=page.pageid,
                is_redirect=page.title != title,
            )

            self._color_log(f"✅ Retrieved article: {page.title}", Color.green)

            return ActionResponse(
                success=True,
                message=formatted_output,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to retrieve Wikipedia article: {str(e)}"
            self.logger.error(f"Wikipedia content retrieval error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=title,
                    language=language,
                    count=0,
                    operation_type="content_retrieval",
                    error_type="content_retrieval_error",
                ).model_dump(),
            )

    def mcp_get_article_summary(
        self,
        title: str = Field(..., description="Title of the Wikipedia article"),
        sentences: int = Field(5, description="Number of sentences to return in the summary"),
        auto_suggest: bool = Field(False, description="Whether to use Wikipedia's auto-suggest feature"),
        redirect: bool = Field(True, description="Whether to follow redirects"),
        language: str = Field("en", description="Language code for Wikipedia (e.g., 'en', 'es', 'fr')"),
        output_format: str = Field("markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Get a summary of a Wikipedia article.

        This tool provides Wikipedia article summary retrieval with:
        - Configurable summary length
        - Auto-suggestion support
        - Redirect handling
        - Multi-language support
        - LLM-optimized result formatting
        - Error handling

        Args:
            title: Title of the Wikipedia article
            sentences: Number of sentences to return in the summary
            auto_suggest: Whether to use Wikipedia's auto-suggest feature
            redirect: Whether to follow redirects
            language: Language code for Wikipedia
            output_format: Format for the response output

        Returns:
            ActionResponse with article summary and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(title, FieldInfo):
                title = title.default
            if isinstance(sentences, FieldInfo):
                sentences = sentences.default
            if isinstance(auto_suggest, FieldInfo):
                auto_suggest = auto_suggest.default
            if isinstance(redirect, FieldInfo):
                redirect = redirect.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default

            # Validate parameters
            if not title:
                return ActionResponse(
                    success=False,
                    message="Article title cannot be empty",
                    metadata=WikipediaMetadata(
                        query=title,
                        language=language,
                        count=0,
                        operation_type="summary_retrieval",
                        error_type="invalid_parameters",
                    ).model_dump(),
                )

            self._color_log(f"📝 Retrieving summary for: {title} (language: {language})", Color.cyan)

            # Get the summary
            summary = wikipedia.summary(title, sentences=sentences, auto_suggest=auto_suggest, redirect=redirect)

            # Get the URL
            url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"

            # Create article object with just the summary
            article = WikipediaArticle(
                title=title,
                url=url,
                content="",  # Empty content since we're just getting the summary
                summary=summary,
            )

            # Format output for LLM
            formatted_output = self._format_article(article, output_format, include_full_content=False)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create metadata
            metadata = WikipediaMetadata(
                query=title,
                language=language,
                count=1,
                operation_type="summary_retrieval",
                execution_time=execution_time,
            )

            self._color_log(f"✅ Retrieved summary for: {title}", Color.green)

            return ActionResponse(
                success=True,
                message=formatted_output,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to retrieve Wikipedia summary: {str(e)}"
            self.logger.error(f"Wikipedia summary retrieval error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=title,
                    language=language,
                    count=0,
                    operation_type="summary_retrieval",
                    error_type="summary_retrieval_error",
                ).model_dump(),
            )

    def mcp_get_article_categories(
        self,
        title: str = Field(..., description="Title of the Wikipedia article"),
        language: str = Field("en", description="Language code for Wikipedia (e.g., 'en', 'es', 'fr')"),
        output_format: str = Field("markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Get categories for a Wikipedia article.

        This tool provides Wikipedia article category retrieval with:
        - Multi-language support
        - LLM-optimized result formatting
        - Error handling

        Args:
            title: Title of the Wikipedia article
            language: Language code for Wikipedia
            output_format: Format for the response output

        Returns:
            ActionResponse with article categories and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(title, FieldInfo):
                title = title.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default

            # Validate parameters
            if not title:
                return ActionResponse(
                    success=False,
                    message="Article title cannot be empty",
                    metadata=WikipediaMetadata(
                        query=title,
                        language=language,
                        count=0,
                        operation_type="categories_retrieval",
                        error_type="invalid_parameters",
                    ).model_dump(),
                )

            self._color_log(
                f"🏷️ Retrieving categories for Wikipedia article: {title} (language: {language})", Color.cyan
            )

            # Get the page
            page = wikipedia.page(title, auto_suggest=True, redirect=True)

            # Format output for LLM
            if output_format == "json":
                formatted_output = json.dumps(page.categories, indent=2)
            elif output_format == "text":
                formatted_output = f"Categories for {title}:\n" + "\n".join(f"- {cat}" for cat in page.categories)
            else:  # markdown
                formatted_output = f"# Categories for {title}\n\n" + "\n".join(f"- {cat}" for cat in page.categories)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create metadata
            metadata = WikipediaMetadata(
                query=title,
                language=language,
                count=len(page.categories),
                operation_type="categories_retrieval",
                execution_time=execution_time,
                article_id=page.pageid,
            )

            self._color_log(f"✅ Retrieved {len(page.categories)} categories for: {title}", Color.green)

            return ActionResponse(
                success=True,
                message=formatted_output,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to retrieve Wikipedia article categories: {str(e)}"
            self.logger.error(f"Wikipedia categories retrieval error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=title,
                    language=language,
                    count=0,
                    operation_type="categories_retrieval",
                    error_type="categories_retrieval_error",
                ).model_dump(),
            )

    def mcp_get_article_links(
        self,
        title: str = Field(..., description="Title of the Wikipedia article"),
        language: str = Field("en", description="Language code for Wikipedia (e.g., 'en', 'es', 'fr')"),
        output_format: str = Field("markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Get links from a Wikipedia article.

        This tool provides Wikipedia article link retrieval with:
        - Multi-language support
        - LLM-optimized result formatting
        - Error handling

        Args:
            title: Title of the Wikipedia article
            language: Language code for Wikipedia
            output_format: Format for the response output

        Returns:
            ActionResponse with article links and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(title, FieldInfo):
                title = title.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default

            # Validate parameters
            if not title:
                return ActionResponse(
                    success=False,
                    message="Article title cannot be empty",
                    metadata=WikipediaMetadata(
                        query=title,
                        language=language,
                        count=0,
                        operation_type="links_retrieval",
                        error_type="invalid_parameters",
                    ).model_dump(),
                )

            self._color_log(f"🔗 Retrieving links from Wikipedia article: {title} (language: {language})", Color.cyan)

            # Get the page
            page = wikipedia.page(title, auto_suggest=True, redirect=True)

            # Format results
            formatted_results = []
            for link_title in page.links:
                try:
                    url = f"https://{language}.wikipedia.org/wiki/{link_title.replace(' ', '_')}"
                    result = WikipediaSearchResult(title=link_title, url=url)
                    formatted_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error formatting link '{link_title}': {str(e)}")

            # Format output for LLM
            if output_format == "json":
                formatted_output = json.dumps([result.model_dump() for result in formatted_results], indent=2)
            elif output_format == "text":
                formatted_output = f"Links from {title}:\n" + "\n".join(
                    f"- {result.title}" for result in formatted_results
                )

            else:  # markdown
                formatted_output = f"# Links from {title}\n\n"
                # Limit to first 50 links to avoid overwhelming output
                for i, result in enumerate(formatted_results[:50], 1):
                    formatted_output += f"{i}. [{result.title}]({result.url})\n"
                if len(formatted_results) > 50:
                    formatted_output += f"\n... and {len(formatted_results) - 50} more links"

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create metadata
            metadata = WikipediaMetadata(
                query=title,
                language=language,
                count=len(formatted_results),
                operation_type="links_retrieval",
                execution_time=execution_time,
                article_id=page.pageid,
            )

            self._color_log(f"✅ Retrieved {len(formatted_results)} links from: {title}", Color.green)

            return ActionResponse(
                success=True,
                message=formatted_output,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to retrieve Wikipedia article links: {str(e)}"
            self.logger.error(f"Wikipedia links retrieval error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=title,
                    language=language,
                    count=0,
                    operation_type="links_retrieval",
                    error_type="links_retrieval_error",
                ).model_dump(),
            )

    def mcp_get_article_history(
        self,
        title: str = Field(..., description="Title of the Wikipedia article"),
        date: str = Field(
            ...,
            description=("Target date in YYYY/MM/DD format. If day is omitted, last day of month will be used"),
        ),
        language: str = Field("en", description="Language code for Wikipedia (e.g., 'en', 'es', 'fr')"),
        auto_suggest: bool = Field(
            False,
            description="Whether to use Wikipedia's auto-suggest feature and handle redirects",
        ),
        output_format: str = Field("markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Get historical version of a Wikipedia page closest to the specified date.

        This tool provides historical Wikipedia article retrieval with:
        - Date-based version lookup
        - Auto-suggestion support
        - Multi-language support
        - LLM-optimized result formatting
        - Error handling

        If exact date version is not available, returns the closest version before that date.
        Supports auto-suggestion and redirects to handle company name changes or variations.

        Args:
            title: The title of the Wikipedia page
            date: Target date in YYYY/MM/DD format
            language: Language code for Wikipedia
            auto_suggest: Whether to use Wikipedia's auto-suggest and handle redirects
            output_format: Format for the response output

        Returns:
            ActionResponse with historical article content and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(title, FieldInfo):
                title = title.default
            if isinstance(date, FieldInfo):
                date = date.default
            if isinstance(language, FieldInfo):
                language = language.default
            if isinstance(auto_suggest, FieldInfo):
                auto_suggest = auto_suggest.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default

            # Validate parameters
            if not title:
                return ActionResponse(
                    success=False,
                    message="Article title cannot be empty",
                    metadata=WikipediaMetadata(
                        query=title,
                        language=language,
                        count=0,
                        operation_type="history_retrieval",
                        error_type="invalid_parameters",
                    ).model_dump(),
                )

            self._color_log(
                f"📅 Retrieving historical version of Wikipedia article: {title} for date: {date}", Color.cyan
            )

            # First try to find the correct page title using search and auto-suggest
            actual_title = title
            if auto_suggest:
                try:
                    # Search for the page and get the actual title
                    search_results = wikipedia.search(title, results=1)
                    if search_results:
                        # Get the page to handle redirects and get the canonical title
                        page = wikipedia.page(search_results[0], auto_suggest=True, redirect=True)
                        actual_title = page.title
                        self.logger.info(f"Found matching page: {actual_title} for query: {title}")
                except Exception as e:
                    self.logger.warning(f"Auto-suggest failed for {title}: {str(e)}")

            # Parse the date
            date_parts = date.split("/")
            year = int(date_parts[0])
            month = int(date_parts[1])
            day = int(date_parts[2]) if len(date_parts) > 2 else calendar.monthrange(year, month)[1]

            target_date = datetime(year, month, day)

            # Get page revisions
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": actual_title,
                "rvprop": "ids|timestamp|user|comment|content",
                "rvlimit": 1,
                "rvdir": "older",
                "rvstart": target_date.isoformat(),
                "format": "json",
            }

            # Make API request
            API_URL = f"https://{language}.wikipedia.org/w/api.php"
            response = requests.get(API_URL, params=params, timeout=5)
            data = response.json()

            # Process response
            page = next(iter(data["query"]["pages"].values()))
            if "revisions" in page:
                revision = page["revisions"][0]
                actual_date = datetime.fromisoformat(revision["timestamp"].replace("Z", "+00:00"))

                # Create URL for this version
                page_id = page["pageid"]
                rev_id = revision["revid"]
                url = f"https://{language}.wikipedia.org/w/index.php?oldid={rev_id}"

                # Create article object
                article = WikipediaArticle(
                    title=actual_title,
                    pageid=page_id,
                    url=url,
                    content=revision["*"],
                    summary=f"Historical version from {actual_date.strftime('%Y/%m/%d')}",
                    images=[],  # Historical versions don't include images
                    categories=[],
                    links=[],
                    references=[],
                    sections=[],
                    original_query=title,
                    requested_date=target_date.strftime("%Y/%m/%d"),
                    actual_date=actual_date.strftime("%Y/%m/%d"),
                    is_exact_date=actual_date.date() == target_date.date(),
                    is_redirect=actual_title != title,
                    editor=revision["user"],
                    edit_comment=revision.get("comment", ""),
                )

                # Format output for LLM
                formatted_output = self._format_article(article, output_format, include_full_content=True)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Create metadata
                metadata = WikipediaMetadata(
                    query=title,
                    language=language,
                    count=1,
                    operation_type="history_retrieval",
                    execution_time=execution_time,
                    article_id=page_id,
                    is_redirect=actual_title != title,
                    requested_date=target_date.strftime("%Y/%m/%d"),
                    actual_date=actual_date.strftime("%Y/%m/%d"),
                )

                self._color_log(f"✅ Retrieved historical version from {actual_date.strftime('%Y/%m/%d')}", Color.green)

                return ActionResponse(
                    success=True,
                    message=formatted_output,
                    metadata=metadata.model_dump(),
                )

            # No revision found
            error_msg = f"No revision found for {actual_title} (original query: {title}) before {date}"

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=title,
                    language=language,
                    count=0,
                    operation_type="history_retrieval",
                    error_type="no_revision_found",
                    requested_date=target_date.strftime("%Y/%m/%d"),
                ).model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to retrieve Wikipedia article history: {str(e)}"
            self.logger.error(f"Wikipedia history retrieval error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WikipediaMetadata(
                    query=title,
                    language=language,
                    count=0,
                    operation_type="history_retrieval",
                    error_type="history_retrieval_error",
                    requested_date=date,
                ).model_dump(),
            )

    def mcp_get_wikipedia_capabilities(self) -> ActionResponse:
        """Get information about Wikipedia service capabilities and configuration.

        Returns:
            ActionResponse with Wikipedia service capabilities and current configuration
        """
        capabilities = {
            "supported_features": [
                "Article search",
                "Full content retrieval",
                "Summary generation",
                "Random article fetching",
                "Category extraction",
                "Link extraction",
                "Historical version access",
                "Multiple output formats (markdown, json, text)",
                "LLM-optimized result formatting",
                "Multi-language support",
            ],
            "supported_formats": ["markdown", "json", "text"],
            "configuration": {
                "default_language": self.default_language,
                "max_search_results": self.max_search_results,
                "max_random_articles": self.max_random_articles,
                "default_summary_sentences": self.default_summary_sentences,
            },
        }

        formatted_info = f"""# Wikipedia Service Capabilities
                
        ## Features
        {chr(10).join(f"- {feature}" for feature in capabilities["supported_features"])}

        ## Supported Output Formats
        {chr(10).join(f"- {fmt}" for fmt in capabilities["supported_formats"])}

        ## Current Configuration
        - **Default Language:** {capabilities["configuration"]["default_language"]}
        - **Max Search Results:** {capabilities["configuration"]["max_search_results"]}
        - **Max Random Articles:** {capabilities["configuration"]["max_random_articles"]}
        - **Default Summary Sentences:** {capabilities["configuration"]["default_summary_sentences"]}
        """

        return ActionResponse(success=True, message=formatted_info, metadata=capabilities)


# Default arguments for testing
if __name__ == "__main__":
    load_dotenv()

    arguments = ActionArguments(
        name="wikipedia",
        transport="stdio",
        workspace=os.getenv("AWORLD_WORKSPACE", "~"),
    )

    try:
        service = WikipediaCollection(arguments)
        service.run()
    except Exception as e:
        print(f"An error occurred: {e}")
