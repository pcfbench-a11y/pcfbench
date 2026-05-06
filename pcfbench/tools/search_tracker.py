"""Stateful search-tracker over the trimmed ``MaterialLibrary``.

- A keyword/vector query that has already been executed yields a
  ``"All queries have already been executed."`` notice.
- Materials previously surfaced are filtered out of subsequent results.
- ``search_results`` returns the candidate activity names the agent
  consumes as user input.
"""

from __future__ import annotations

import logging
from typing import Any

import pydantic as pyd

from pcfbench.tools.material_library import (
    MaterialLibrary,
)

logger = logging.getLogger(__name__)


class SearchMaterialsParams(pyd.BaseModel):
    keyword_queries: list[str] = pyd.Field(
        description=(
            "List of keyword queries to search for materials based on " "activity name."
        )
    )
    vector_queries: list[str] = pyd.Field(
        description="List of vector queries for semantic search of materials."
    )
    max_keyword_search_results: int = pyd.Field(
        description="Maximum keyword search results to return.", default=10
    )
    max_vector_search_results: int = pyd.Field(
        description="Maximum vector search results to return.", default=10
    )


class InspectMaterialsParams(pyd.BaseModel):
    material_names: list[str] = pyd.Field(
        description=(
            "List of activity names to inspect in detail (matched against "
            "the picklist's ``activity_name`` field)."
        )
    )


class SearchMaterialTracker:
    """Per-call search/inspect state. Constructed fresh per dataset item."""

    def __init__(self, material_library: MaterialLibrary) -> None:
        self.material_library = material_library
        self._executed_keyword_queries: set[str] = set()
        self._executed_vector_queries: set[str] = set()
        self._returned_material_ids: set[str] = set()

    def search_materials(
        self,
        keyword_queries: list[str],
        vector_queries: list[str],
        max_keyword_search_results: int = 10,
        max_vector_search_results: int = 10,
    ) -> dict[str, Any]:
        new_keyword_queries = [
            q for q in keyword_queries if q not in self._executed_keyword_queries
        ]
        new_vector_queries = [
            q for q in vector_queries if q not in self._executed_vector_queries
        ]

        if not new_keyword_queries and not new_vector_queries:
            logger.info(
                "All provided keyword/vector queries have already been "
                "executed. No new items will be returned."
            )
            return {
                "search_results": (
                    "All queries have already been executed. Submit new "
                    "queries or use the existing results."
                )
            }

        logger.info(
            "Searching library with NEW keywords: %s | NEW vectors: %s",
            new_keyword_queries,
            new_vector_queries,
        )

        search_results = self.material_library.search_combined(
            keyword_queries=new_keyword_queries,
            vector_queries=new_vector_queries,
            max_keyword_search_results=max_keyword_search_results,
            max_vector_search_results=max_vector_search_results,
        )

        novel_results = [
            m
            for m in search_results
            if m.activity_uuid_product_uuid not in self._returned_material_ids
        ]

        already_executed_keyword = [
            q for q in keyword_queries if q in self._executed_keyword_queries
        ]
        already_executed_vector = [
            q for q in vector_queries if q in self._executed_vector_queries
        ]

        self._executed_keyword_queries.update(new_keyword_queries)
        self._executed_vector_queries.update(new_vector_queries)
        self._returned_material_ids.update(
            m.activity_uuid_product_uuid for m in novel_results
        )

        if not novel_results:
            logger.info(
                "Search returned materials, but all have already been "
                "provided previously. No new items found."
            )
            return {
                "search_results": (
                    "No new items found for the provided queries. All "
                    "retrieved items have already been retrieved previously."
                )
            }

        out: dict[str, Any] = {
            "search_results": [m.reference_product_name for m in novel_results],
        }
        if already_executed_keyword:
            out["already_executed_keyword_queries"] = (
                "The following keyword queries were not re-run as they were "
                "already executed and won't return new results: "
                + ", ".join(already_executed_keyword)
            )
        if already_executed_vector:
            out["already_executed_vector_queries"] = (
                "The following vector queries were not re-run as they were "
                "already executed and won't return new results: "
                + ", ".join(already_executed_vector)
            )
        return out

    def inspect_materials(
        self,
        material_names: list[str],
    ) -> list[dict[str, Any]]:
        """Look up by reference_product_name. Tools surface that form to
        the model in search results, so the model is expected to inspect
        by the same form."""
        logger.info("Inspecting materials: %s", material_names)
        rows: list[dict[str, Any]] = []
        for name in material_names:
            for (
                material
            ) in self.material_library.get_materials_by_reference_product_name(name):
                rows.append(
                    {
                        "reference_product_name": material.reference_product_name,
                        "product_information": material.product_information,
                    }
                )
        return rows
