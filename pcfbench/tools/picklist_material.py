"""Trimmed picklist-material model.

A flat Pydantic v2 model containing only the fields the published
PCFBench picklist surfaces. Loaded from the JSONL shipped with the
package (``pcfbench/picklist/ecoinvent_picklist.jsonl``).
"""

from __future__ import annotations

import pydantic as pyd


class PicklistMaterial(pyd.BaseModel):
    """A single ecoinvent picklist row, sourced from the public ecoinvent
    v3.11 "Database Overview" workbook (sheet ``Cut-Off AO``).

    The model picks from ``reference_product_name`` (e.g. "corrugated
    board box"); ``activity_name`` is the parent market activity (e.g.
    "market for corrugated board box") and stays in the JSON for
    cross-referencing the source xlsx.
    """

    activity_uuid_product_uuid: str
    activity_name: str
    reference_product_name: str
    product_information: str = ""
