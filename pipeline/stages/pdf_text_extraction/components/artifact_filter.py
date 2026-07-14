"""
ArtifactFilter

Removes common layout artifacts (floats outside the text body, single-char
elements on figure pages, lines with no alphabetic content) from a list of
LayoutElements.

Delegates to ``parsers.layout_utils.filter_artifacts``.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from pipeline.stages.pdf_text_extraction.config import FilteringConfig
from pipeline.stages.pdf_text_extraction.models.dto import LayoutElement
from nlp_histo.parsers.layout_utils import filter_artifacts

logger = logging.getLogger(__name__)


class ArtifactFilter:
    """
    Artifact filter backed by the rule-based + optional NER approach in
    ``parsers/layout_utils.py``.

    Parameters
    ----------
    config:
        FilteringConfig controlling NER filtering and ligature fixing.
    nlp:
        Optional scispaCy model.  Required when
        ``config.apply_ner_filtering`` is True.
    """

    def __init__(
        self,
        config: Optional[FilteringConfig] = None,
        nlp=None,
    ) -> None:
        self._config = config or FilteringConfig()
        self._nlp = nlp

    def filter_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Remove artifact elements from ``elements``.

        Args:
            elements: Raw layout elements to filter.

        Returns:
            Cleaned list with artifacts removed.
        """
        nlp = self._nlp if self._config.apply_ner_filtering else None
        element_dicts = [el.to_dict() for el in elements]
        # Identity map: object id → original LayoutElement.  filter_artifacts
        # filters by reference (it never constructs new dicts), so id() of
        # each kept dict matches an entry in this map regardless of any
        # in-place mutation a future contributor might introduce.  Replaces
        # the prior `dict in list` rebuild which was O(N²) and silently
        # dropped elements the moment any kept dict was mutated.
        dict_id_to_element = {id(d): el for d, el in zip(element_dicts, elements)}
        filtered_dicts = filter_artifacts(element_dicts, nlp=nlp)
        result = [dict_id_to_element[id(d)] for d in filtered_dicts]
        logger.debug(
            "ArtifactFilter: %d → %d elements (removed %d)",
            len(elements),
            len(result),
            len(elements) - len(result),
        )
        return result
