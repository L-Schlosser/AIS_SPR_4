"""Postprocessing utilities for OCR results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr.engine import TextRegion


def sort_regions_by_position(regions: list[TextRegion]) -> list[TextRegion]:
    """Sort text regions by reading order: top-to-bottom, then left-to-right.

    Groups regions into lines based on vertical proximity, then sorts
    left-to-right within each line.
    """
    if not regions:
        return []

    # Sort primarily by y_min, then by x_min for same-line regions
    # Use a tolerance band for y-coordinate grouping
    sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

    if len(sorted_regions) <= 1:
        return sorted_regions

    # Group into lines: regions within similar y-coordinates
    lines: list[list[TextRegion]] = []
    current_line: list[TextRegion] = [sorted_regions[0]]
    line_y = sorted_regions[0].bbox[1]

    for region in sorted_regions[1:]:
        # Height-based tolerance for same-line grouping
        height = region.bbox[3] - region.bbox[1]
        tolerance = max(height * 0.5, 10)

        if abs(region.bbox[1] - line_y) <= tolerance:
            current_line.append(region)
        else:
            lines.append(current_line)
            current_line = [region]
            line_y = region.bbox[1]

    lines.append(current_line)

    # Sort each line left-to-right and flatten
    result: list[TextRegion] = []
    for line in lines:
        line.sort(key=lambda r: r.bbox[0])
        result.extend(line)

    return result


def merge_text(regions: list[TextRegion]) -> str:
    """Merge sorted text regions into a single string.

    Inserts line breaks where vertical gaps between regions exceed
    a threshold, and spaces between horizontally adjacent regions.
    """
    if not regions:
        return ""

    parts: list[str] = [regions[0].text]

    for i in range(1, len(regions)):
        prev = regions[i - 1]
        curr = regions[i]

        # Check vertical gap
        prev_bottom = prev.bbox[3]
        curr_top = curr.bbox[1]
        prev_height = prev.bbox[3] - prev.bbox[1]
        vertical_gap = curr_top - prev_bottom

        if vertical_gap > max(prev_height * 0.5, 5):
            # New line
            parts.append("\n")
            parts.append(curr.text)
        else:
            # Same line, add space
            parts.append(" ")
            parts.append(curr.text)

    return "".join(parts)
