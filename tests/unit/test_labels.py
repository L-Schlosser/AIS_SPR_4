"""Tests for NER BIO label definitions."""

import pytest

from edge_model.extraction.labels import (
    ARZTBESUCH_LABELS,
    LABEL_SETS,
    LIEFERSCHEIN_LABELS,
    REISEKOSTEN_LABELS,
    get_id2label,
    get_label2id,
)


class TestLabelListsStartWithO:
    """All label lists must start with the 'O' (outside) tag."""

    def test_arztbesuch_starts_with_o(self):
        assert ARZTBESUCH_LABELS[0] == "O"

    def test_reisekosten_starts_with_o(self):
        assert REISEKOSTEN_LABELS[0] == "O"

    def test_lieferschein_starts_with_o(self):
        assert LIEFERSCHEIN_LABELS[0] == "O"


class TestBIOTagPairing:
    """Every B- tag that has multi-token spans should have a matching I- tag."""

    @pytest.mark.parametrize("labels", [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS])
    def test_every_i_tag_has_matching_b_tag(self, labels):
        b_entities = {tag[2:] for tag in labels if tag.startswith("B-")}
        i_entities = {tag[2:] for tag in labels if tag.startswith("I-")}
        # Every I- entity must have a corresponding B- entity
        assert i_entities.issubset(b_entities), f"I- tags without B- counterpart: {i_entities - b_entities}"

    def test_arztbesuch_paired_tags(self):
        # PATIENT, DOCTOR, FACILITY, ADDRESS, DATE, TIME all have I- tags
        for entity in ["PATIENT", "DOCTOR", "FACILITY", "ADDRESS", "DATE", "TIME"]:
            assert f"B-{entity}" in ARZTBESUCH_LABELS
            assert f"I-{entity}" in ARZTBESUCH_LABELS

    def test_reisekosten_paired_tags(self):
        for entity in ["VENDOR", "VADDRESS", "DATE", "DESC"]:
            assert f"B-{entity}" in REISEKOSTEN_LABELS
            assert f"I-{entity}" in REISEKOSTEN_LABELS

    def test_lieferschein_paired_tags(self):
        for entity in ["SENDER", "SADDR", "RECIP", "RADDR", "ITEM_DESC"]:
            assert f"B-{entity}" in LIEFERSCHEIN_LABELS
            assert f"I-{entity}" in LIEFERSCHEIN_LABELS


class TestLabel2IdAndId2Label:
    """label2id and id2label must be inverses of each other."""

    @pytest.mark.parametrize("labels", [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS])
    def test_label2id_and_id2label_are_inverse(self, labels):
        l2i = get_label2id(labels)
        i2l = get_id2label(labels)
        for label, idx in l2i.items():
            assert i2l[idx] == label
        for idx, label in i2l.items():
            assert l2i[label] == idx

    @pytest.mark.parametrize("labels", [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS])
    def test_label2id_length_matches(self, labels):
        l2i = get_label2id(labels)
        assert len(l2i) == len(labels)

    @pytest.mark.parametrize("labels", [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS])
    def test_id2label_length_matches(self, labels):
        i2l = get_id2label(labels)
        assert len(i2l) == len(labels)

    def test_o_tag_is_id_zero(self):
        for labels in [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS]:
            l2i = get_label2id(labels)
            assert l2i["O"] == 0


class TestLabelSets:
    """LABEL_SETS must contain all 3 document types."""

    def test_contains_all_document_types(self):
        assert "arztbesuchsbestaetigung" in LABEL_SETS
        assert "reisekostenbeleg" in LABEL_SETS
        assert "lieferschein" in LABEL_SETS

    def test_exactly_three_types(self):
        assert len(LABEL_SETS) == 3

    def test_arztbesuch_maps_to_correct_labels(self):
        assert LABEL_SETS["arztbesuchsbestaetigung"] is ARZTBESUCH_LABELS

    def test_reisekosten_maps_to_correct_labels(self):
        assert LABEL_SETS["reisekostenbeleg"] is REISEKOSTEN_LABELS

    def test_lieferschein_maps_to_correct_labels(self):
        assert LABEL_SETS["lieferschein"] is LIEFERSCHEIN_LABELS


class TestLabelUniqueness:
    """All labels within a list must be unique."""

    @pytest.mark.parametrize("labels", [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS])
    def test_no_duplicate_labels(self, labels):
        dupes = [lbl for lbl in labels if labels.count(lbl) > 1]
        assert len(labels) == len(set(labels)), f"Duplicate labels found: {dupes}"

    @pytest.mark.parametrize("labels", [ARZTBESUCH_LABELS, REISEKOSTEN_LABELS, LIEFERSCHEIN_LABELS])
    def test_all_tags_are_valid_bio_format(self, labels):
        for tag in labels:
            assert tag == "O" or tag.startswith("B-") or tag.startswith("I-"), f"Invalid BIO tag: {tag}"
