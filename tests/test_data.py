from __future__ import annotations

from mimic_comal.data import MIMICRecord, audit_records


def test_subject_group_leakage_is_detected() -> None:
    labels = ("250", "401")
    clean = [
        MIMICRecord(0, "1", "10", "train", ("250",), "a"),
        MIMICRecord(1, "2", "11", "validation", ("401",), "b"),
        MIMICRecord(2, "3", "12", "test", ("250", "401"), "c"),
    ]
    assert not audit_records(clean, labels)["group_leakage"]
    leaked = clean + [MIMICRecord(3, "4", "10", "test", ("250",), "d")]
    assert audit_records(leaked, labels)["group_leakage"]
