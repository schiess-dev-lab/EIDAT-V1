from ui_next.qt_main import _td_metric_program_segments, _td_order_metric_serials


def test_td_metric_program_segments_groups_contiguous_programs() -> None:
    labels = ["SN-001", "SN-002", "SN-003", "SN-004", "SN-005"]
    rows = [
        {"serial": "SN-003", "program_title": "Program Beta"},
        {"serial": "SN-005", "program_title": "Program Alpha"},
        {"serial": "SN-001", "program_title": "Program Alpha"},
        {"serial": "SN-004", "program_title": "Program Beta"},
        {"serial": "SN-002", "program_title": "Program Alpha"},
    ]

    ordered = _td_order_metric_serials(labels, rows)
    segments = _td_metric_program_segments(ordered, rows)

    assert ordered == ["SN-001", "SN-002", "SN-005", "SN-003", "SN-004"]
    assert segments == [
        {"program": "Program Alpha", "start": 0, "end": 2, "serials": ["SN-001", "SN-002", "SN-005"]},
        {"program": "Program Beta", "start": 3, "end": 4, "serials": ["SN-003", "SN-004"]},
    ]


def test_td_metric_program_segments_falls_back_to_unknown_program() -> None:
    labels = ["SN-101", "SN-100", "SN-200", "SN-150"]
    rows = [
        {"serial": "SN-100", "program_title": ""},
        {"serial": "SN-200", "program_title": "Program Beta"},
    ]

    ordered = _td_order_metric_serials(labels, rows)
    segments = _td_metric_program_segments(ordered, rows)

    assert ordered == ["SN-200", "SN-100", "SN-101", "SN-150"]
    assert segments == [
        {"program": "Program Beta", "start": 0, "end": 0, "serials": ["SN-200"]},
        {"program": "Unknown Program", "start": 1, "end": 3, "serials": ["SN-100", "SN-101", "SN-150"]},
    ]
