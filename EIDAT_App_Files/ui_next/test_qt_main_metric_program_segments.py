from ui_next.qt_main import _td_metric_program_segments


def test_td_metric_program_segments_groups_contiguous_programs() -> None:
    labels = ["SN-001", "SN-002", "SN-003", "SN-004", "SN-005"]
    rows = [
        {"serial": "SN-001", "program_title": "Program Alpha"},
        {"serial": "SN-002", "program_title": "Program Alpha"},
        {"serial": "SN-003", "program_title": "Program Beta"},
        {"serial": "SN-004", "program_title": "Program Beta"},
        {"serial": "SN-005", "program_title": "Program Alpha"},
    ]

    segments = _td_metric_program_segments(labels, rows)

    assert segments == [
        {"program": "Program Alpha", "start": 0, "end": 1, "serials": ["SN-001", "SN-002"]},
        {"program": "Program Beta", "start": 2, "end": 3, "serials": ["SN-003", "SN-004"]},
        {"program": "Program Alpha", "start": 4, "end": 4, "serials": ["SN-005"]},
    ]


def test_td_metric_program_segments_falls_back_to_unknown_program() -> None:
    labels = ["SN-100", "SN-101"]
    rows = [{"serial": "SN-100", "program_title": ""}]

    segments = _td_metric_program_segments(labels, rows)

    assert segments == [
        {"program": "Unknown Program", "start": 0, "end": 1, "serials": ["SN-100", "SN-101"]},
    ]
