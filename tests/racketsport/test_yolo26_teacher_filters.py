from __future__ import annotations

from scripts.racketsport import run_yolo26_teacher as teacher
from scripts.racketsport.run_yolo26_teacher import PERSON_CLASS, SPORTS_BALL_CLASS, TENNIS_RACKET_CLASS


def _candidate(cls: int, score: float, xyxy: tuple[float, float, float, float], index: int):
    assert hasattr(teacher, "DetectionCandidate")
    return teacher.DetectionCandidate(cls=cls, score=score, xyxy=xyxy, box_index=index)


def test_filter_frame_detections_applies_class_thresholds_and_geometry() -> None:
    assert hasattr(teacher, "build_detection_filters")
    assert hasattr(teacher, "filter_frame_detections")
    filters = teacher.build_detection_filters(
        person_min_conf=0.5,
        ball_min_conf=0.25,
        racket_min_conf=0.3,
    )

    kept = teacher.filter_frame_detections(
        [
            _candidate(PERSON_CLASS, 0.49, (10, 10, 90, 210), 0),
            _candidate(PERSON_CLASS, 0.8, (10, 10, 90, 210), 1),
            _candidate(PERSON_CLASS, 0.9, (10, 10, 14, 210), 2),
            _candidate(SPORTS_BALL_CLASS, 0.4, (100, 100, 101, 106), 3),
            _candidate(SPORTS_BALL_CLASS, 0.4, (100, 100, 106, 106), 4),
            _candidate(TENNIS_RACKET_CLASS, 0.29, (200, 200, 235, 210), 5),
            _candidate(TENNIS_RACKET_CLASS, 0.31, (200, 200, 235, 210), 6),
        ],
        frame_width=1920,
        frame_height=1080,
        filters=filters,
        max_players_per_frame=4,
    )

    assert [(candidate.cls, candidate.box_index) for candidate in kept] == [
        (PERSON_CLASS, 1),
        (SPORTS_BALL_CLASS, 4),
        (TENNIS_RACKET_CLASS, 6),
    ]


def test_filter_frame_detections_caps_people_per_frame_by_confidence() -> None:
    filters = teacher.build_detection_filters(person_min_conf=0.4)

    kept = teacher.filter_frame_detections(
        [
            _candidate(PERSON_CLASS, 0.50, (10, 10, 90, 210), 0),
            _candidate(PERSON_CLASS, 0.90, (110, 10, 190, 210), 1),
            _candidate(PERSON_CLASS, 0.70, (210, 10, 290, 210), 2),
            _candidate(PERSON_CLASS, 0.80, (310, 10, 390, 210), 3),
            _candidate(PERSON_CLASS, 0.60, (410, 10, 490, 210), 4),
            _candidate(SPORTS_BALL_CLASS, 0.30, (100, 100, 106, 106), 5),
        ],
        frame_width=1920,
        frame_height=1080,
        filters=filters,
        max_players_per_frame=4,
    )

    assert [(candidate.cls, candidate.box_index) for candidate in kept] == [
        (PERSON_CLASS, 1),
        (PERSON_CLASS, 3),
        (PERSON_CLASS, 2),
        (PERSON_CLASS, 4),
        (SPORTS_BALL_CLASS, 5),
    ]
