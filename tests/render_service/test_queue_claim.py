"""Queue claim atomicity + stale-heartbeat sweep (INFRA-2), against
mongomock directly -- no FastAPI/TestClient/threading involved. Real
concurrent-claim races are out of scope here (mongomock is single-threaded);
that proof is reserved for a `@pytest.mark.integration` testcontainers test
per the approved design, not part of this lane.
"""

import mongomock

from server.routes.worker import (
    MAX_ATTEMPTS_BEFORE_FAIL,
    STALE_HEARTBEAT_S,
    _claim_next_queued_job,
    _reclaim_stale_jobs,
)


def _db():
    return mongomock.MongoClient()["pickleball"]


def _queued_doc(job_id: str, created_at: float) -> dict:
    return {
        "_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "attempts": 0,
        "worker_id": None,
        "heartbeat_at": None,
        "progress": {"percent": 0, "stage": "Queued", "message": "Waiting for the GPU worker."},
    }


def test_two_sequential_claims_on_one_queued_doc_exactly_one_wins() -> None:
    db = _db()
    db.jobs.insert_one(_queued_doc("job_1", created_at=1.0))

    first = _claim_next_queued_job(db, worker_id="vm-1", now=100.0)
    second = _claim_next_queued_job(db, worker_id="vm-2", now=101.0)

    assert first is not None
    assert first["status"] == "claimed"
    assert first["worker_id"] == "vm-1"
    assert first["attempts"] == 1
    assert second is None


def test_claim_picks_oldest_created_at_first() -> None:
    db = _db()
    db.jobs.insert_one(_queued_doc("job_newer", created_at=200.0))
    db.jobs.insert_one(_queued_doc("job_older", created_at=100.0))

    claimed = _claim_next_queued_job(db, worker_id="vm-1", now=300.0)

    assert claimed is not None
    assert claimed["_id"] == "job_older"


def test_claim_returns_none_when_queue_is_empty() -> None:
    db = _db()

    assert _claim_next_queued_job(db, worker_id="vm-1", now=1.0) is None


def test_reclaim_stale_jobs_requeues_when_attempts_below_ladder_ceiling() -> None:
    db = _db()
    now = 10_000.0
    db.jobs.insert_one(
        {
            "_id": "job_1",
            "status": "claimed",
            "created_at": 1.0,
            "attempts": 1,
            "worker_id": "vm-1",
            "heartbeat_at": now - (STALE_HEARTBEAT_S + 100),
            "progress": {"percent": 40, "stage": "Running pipeline on GPU"},
        }
    )

    mutated = _reclaim_stale_jobs(db, now=now)

    assert mutated == 1
    doc = db.jobs.find_one({"_id": "job_1"})
    assert doc["status"] == "queued"
    assert doc["worker_id"] is None
    assert doc["attempts"] == 1  # sweep never increments attempts, only claim does
    assert doc["progress"]["percent"] == 40  # untouched


def test_reclaim_stale_jobs_ignores_fresh_heartbeats() -> None:
    db = _db()
    now = 10_000.0
    db.jobs.insert_one(
        {
            "_id": "job_1",
            "status": "running",
            "created_at": 1.0,
            "attempts": 1,
            "worker_id": "vm-1",
            "heartbeat_at": now - 10,
            "progress": {},
        }
    )

    mutated = _reclaim_stale_jobs(db, now=now)

    assert mutated == 0
    assert db.jobs.find_one({"_id": "job_1"})["status"] == "running"


def test_reclaim_stale_jobs_ignores_queued_and_terminal_statuses() -> None:
    db = _db()
    now = 10_000.0
    db.jobs.insert_one(_queued_doc("job_queued", created_at=1.0))
    db.jobs.insert_one(
        {
            "_id": "job_complete",
            "status": "complete",
            "created_at": 1.0,
            "attempts": 1,
            "worker_id": "vm-1",
            "heartbeat_at": now - (STALE_HEARTBEAT_S + 100),
            "progress": {},
        }
    )

    mutated = _reclaim_stale_jobs(db, now=now)

    assert mutated == 0


def test_attempts_ladder_second_stale_heartbeat_fails_with_progress_preserved() -> None:
    db = _db()
    db.jobs.insert_one(_queued_doc("job_1", created_at=1.0))

    # Round 1: claim, run, go stale -> requeued (attempts=1 < ceiling).
    claimed = _claim_next_queued_job(db, worker_id="vm-1", now=100.0)
    assert claimed["attempts"] == 1
    db.jobs.update_one(
        {"_id": "job_1"},
        {"$set": {"progress": {"percent": 55, "stage": "Running pipeline on GPU", "message": "tracking"}}},
    )
    first_sweep_now = 100.0 + STALE_HEARTBEAT_S + 50
    mutated = _reclaim_stale_jobs(db, now=first_sweep_now)
    assert mutated == 1
    doc = db.jobs.find_one({"_id": "job_1"})
    assert doc["status"] == "queued"
    assert doc["attempts"] == MAX_ATTEMPTS_BEFORE_FAIL - 1

    # Round 2: claim again (attempts -> ceiling), go stale again -> failed,
    # with the last-seen progress preserved verbatim.
    reclaimed = _claim_next_queued_job(db, worker_id="vm-2", now=first_sweep_now + 1)
    assert reclaimed["attempts"] == MAX_ATTEMPTS_BEFORE_FAIL
    second_sweep_now = first_sweep_now + 1 + STALE_HEARTBEAT_S + 50
    mutated = _reclaim_stale_jobs(db, now=second_sweep_now)
    assert mutated == 1

    final = db.jobs.find_one({"_id": "job_1"})
    assert final["status"] == "failed"
    assert final["error"] == "worker lost (stale heartbeat)"
    assert final["progress"] == {"percent": 55, "stage": "Running pipeline on GPU", "message": "tracking"}
