from pathlib import Path

PRODUCT_INFRA_PINS = (
    "pymongo>=4.10,<5",
    "boto3>=1.34,<2",
    "argon2-cffi>=21.3,<24",
    "pyjwt>=2.9,<3",
    "slowapi>=0.1.9,<1",
    "moto[s3]>=5.0,<6",
    "mongomock>=4.1,<5",
)

# INFRA-2: the pull-worker daemon's own tiny venv -- deliberately just
# httpx + boto3 (it shells out to the heavy pipeline venv for GPU work
# rather than importing torch/cv2 itself). Kept in a SEPARATE requirements
# file (requirements-worker.txt) from requirements-render.txt because it
# deploys to a different box (the fleet worker VM, not the Render service).
WORKER_REQUIRED_PINS = (
    "httpx>=0.27,<1",
    "boto3>=1.34,<2",
)


def test_render_requirements_include_court_prediction_runtime_dependencies() -> None:
    requirements = Path("requirements-render.txt").read_text(encoding="utf-8").splitlines()

    assert "-r requirements-racketsport.txt" in requirements
    assert any(line.startswith("opencv-python-headless") for line in requirements)


def test_render_requirements_pin_product_infra_dependencies() -> None:
    requirements = Path("requirements-render.txt").read_text(encoding="utf-8").splitlines()

    for pin in PRODUCT_INFRA_PINS:
        assert pin in requirements, f"missing pinned dependency line: {pin}"


def test_requirements_worker_pins_are_present_and_minimal() -> None:
    lines = Path("requirements-worker.txt").read_text(encoding="utf-8").splitlines()
    non_comment = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    for pin in WORKER_REQUIRED_PINS:
        assert pin in non_comment, f"missing pinned dependency line: {pin}"
    # Deliberately tiny venv (plan §INFRA-2): nothing beyond httpx + boto3.
    assert len(non_comment) == len(WORKER_REQUIRED_PINS), (
        f"requirements-worker.txt should only pin {WORKER_REQUIRED_PINS}; found {non_comment}"
    )
