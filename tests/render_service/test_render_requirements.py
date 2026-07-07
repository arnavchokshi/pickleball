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


def test_render_requirements_include_court_prediction_runtime_dependencies() -> None:
    requirements = Path("requirements-render.txt").read_text(encoding="utf-8").splitlines()

    assert "-r requirements-racketsport.txt" in requirements
    assert any(line.startswith("opencv-python-headless") for line in requirements)


def test_render_requirements_pin_product_infra_dependencies() -> None:
    requirements = Path("requirements-render.txt").read_text(encoding="utf-8").splitlines()

    for pin in PRODUCT_INFRA_PINS:
        assert pin in requirements, f"missing pinned dependency line: {pin}"
