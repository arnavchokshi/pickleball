from __future__ import annotations

import plistlib
from pathlib import Path


def test_ios_app_supports_portrait_and_landscape_orientations() -> None:
    plist_path = Path("ios/App/Info.plist")
    payload = plistlib.loads(plist_path.read_bytes())

    required = {
        "UIInterfaceOrientationPortrait",
        "UIInterfaceOrientationPortraitUpsideDown",
        "UIInterfaceOrientationLandscapeLeft",
        "UIInterfaceOrientationLandscapeRight",
    }

    assert required.issubset(set(payload["UISupportedInterfaceOrientations"]))
    assert required.issubset(set(payload["UISupportedInterfaceOrientations~iphone"]))
    assert required.issubset(set(payload["UISupportedInterfaceOrientations~ipad"]))
