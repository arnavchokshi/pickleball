# Pickleball additions to vendored checkouts (tracked backups)

The vendor dirs (see ../VENDOR_PINS.md) are pinned gitlinks; files here are OUR additions living inside
them, backed up because the outer repo cannot track gitlink contents. After a fresh vendor clone at the
pinned SHA, restore with:
  cp -R third_party/pickleball_vendor_additions/WASB-SBDT/* third_party/WASB-SBDT/
  cp -R third_party/pickleball_vendor_additions/blurball/* third_party/blurball/
Each file carries a "pickleball addition, not upstream" header. Registry files (src/datasets/__init__.py)
are FULL modified copies of the upstream file at the pinned SHA + our marked entries — overwrite is safe
at that SHA only. Tests covering these live in tests/racketsport/test_ball_wasb_dataset.py.
