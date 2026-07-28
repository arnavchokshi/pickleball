# Durable model store — S3 + GCP snapshot

Created 2026-07-28 (overnight fleet lane, owner directive: "store models so
future runs don't have to do as much setup").

## Fast-boot path (primary)

New GPU VMs boot from the GCP snapshot **`pickleball-court23-warm-20260728`**
(300 GB, us-central1, taken from `pickleball-gpu-court23`'s prepared boot disk).
It contains the full warm runtime: `/home/arnavchokshi/coldstart_20260706/`
(Fast-SAM-3D-Body checkout + checkpoints + `body_venv` + `hf_home` + repo
checkout). `pickleball-gpu-night1` and `pickleball-gpu-night2` were created
from it and had working `nvidia-smi` + runtime with zero additional setup.

```bash
gcloud compute instances create <name> --zone=us-central1-f \
  --machine-type=a2-highgpu-1g --provisioning-model=SPOT \
  --instance-termination-action=STOP --maintenance-policy=TERMINATE \
  --create-disk=boot=yes,source-snapshot=pickleball-court23-warm-20260728,size=300,type=pd-balanced,auto-delete=yes
```

After boot: pin the host key (`scripts/fleet/refresh_remote_host.sh --host <ip>`),
arm a poweroff rail (`sudo shutdown -P +720`), and sync the repo to main HEAD
(`scripts/racketsport/remote_body_dispatch.py --sync-remote-code`).

## Durable archive (S3)

Bucket prefix: `s3://sway-videos/pickleball-models/20260728/` (AWS account
823214746722, writer identity `sway-s3-upload`). Restore with
`scripts/fleet/bootstrap_models_from_s3.sh` (sha-verified, idempotent).

| S3 object | Role | sha256 |
|---|---|---|
| `yolo26m.pt` | TRK person detector (default) | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| `yolo26n-cls.pt` | aux classifier | `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81` |
| `osnet_x1_0_market1501.pt` | ReID (global association) | `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154` |
| `court_model_v2.pt` | CAL court keypoint evidence head (current) | `31da51630b82a85ee39384e65eb705b045adcdda900dd025ca15784a2edd3ffe` |
| `resnet34-b627a593.pth` | torchvision backbone for court model | `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` |
| `model_tennis_court_det.pt` | external tennis-court detector (diagnostic) | `09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04` |
| `wasb_tennis_best.pth.tar` | BALL WASB detector (default) | `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb` |
| `sam-3d-body-dinov3_model.ckpt` | BODY SAM-3D-Body checkpoint (2.0 GiB) | `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf` |
| `mhr_model.pt` | BODY MHR asset (664 MiB) | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |

The two BODY files match the shas pinned in `models/MANIFEST.json`; they were
uploaded from the VM via presigned PUT URLs, so no AWS credentials ever landed
on a VM. Credential-less hosts restore via presigned GET URLs generated on a
credentialed machine (see the header of `bootstrap_models_from_s3.sh`).

Storage identity and existence are not accuracy claims; checkpoint trust bands
are unchanged and `VERIFIED=0` remains binding.
