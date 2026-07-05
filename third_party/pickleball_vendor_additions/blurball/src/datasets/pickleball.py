# pickleball addition, not upstream
import csv
import logging
import os
import os.path as osp

import numpy as np
from omegaconf import DictConfig

from utils import Center

log = logging.getLogger(__name__)


def load_csv(csv_path, visible_flags, frame_dir=None):
    xyvs = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["file name", "visibility", "x-coordinate", "y-coordinate"]
        if reader.fieldnames != expected:
            raise ValueError("{} columns must be {}".format(csv_path, expected))
        for row in reader:
            fname = row["file name"]
            fid = int(osp.splitext(fname)[0])
            if fid in xyvs:
                raise KeyError("fid {} already exists".format(fid))
            frame_path = osp.join(frame_dir, fname) if frame_dir is not None else None
            visibility = int(row["visibility"])
            xyvs[fid] = {
                "center": Center(
                    x=float(row["x-coordinate"]),
                    y=float(row["y-coordinate"]),
                    is_visible=True if visibility in visible_flags else False,
                ),
                "file_name": fname,
                "frame_path": frame_path,
            }
    return xyvs


def get_clips(cfg, train_or_test="test", gt=True):
    root_dir = cfg["dataset"]["root_dir"]
    matches = cfg["dataset"][train_or_test]["matches"]
    csv_filename = cfg["dataset"]["csv_filename"]
    ext = cfg["dataset"]["ext"]
    visible_flags = cfg["dataset"]["visible_flags"]

    clip_dict = {}
    for match in matches:
        match_dir = osp.join(root_dir, match)
        clip_names = [name for name in os.listdir(match_dir) if osp.isdir(osp.join(match_dir, name))]
        clip_names.sort()
        for clip_name in clip_names:
            clip_dir = osp.join(root_dir, match, clip_name)
            clip_csv_path = osp.join(root_dir, match, clip_name, csv_filename)
            frame_names = [name for name in os.listdir(clip_dir) if name.endswith(ext)]
            frame_names.sort()
            ball_xyvs = load_csv(clip_csv_path, visible_flags) if gt else None
            clip_dict[(match, clip_name)] = {
                "clip_dir_or_path": clip_dir,
                "clip_gt_dict": ball_xyvs,
                "frame_names": frame_names,
            }
    return clip_dict


class Pickleball(object):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self._root_dir = cfg["dataset"]["root_dir"]
        self._ext = cfg["dataset"]["ext"]
        self._csv_filename = cfg["dataset"]["csv_filename"]
        self._visible_flags = cfg["dataset"]["visible_flags"]
        self._train_matches = cfg["dataset"]["train"]["matches"]
        self._test_matches = cfg["dataset"]["test"]["matches"]
        self._train_num_clip_ratio = cfg["dataset"]["train"]["num_clip_ratio"]
        self._test_num_clip_ratio = cfg["dataset"]["test"]["num_clip_ratio"]

        self._frames_in = cfg["model"]["frames_in"]
        self._frames_out = cfg["model"]["frames_out"]
        self._step = cfg["detector"]["step"]

        self._load_train = cfg["dataloader"]["train"]
        self._load_test = cfg["dataloader"]["test"]
        self._load_train_clip = cfg["dataloader"]["train_clip"]
        self._load_test_clip = cfg["dataloader"]["test_clip"]

        self._train_all = []
        self._train_clips = {}
        self._train_clip_gts = {}
        self._train_clip_disps = {}
        if self._load_train or self._load_train_clip:
            train_outputs = self._gen_seq_list(self._train_matches, self._train_num_clip_ratio)
            self._train_all = train_outputs["seq_list"]
            if self._load_train_clip:
                self._train_clips = train_outputs["clip_seq_list_dict"]
                self._train_clip_gts = train_outputs["clip_seq_gt_dict_dict"]
                self._train_clip_disps = train_outputs["clip_seq_disps"]

        self._test_all = []
        self._test_clips = {}
        self._test_clip_gts = {}
        self._test_clip_disps = {}
        if self._load_test or self._load_test_clip:
            test_outputs = self._gen_seq_list(self._test_matches, self._test_num_clip_ratio)
            self._test_all = test_outputs["seq_list"]
            if self._load_test_clip:
                self._test_clips = test_outputs["clip_seq_list_dict"]
                self._test_clip_gts = test_outputs["clip_seq_gt_dict_dict"]
                self._test_clip_disps = test_outputs["clip_seq_disps"]

        log.info("=> Pickleball loaded")

    def _gen_seq_list(self, matches, num_clip_ratio):
        seq_list = []
        clip_seq_list_dict = {}
        clip_seq_gt_dict_dict = {}
        clip_seq_disps = {}
        num_frames = 0
        num_matches = len(matches)
        num_rallies = 0
        num_frames_with_gt = 0
        disps = []
        for match in matches:
            match_clip_dir = osp.join(self._root_dir, match)
            clip_names = [name for name in os.listdir(match_clip_dir) if osp.isdir(osp.join(match_clip_dir, name))]
            clip_names.sort()
            clip_names = clip_names[: int(len(clip_names) * num_clip_ratio)]
            num_rallies += len(clip_names)
            for clip_name in clip_names:
                clip_seq_list = []
                clip_seq_gt_dict = {}
                clip_frame_dir = osp.join(self._root_dir, match, clip_name)
                clip_csv_path = osp.join(self._root_dir, match, clip_name, self._csv_filename)
                ball_xyvs = load_csv(clip_csv_path, self._visible_flags, frame_dir=clip_frame_dir)
                frame_names = [name for name in os.listdir(clip_frame_dir) if name.endswith(self._ext)]
                frame_names.sort()
                frame_fids = [int(osp.splitext(name)[0]) for name in frame_names]
                num_frames += len(frame_names)
                num_frames_with_gt += len(ball_xyvs)

                for i in range(len(frame_names) - self._frames_in + 1):
                    names = frame_names[i : i + self._frames_in]
                    paths = [osp.join(clip_frame_dir, name) for name in names]
                    anno_fids = frame_fids[i + self._frames_in - self._frames_out : i + self._frames_in]
                    annos = [ball_xyvs[fid] for fid in anno_fids]
                    seq_list.append({"frames": paths, "annos": annos, "match": match, "clip": clip_name})
                    if i % self._step == 0:
                        clip_seq_list.append({"frames": paths, "annos": annos, "match": match, "clip": clip_name})

                clip_disps = []
                for first_fid, second_fid in zip(frame_fids, frame_fids[1:]):
                    xy1, visi1 = ball_xyvs[first_fid]["center"].xy, ball_xyvs[first_fid]["center"].is_visible
                    xy2, visi2 = ball_xyvs[second_fid]["center"].xy, ball_xyvs[second_fid]["center"].is_visible
                    if visi1 and visi2:
                        disp = np.linalg.norm(np.array(xy1) - np.array(xy2))
                        disps.append(disp)
                        clip_disps.append(disp)

                for frame_name, fid in zip(frame_names, frame_fids):
                    path = osp.join(clip_frame_dir, frame_name)
                    clip_seq_gt_dict[path] = ball_xyvs[fid]["center"]

                clip_seq_list_dict[(match, clip_name)] = clip_seq_list
                clip_seq_gt_dict_dict[(match, clip_name)] = clip_seq_gt_dict
                clip_seq_disps[(match, clip_name)] = clip_disps

        disp_array = np.array(disps)
        return {
            "seq_list": seq_list,
            "clip_seq_list_dict": clip_seq_list_dict,
            "clip_seq_gt_dict_dict": clip_seq_gt_dict_dict,
            "clip_seq_disps": clip_seq_disps,
            "num_frames": num_frames,
            "num_frames_with_gt": num_frames_with_gt,
            "num_matches": num_matches,
            "num_rallies": num_rallies,
            "disp_mean": float(np.mean(disp_array)) if len(disp_array) else 0.0,
            "disp_std": float(np.std(disp_array)) if len(disp_array) else 0.0,
        }

    @property
    def train(self):
        return self._train_all

    @property
    def test(self):
        return self._test_all

    @property
    def train_clips(self):
        return self._train_clips

    @property
    def train_clip_gts(self):
        return self._train_clip_gts

    @property
    def test_clips(self):
        return self._test_clips

    @property
    def test_clip_gts(self):
        return self._test_clip_gts
