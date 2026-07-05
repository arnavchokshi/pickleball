# pickleball addition, not upstream
import logging
import os
import os.path as osp
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Center:
    x: float
    y: float
    is_visible: bool

    @property
    def xy(self):
        return (self.x, self.y)


def load_csv(csv_path, fids, frame_dir=None, ext=".png"):
    with open(csv_path, "r") as f:
        data = f.read().split("\n")
    xyvs = {}
    for i in range(len(data)):
        if data[i] == "":
            break
        es = data[i].split(" ")
        x, y = float(es[0]), float(es[1])
        visi = not (x == 0 and y == 0)
        frame_path = None
        if frame_dir is not None:
            frame_path = osp.join(frame_dir, "{}{}".format(fids[i], ext))
        xyvs[i] = {"center": Center(x=x, y=y, is_visible=visi), "frame_path": frame_path}
    return xyvs


def get_clips(cfg, train_or_test="test", gt=True):
    root_dir = cfg["dataset"]["root_dir"]
    frame_dirname = cfg["dataset"]["frame_dirname"]
    csv_dirname = cfg["dataset"]["csv_dirname"]
    matches = cfg["dataset"][train_or_test]["matches"]
    ext = cfg["dataset"]["ext"]
    clip_dict = {}
    for match in matches:
        match_clip_dir = osp.join(root_dir, frame_dirname, "{}".format(match))
        clip_names = []
        for clip_name in os.listdir(match_clip_dir):
            if osp.isdir(os.path.join(match_clip_dir, clip_name)):
                clip_names.append(clip_name)
        for clip_name in clip_names:
            clip_frame_dir = osp.join(root_dir, frame_dirname, "{}".format(match), clip_name)
            clip_csv_path = osp.join(root_dir, csv_dirname, "{}".format(match), "{}.txt".format(clip_name))
            fids = []
            for frame_name in os.listdir(clip_frame_dir):
                if frame_name.endswith(ext):
                    fids.append(int(osp.splitext(frame_name)[0]))
            fids.sort()
            ball_xyvs = load_csv(clip_csv_path, fids, ext=ext) if gt else None
            frame_names = ["{}{}".format(fid, ext) for fid in fids]
            clip_dict[(match, clip_name)] = {
                "clip_dir_or_path": clip_frame_dir,
                "clip_gt_dict": ball_xyvs,
                "frame_names": frame_names,
            }
    return clip_dict


class Pickleball(object):
    def __init__(self, cfg):
        self._root_dir = cfg["dataset"]["root_dir"]
        self._frame_dirname = cfg["dataset"]["frame_dirname"]
        self._csv_dirname = cfg["dataset"]["csv_dirname"]
        self._ext = cfg["dataset"]["ext"]
        self._train_matches = cfg["dataset"]["train"]["matches"]
        self._val_matches = cfg["dataset"].get("val", {}).get("matches", [])
        self._test_matches = cfg["dataset"]["test"]["matches"]

        self._train_num_clip_ratio = cfg["dataset"]["train"]["num_clip_ratio"]
        self._val_num_clip_ratio = cfg["dataset"].get("val", {}).get("num_clip_ratio", 1.0)
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

        self._val_all = []
        self._val_clips = {}
        self._val_clip_gts = {}
        self._val_clip_disps = {}
        if self._val_matches:
            val_outputs = self._gen_seq_list(self._val_matches, self._val_num_clip_ratio)
            self._val_all = val_outputs["seq_list"]
            self._val_clips = val_outputs["clip_seq_list_dict"]
            self._val_clip_gts = val_outputs["clip_seq_gt_dict_dict"]
            self._val_clip_disps = val_outputs["clip_seq_disps"]

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
        num_clips_no_ball = 0
        for match in matches:
            match_clip_dir = osp.join(self._root_dir, self._frame_dirname, "{}".format(match))
            clip_names = []
            for clip_name in os.listdir(match_clip_dir):
                if osp.isdir(os.path.join(match_clip_dir, clip_name)):
                    clip_names.append(clip_name)
            clip_names = clip_names[: int(len(clip_names) * num_clip_ratio)]
            num_rallies += len(clip_names)
            for clip_name in clip_names:
                clip_seq_list = []
                clip_seq_gt_dict = {}
                clip_frame_dir = osp.join(self._root_dir, self._frame_dirname, "{}".format(match), clip_name)
                clip_csv_path = osp.join(self._root_dir, self._csv_dirname, "{}".format(match), "{}.txt".format(clip_name))
                fids = []
                for frame_name in os.listdir(clip_frame_dir):
                    if frame_name.endswith(self._ext):
                        fids.append(int(osp.splitext(frame_name)[0]))
                fids.sort()
                ball_xyvs = load_csv(clip_csv_path, fids, frame_dir=clip_frame_dir, ext=self._ext)

                no_ball = True
                for fid, xyv in ball_xyvs.items():
                    if xyv["center"].is_visible:
                        no_ball = False
                if no_ball:
                    num_clips_no_ball += 1

                num_frames += len(fids)
                num_frames_with_gt += len(ball_xyvs)
                for i in range(len(ball_xyvs) - self._frames_in + 1):
                    inds = fids[i : i + self._frames_in]
                    names = ["{}{}".format(ind, self._ext) for ind in inds]
                    paths = [osp.join(clip_frame_dir, name) for name in names]
                    annos = [ball_xyvs[j] for j in range(i, i + self._frames_in)]
                    seq_list.append({"frames": paths, "annos": annos, "match": match, "clip": clip_name})
                    if i % self._step == 0:
                        clip_seq_list.append({"frames": paths, "annos": annos, "match": match, "clip": clip_name})
                clip_seq_list_dict[(match, clip_name)] = clip_seq_list

                clip_disps = []
                for i in range(len(ball_xyvs) - 1):
                    xy1, visi1 = ball_xyvs[i]["center"].xy, ball_xyvs[i]["center"].is_visible
                    xy2, visi2 = ball_xyvs[i + 1]["center"].xy, ball_xyvs[i + 1]["center"].is_visible
                    if visi1 and visi2:
                        disp = np.linalg.norm(np.array(xy1) - np.array(xy2))
                        disps.append(disp)
                        clip_disps.append(disp)

                for i in range(len(ball_xyvs)):
                    path = osp.join(clip_frame_dir, "{}{}".format(fids[i], self._ext))
                    clip_seq_gt_dict[path] = ball_xyvs[i]["center"]

                clip_seq_gt_dict_dict[(match, clip_name)] = clip_seq_gt_dict
                clip_seq_disps[(match, clip_name)] = clip_disps

        log.info("{}/{} clips do not include ball trajectory".format(num_clips_no_ball, len(clip_seq_list_dict)))
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
    def val(self):
        return self._val_all

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
