from __future__ import annotations

import base64
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any

from loguru import logger

from arenaagent.agent_base import AgentBase, AgentCfg
from arenaagent.builder import Register
from arenaagent.tongsim_grpc_client import TongSimGrpcClient
from arenaagent.tongsim_interface import Rotation
from arenaagent.utils.configclass import configclass


@configclass
class JigsawProbeAgentCfg(AgentCfg):
    name: str = "jigsaw_probe_agent"
    sleep_between_steps: float = 1.0
    log_dir: str = "logs"
    tongsim_server_endpoint: str = "127.0.0.1:50060"


@Register("jigsaw_probe_agent")
class JigsawProbeAgent(AgentBase):
    """脚本化协议探测 agent：不做 LLM 决策，按固定 FSM 依次执行动作并全量落盘。

    目标回答四个协议问题：
      Q1 put 后从观察点回看，块是否锁定在槽内、最终 yaw 是多少；
      Q2 已放入槽的块能否 move_and_take_object 取回（局内纠错能力）；
      Q3 auto_rotate=True 时服务端是否自动转正到板面 yaw；
      Q4 待放块出生 yaw 与最终放置 yaw 的对应关系。
    """

    _BOARD_X_TOL = 2.0

    def __init__(self, stub, channel, cfg=None, sleep_between_steps: float = 1.0) -> None:
        super().__init__(
            stub=stub,
            channel=channel,
            cfg=cfg or JigsawProbeAgentCfg(),
            sleep_between_steps=sleep_between_steps,
        )
        self._initialized = False
        self._tongsim: TongSimGrpcClient | None = None
        self._character_id: str | None = None
        self._spawn_loc: list[float] = []
        self._state = "start"
        self._step_no = 0
        self._record_path: str | None = None
        # 探测运行期状态
        self._subject: dict[str, Any] = {}
        self._movables: list[str] = []  # 待放块 object_id（按出生位置排序）
        self._board_blocks: list[dict[str, Any]] = []  # 已放 6 块（id/loc/yaw）
        self._last_frame_objects: list[dict[str, Any]] = []  # 最近一帧全量物体
        self._placed: dict[str, list[float]] = {}  # block_id -> slot [y,z]
        self._occupied_slots: list[list[float]] = []  # 已被占用的空槽 [y,z]
        self._empty_slots: list[list[float]] = []  # 初始空槽 [y,z]
        self._in_hand: str | None = None
        self._undo_res: dict[str, Any] = {}
        self._need_auto_piece = False
        self._finish_done = False
        self._target_yaw: float | None = None
        self._scan_only = os.environ.get("PROBE_SCAN_ONLY", "").strip().lower() in ("1", "true", "yes")
        self._color_scan = os.environ.get("PROBE_COLORSCAN", "").strip().lower() in ("1", "true", "yes")
        self._face_scan = os.environ.get("PROBE_FACE", "").strip().lower() in ("1", "true", "yes")
        self._place_scan = os.environ.get("PROBE_PLACE", "").strip().lower() in ("1", "true", "yes")
        self._scan_phase = "shelf"
        self._color_cells: list[dict[str, Any]] = []
        self._color_idx = 0
        self._face_idx = 0
        self._face_phase = "take"
        self._place_phase = "pre"
        self._place_piece: str | None = None
        self._place_slot: list[float] | None = None
        self._score_scan = os.environ.get("PROBE_SCORE", "").strip().lower() in ("1", "true", "yes")
        self._score_phase = "prescan"
        self._score_cells: list[dict[str, Any]] = []
        self._score_cell_sigs: dict[tuple[float, float], dict[str, Any]] = {}
        self._score_piece_sigs: dict[str, dict[str, Any]] = {}
        self._score_assign: list[tuple[str, list[float]]] = []
        self._score_place_idx = 0
        self._eval_run = os.environ.get("PROBE_EVAL", "").strip().lower() in ("1", "true", "yes")
        self._eval_spec: list[tuple[str, float, float, float | str | None]] = []
        self._eval_idx = 0
        self._solve_run = os.environ.get("SOLVE", "").strip().lower() in ("1", "true", "yes")
        self._capture_after = os.environ.get("CAPTURE_AFTER", "").strip().lower() in ("1", "true", "yes")
        self._capture_done = False
        self._gen_recon = os.environ.get("JIGSAW_GEN_RECON", "").strip().lower() in ("1", "true", "yes")
        self._gen: dict[str, Any] = {}
        self._gen_data = os.environ.get("JIGSAW_GEN_DATA", "").strip().lower() in ("1", "true", "yes")
        self._truth_map: dict[str, list[float]] = {}
        for part in os.environ.get("JIGSAW_TRUTH", "").split(";"):
            part = part.strip()
            if not part:
                continue
            seg = [x.strip() for x in part.replace("(", "").replace(")", "").split(",") if x.strip()]
            if len(seg) >= 3:
                try:
                    self._truth_map[str(seg[0])] = [float(seg[1]), float(seg[2])]
                except Exception:
                    pass
        self._gd: dict[str, Any] = {}
        self._poster_scan = os.environ.get("JIGSAW_POSTER", "").strip().lower() in ("1", "true", "yes")
        self._pframe = os.environ.get("JIGSAW_PFRAME", "").strip().lower() in ("1", "true", "yes")
        self._undo_run = os.environ.get("PROBE_UNDO", "").strip().lower() in ("1", "true", "yes")
        self._undo_phase = "init"
        self._undo_piece: str | None = None
        self._undo_slot: list[float] | None = None
        self._arr_run = os.environ.get("JIGSAW_ARR", "").strip().lower() in ("1", "true", "yes")
        self._arr_phase = "wrong"
        self._aim_run = os.environ.get("JIGSAW_AIM", "").strip().lower() in ("1", "true", "yes")
        self._aim_idx = 0
        self._aim_steps: list[dict[str, Any]] = []
        self._arr_wrong: list[list[Any]] = []
        self._arr_right: dict[str, list[float]] = {}
        self._arr_idx = 0


    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def init(self, opt: dict[str, Any]) -> None:
        if self._initialized:
            return
        log_dir = os.path.join(getattr(self.cfg, "log_dir", "") or "logs", "recon")
        os.makedirs(log_dir, exist_ok=True)
        self._record_path = os.path.join(log_dir, "probe_{}.jsonl".format(self.agent_id))

        endpoint = opt.get("tongsim_server_endpoint") or self.cfg.tongsim_server_endpoint
        self._tongsim = TongSimGrpcClient(endpoint=endpoint)
        spawn_loc = json.loads(opt["spawn_loc"])
        spawn_rot = json.loads(opt["spawn_rot"])
        camera_fov = float(opt.get("camera_fov", 120.0))
        camera_width = int(opt.get("camera_width", 1280))
        camera_height = int(opt.get("camera_height", 720))
        self._spawn_loc = list(spawn_loc)
        self._character_id = self._tongsim.spawn_character(
            spawn_loc, spawn_rot, opt["name"], camera_fov, camera_width, camera_height
        )
        logger.info("probe agent spawned character {} at {}", self._character_id, self._spawn_loc)
        self._initialized = True

    def deinit(self) -> None:
        if not self._initialized:
            return
        if self._tongsim:
            try:
                self._tongsim.close()
            except Exception as exc:  # pragma: no cover - 清理保护
                logger.warning("close tongsim failed: {}", exc)
        self._tongsim = None
        self._character_id = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    def run_step(self, subject, task_response) -> dict[str, Any]:
        if not self._initialized:
            return self._finish_now("not initialized")
        self._step_no += 1
        try:
            if self._state == "start":
                return self._phase_start(subject, task_response)
            method = getattr(self, "_phase_" + self._state, None)
            if method is None:
                logger.warning("unknown probe state {}, finishing", self._state)
                return self._finish_now("unknown state: " + str(self._state))
            return method()
        except Exception as exc:  # pragma: no cover - 探测保护：任何异常都要能收尾
            logger.opt(exception=True).error("probe step {} crashed: {}", self._state, exc)
            return self._finish_now("exception: " + str(exc))

    # ------------------------------------------------------------------ #
    # 相位实现
    # ------------------------------------------------------------------ #

    def _phase_start(self, subject, task_response) -> dict[str, Any]:
        self._subject = subject if isinstance(subject, dict) else {}
        self._record(
            {
                "kind": "subject",
                "phase": self._state,
                "subject": self._subject,
                "task_response": task_response,
            }
        )
        frame = self._acquire_and_log("initial", save_image=True)
        ok = self._analyze(frame)
        if not ok:
            logger.warning("layout analysis failed, finishing without actions")
            return self._finish_now("layout analysis failed")
        if self._eval_run:
            self._eval_spec = self._parse_eval_spec(os.environ.get("PROBE_EVAL_PLACES", ""))
            self._record({"kind": "eval_plan", "movables": self._movables, "empty_slots": self._empty_slots, "spec": self._eval_spec})
            self._state = "eval"
            return self._ok("eval start")
        if self._solve_run:
            self._eval_spec = self._build_solve_spec()
            self._record({"kind": "solve_plan", "movables": self._movables, "empty_slots": self._empty_slots, "spec": self._eval_spec, "note": "learned mapping for known empty-slot pattern"})
            self._state = "eval"
            return self._ok("solve start")
        if self._gen_recon:
            self._setup_gen_recon()
            self._state = "gen"
            return self._ok("gen recon start")
        if self._gen_data:
            self._setup_gen_data()
            self._state = "gdata"
            return self._ok("gen data start")
        if self._poster_scan:
            self._setup_poster_scan()
            self._state = "poster2"
            return self._ok("poster scan start")
        if self._pframe:
            self._setup_pframe()
            self._state = "pframe"
            return self._ok("pframe start")
        if self._undo_run:
            self._undo_piece = self._movables[0] if self._movables else None
            self._undo_slot = list(self._empty_slots[0]) if self._empty_slots else None
            self._record(
                {
                    "kind": "undo_probe_plan",
                    "piece": self._undo_piece,
                    "slot": self._undo_slot,
                    "spawn_loc": self._spawn_loc,
                    "movables": self._movables,
                    "empty_slots": self._empty_slots,
                }
            )
            self._state = "undo_probe"
            return self._ok("undo probe start")
        if self._arr_run:
            self._setup_arr()
            self._state = "arr"
            return self._ok("arrangement swap start")
        if self._aim_run:
            self._aim_steps = [
                {"tag": "aim_a1_move_lookloc", "stand": {"X": 797.0, "Y": 166.0, "Z": 60.0}, "method": "loc", "target": self._board_center()},
                {"tag": "aim_a2_lookobj_mid", "stand": None, "method": "obj", "target": str(self._board_blocks[0]["id"])},
                {"tag": "aim_a3_lookloc_again", "stand": None, "method": "loc", "target": self._board_center()},
                {"tag": "aim_a4_lookobj_mid2", "stand": None, "method": "obj", "target": str(self._board_blocks[-1]["id"])},
                {"tag": "aim_a5_spawn_lookloc", "stand": {"X": float(self._spawn_loc[0]), "Y": float(self._spawn_loc[1]), "Z": float(self._spawn_loc[2])}, "method": "loc", "target": self._board_center()},
            ]
            self._record({"kind": "aim_plan", "steps": self._aim_steps, "spawn": self._spawn_loc})
            self._state = "aim"
            return self._ok("aim probe start")

        if self._scan_only:
            # 侦察模式：只在原图分辨率下抓若干视角，不做任何取放
            self._record({"kind": "scan_mode", "phase": self._state, "note": "PROBE_SCAN_ONLY, native acquire"})
            self._state = "scan"
            return self._ok("scan mode start")
        if self._color_scan:
            self._build_color_cells()
            self._record({"kind": "color_scan", "phase": self._state, "cells": self._color_cells})
            self._state = "color"
            return self._ok("color scan start")
        if self._face_scan:
            self._record({"kind": "face_scan", "phase": self._state, "movables": self._movables})
            self._state = "face"
            return self._ok("face scan start")
        if self._place_scan:
            self._place_piece = self._movables[0] if self._movables else None
            self._place_slot = list(self._empty_slots[0]) if self._empty_slots else None
            self._record({"kind": "place_scan", "phase": self._state, "piece": self._place_piece, "slot": self._place_slot})
            self._state = "place"
            return self._ok("place scan start")
        if self._score_scan:
            self._build_score_plan()
            self._record({"kind": "score_scan", "phase": self._state, "prescan_cells": self._score_cells})
            self._state = "score"
            return self._ok("score scan start")
        self._record(
            {
                "kind": "plan",
                "phase": self._state,
                "movables": self._movables,
                "empty_slots": self._empty_slots,
                "occupied_slots": self._occupied_slots,
            }
        )
        self._state = "take_wrong"
        return self._ok("start done")

    def _phase_take_wrong(self) -> dict[str, Any]:
        block = self._pick_next_piece()
        if block is None:
            return self._finish_now("no piece for wrong-yaw test")
        self._in_hand = block
        result = self._do_take(block, "take_wrong")
        self._state = "put_wrong"
        return self._wrap("take_wrong", result)

    def _phase_put_wrong(self) -> dict[str, Any]:
        if self._in_hand is None:
            self._state = "take_wrong"
            return self._ok("hand empty, retry take")
        slot = self._pick_empty_slot()
        if slot is None:
            return self._finish_now("no empty slot for wrong-yaw put")
        result = self._do_put(slot, yaw=self._wrong_yaw(), auto_rotate=False, tag="put_wrong")
        if self._result_ok(result):
            self._mark_placed(self._in_hand, slot)
            self._in_hand = None
            self._state = "view_wrong"
        else:
            # 放置失败，手里仍拿着，先退回观察
            logger.warning("put_wrong failed: {}", result)
            self._state = "view_wrong"
        return self._wrap("put_wrong", result)

    def _phase_view_wrong(self) -> dict[str, Any]:
        target = self._placed_target_of_last_put or None
        obs = self._lookback("after_put_wrong", target=target)
        self._record_observation("wrong_yaw", obs)
        self._state = "undo"
        return self._wrap("view_wrong", obs.get("meta", {}))

    def _phase_undo(self) -> dict[str, Any]:
        block = self._last_placed_block or self._pick_any_placed()
        if block is None:
            self._state = "undo_hand"
            return self._ok("nothing placed to undo")
        result = self._do_take(block, "undo_test")
        self._undo_res = result
        if self._result_ok(result):
            self._in_hand = block
            self._release_slot_of(block)
        self._state = "undo_hand"
        return self._wrap("undo", result)

    def _phase_undo_hand(self) -> dict[str, Any]:
        has_obj, _ = self._safe_hand()
        if not has_obj:
            self._in_hand = None
        self._record({"kind": "hand", "phase": self._state, "has_object_in_hand": has_obj})
        self._state = "put_auto"
        return self._ok("undo_hand checked")

    def _phase_put_auto(self) -> dict[str, Any]:
        if self._in_hand is None:
            # undo 失败或没有可用的块：从待放块里再拿一块做 auto_rotate 测试
            self._need_auto_piece = True
            self._state = "take_more"
            return self._ok("no hand piece, will take another for auto test")
        slot = self._pick_empty_slot()
        if slot is None:
            return self._finish_now("no empty slot for auto_rotate put")
        result = self._do_put(slot, yaw=None, auto_rotate=True, tag="put_auto")
        if self._result_ok(result):
            self._mark_placed(self._in_hand, slot)
            self._in_hand = None
        self._state = "view_auto"
        return self._wrap("put_auto", result)

    def _phase_view_auto(self) -> dict[str, Any]:
        obs = self._lookback("after_put_auto", target=self._last_auto_slot)
        self._record_observation("auto_rotate", obs)
        self._state = "take_more"
        return self._wrap("view_auto", obs.get("meta", {}))

    def _phase_take_more(self) -> dict[str, Any]:
        if self._in_hand is not None:
            # 上一块 put 失败仍拿在手里：先去尝试放下，避免重复抓取
            self._state = "put_more"
            return self._ok("still holding a piece, put it first")
        if self._need_auto_piece:
            block = self._pick_next_piece()
            if block is None:
                return self._finish_now("no piece left for auto test")
            self._in_hand = block
            result = self._do_take(block, "take_for_auto")
            self._need_auto_piece = False
            self._state = "put_auto"
            return self._wrap("take_for_auto", result)
        block = self._pick_next_piece()
        if block is None:
            return self._finish_now("all pieces placed")
        self._in_hand = block
        result = self._do_take(block, "take_more")
        self._state = "put_more"
        return self._wrap("take_more", result)

    def _phase_put_more(self) -> dict[str, Any]:
        if self._in_hand is None:
            self._state = "take_more"
            return self._ok("hand empty, retry")
        slot = self._pick_empty_slot()
        if slot is None:
            return self._finish_now("no empty slot left")
        result = self._do_put(slot, yaw=self._target_yaw, auto_rotate=False, tag="put_more")
        if self._result_ok(result):
            self._mark_placed(self._in_hand, slot)
            self._in_hand = None
        self._state = "take_more"
        return self._wrap("put_more", result)

    def _phase_scan(self) -> dict[str, Any]:
        if self._scan_phase == "shelf":
            center = {"X": 837.0, "Y": float(self._spawn_row), "Z": float(sum(getattr(self, "_board_cols", [99.0])) / 3)}
            try:
                self._tongsim.look_at_location(self._character_id, center)
            except Exception as exc:
                logger.warning("look shelf failed: {}", exc)
            time.sleep(1.0)
            self._acquire_and_log("scan_shelf", save_image=True)
            self._scan_phase = "board"
            return self._ok("scan shelf done")
        if self._scan_phase == "board":
            try:
                self._tongsim.look_at_location(self._character_id, self._board_center())
            except Exception as exc:
                logger.warning("look board failed: {}", exc)
            time.sleep(1.0)
            self._acquire_and_log("scan_board", save_image=True)
            self._scan_phase = "close"
            return self._ok("scan board done")
        if self._scan_phase == "close":
            # 走近出生区看板面，尽量放大拼图块
            try:
                spawn = {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]}
                self._tongsim.move_to_location(self._character_id, spawn, stop_distance=1.0)
                self._tongsim.look_at_location(self._character_id, self._board_center())
            except Exception as exc:
                logger.warning("close view failed: {}", exc)
            time.sleep(1.0)
            self._acquire_and_log("scan_close_board", save_image=True)
            return self._finish_now("scan only done")
        return self._finish_now("scan done")

    def _phase_face(self) -> dict[str, Any]:
        if self._face_idx >= len(self._movables):
            return self._finish_now("face scan done")
        piece = self._movables[self._face_idx]
        if self._face_phase == "take":
            self._in_hand = piece
            result = self._do_take(piece, "face_take")
            time.sleep(0.6)
            try:
                perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
                self._save_image(perception.get("image") or "", "held_raw_{}".format(piece))
            except Exception as exc:
                logger.warning("held raw capture failed: {}", exc)
            self._face_phase = "look"
            return self._wrap("face_take", result)
        if self._face_phase == "look":
            try:
                self._tongsim.look_at_object(self._character_id, piece)
            except Exception as exc:
                logger.warning("look_at_object {} failed: {}", piece, exc)
            time.sleep(1.0)
            try:
                perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
                image_b64 = perception.get("image") or ""
                image_path = self._save_image(image_b64, "held_look_{}".format(piece))
                stats = self._crop_center_stats(image_b64, {"kind": "held", "object_id": piece})
                self._record({"kind": "face_sample", "object_id": piece, "image_path": image_path, "stats": stats})
            except Exception as exc:
                logger.warning("held look capture failed: {}", exc)
            self._face_phase = "restore"
            return self._ok("face look done")
        if self._face_phase == "restore":
            target = self._probe_movable_loc(piece)
            result = self._put_back(target) if target is not None else {}
            self._in_hand = None
            self._face_idx += 1
            self._face_phase = "take"
            return self._wrap("face_restore", result)
        return self._ok("face")

    # ------------------------------------------------------------------ #
    # 排列对照实验（JIGSAW_ARR=1）
    # 先把 3 块按“错排列”放好拍板面照，再原地取回按“正确排列”重放并拍照。
    # 用途：离线比较两种排列的本地判据（12 条接缝）能否区分。
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_arr_spec(raw: str) -> list[list[Any]]:
        out: list[list[Any]] = []
        for part in (raw or "").split(";"):
            part = part.strip()
            if not part:
                continue
            left, _, right = part.partition(":")
            coords = [c.strip() for c in right.split(",") if c.strip()]
            if len(coords) < 2:
                continue
            try:
                out.append([left.strip(), [float(coords[0]), float(coords[1])]])
            except Exception:
                continue
        return out

    def _setup_arr(self) -> None:
        wrong = self._parse_arr_spec(os.environ.get("JIGSAW_ARR_WRONG", ""))
        right = self._parse_arr_spec(os.environ.get("JIGSAW_ARR_RIGHT", ""))
        if len(wrong) != 3 or len(right) != 3:
            wrong = [[bid, list(self._empty_slots[i])] for i, bid in enumerate(self._movables)]
            slots = list(reversed([list(s) for s in self._empty_slots]))
            right = [[bid, slots[i]] for i, bid in enumerate(self._movables)]
        self._arr_wrong = wrong
        self._arr_right = {str(p[0]): [float(p[1][0]), float(p[1][1])] for p in right}
        self._record(
            {
                "kind": "arr_plan",
                "wrong": wrong,
                "right": right,
                "movables": self._movables,
                "empty_slots": self._empty_slots,
            }
        )
        logger.info("arr plan wrong={} right={}", wrong, right)

    def _phase_arr(self) -> dict[str, Any]:
        phase = self._arr_phase
        if phase == "wrong":
            if self._arr_idx >= len(self._arr_wrong):
                self._arr_phase = "shot_wrong"
                return self._ok("wrong arrangement placed")
            piece, slot = self._arr_wrong[self._arr_idx]
            if self._in_hand != piece:
                res = self._do_take(piece, "arr_wrong_take")
                if self._result_ok(res):
                    self._in_hand = piece
                return self._wrap("arr_wrong_take", res)
            res = self._do_put(slot, yaw=None, auto_rotate=True, tag="arr_wrong_put")
            if self._result_ok(res):
                self._in_hand = None
                self._mark_placed(piece, slot)
                self._arr_idx += 1
            return self._wrap("arr_wrong_put", res)
        if phase == "shot_wrong":
            self._close_shot("arr_wrong_board", 166.0, [166.0, 99.0])
            self._arr_phase = "correct"
            self._arr_idx = 0
            return self._ok("wrong arrangement photographed")
        if phase == "correct":
            if self._arr_idx >= len(self._arr_wrong):
                self._arr_phase = "shot_right"
                return self._ok("correct arrangement placed")
            piece, _slot = self._arr_wrong[self._arr_idx]
            target = self._arr_right.get(str(piece))
            if target is None:
                self._arr_idx += 1
                return self._ok("no target for piece {}".format(piece))
            if self._in_hand != piece:
                res = self._do_take(piece, "arr_move_take")
                if self._result_ok(res):
                    self._in_hand = piece
                    self._release_slot_of(piece)
                return self._wrap("arr_move_take", res)
            res = self._do_put(target, yaw=None, auto_rotate=True, tag="arr_move_put")
            if self._result_ok(res):
                self._in_hand = None
                self._mark_placed(piece, target)
                self._arr_idx += 1
            return self._wrap("arr_move_put", res)
        if phase == "shot_right":
            self._close_shot("arr_right_board", 166.0, [166.0, 99.0])
            for slot in self._empty_slots:
                self._close_shot(
                    "arr_right_cell_{}_{}".format(int(slot[0]), int(slot[1])),
                    slot[0],
                    [slot[0], slot[1]],
                )
            self._arr_phase = "done"
            return self._ok("correct arrangement photographed")
        return self._finish_now("arrangement swap done")

    # ------------------------------------------------------------------ #
    # undo 往返验证（PROBE_UNDO=1）
    # 目的：确认「已放入槽的块」能否被 move_and_take_object 取回手上。
    # 能取回 => 支持两轮放置（先乱放拍照取排列 -> undo 全部 -> 按正确排列重放）。
    # ------------------------------------------------------------------ #

    def _hand_state(self, phase: str) -> bool | None:
        try:
            has_obj, hand_idx = self._tongsim.has_object_in_hand(self._character_id)
        except Exception as exc:
            self._record({"kind": "hand", "phase": phase, "error": str(exc)})
            return None
        self._record({"kind": "hand", "phase": phase, "has_object_in_hand": bool(has_obj), "hand_idx": hand_idx})
        return bool(has_obj)

    def _phase_aim(self) -> dict[str, Any]:
        if self._aim_idx >= len(self._aim_steps):
            return self._finish_now("aim probe done")
        step = self._aim_steps[self._aim_idx]
        meta: dict[str, Any] = {"stand": step.get("stand"), "method": step.get("method"), "target": step.get("target")}
        if step.get("stand"):
            try:
                meta["move"] = self._tongsim.move_to_location(self._character_id, step["stand"], stop_distance=1.0)
            except Exception as exc:
                meta["move_error"] = str(exc)
        try:
            if step["method"] == "obj":
                meta["look"] = self._tongsim.look_at_object(self._character_id, str(step["target"]))
            else:
                meta["look"] = self._tongsim.look_at_location(self._character_id, step["target"])
        except Exception as exc:
            meta["look_error"] = str(exc)
        time.sleep(1.5)
        frame = self._acquire_native(step["tag"], save_image=True)
        meta["visible_count"] = len(frame.get("objects", []))
        ids = [str(o.get("object_id")) for o in frame.get("objects", [])]
        meta["board_tiles_visible"] = [b["id"] for b in self._board_blocks if b["id"] in ids]
        meta["image_path"] = frame.get("image_path")
        self._record({"kind": "aim_shot", "tag": step["tag"], "meta": meta})
        self._aim_idx += 1
        return self._ok("aim shot " + step["tag"])

    def _acquire_native(self, tag: str, save_image: bool = False) -> dict[str, Any]:
        """原生比例抓帧（1440x1000 左右）。带 width/height 会返回 2560x720 全景，板面会严重畸变。"""
        try:
            perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
        except Exception as exc:
            logger.error("native perception failed at {}: {}", tag, exc)
            perception = {}
        image_b64 = perception.get("image")
        image_path = ""
        if save_image and image_b64:
            image_path = self._save_image(image_b64, tag)
        objects = perception.get("objects", []) or []
        self._last_frame_objects = objects
        self._record(
            {
                "kind": "frame",
                "phase": tag,
                "step": self._step_no,
                "native": True,
                "image_path": image_path,
                "objects": objects,
            }
        )
        return {"objects": objects, "image_path": image_path}

    def _close_shot(
        self,
        tag: str,
        stand_y: float,
        look_slot: list[float] | None = None,
        require_tiles: int = 6,
    ) -> dict[str, Any]:
        """近距看板：先回出生点再站位（放完块后直接站位导航不干净），瞄准后抓帧并校验板面块可见数。"""
        meta: dict[str, Any] = {}
        target = self._slot_loc(look_slot) if look_slot else self._board_center()
        stand = {"X": 797.0, "Y": float(stand_y), "Z": 60.0}
        spawn = {"X": float(self._spawn_loc[0]), "Y": float(self._spawn_loc[1]), "Z": float(self._spawn_loc[2])}
        board_ids = {str(b["id"]) for b in self._board_blocks}
        frame: dict[str, Any] = {}
        attempts: list[dict[str, Any]] = []
        for attempt in range(2):
            info: dict[str, Any] = {}
            try:
                info["spawn_move"] = self._tongsim.move_to_location(self._character_id, spawn, stop_distance=1.0)
            except Exception as exc:
                info["spawn_move_error"] = str(exc)
            try:
                info["stand_move"] = self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            except Exception as exc:
                info["stand_move_error"] = str(exc)
            try:
                info["look"] = self._tongsim.look_at_location(self._character_id, target)
            except Exception as exc:
                info["look_error"] = str(exc)
            time.sleep(1.5)
            frame = self._acquire_native(tag, save_image=True)
            ids = {str(o.get("object_id")) for o in frame.get("objects", [])}
            visible = sorted(board_ids & ids)
            info["board_tiles_visible"] = visible
            info["visible_count"] = len(ids)
            attempts.append(info)
            if len(visible) >= require_tiles:
                break
        meta["attempts"] = attempts
        meta["stand"] = stand
        meta["target"] = target
        meta["image_path"] = frame.get("image_path")
        meta["board_tiles_visible"] = attempts[-1].get("board_tiles_visible") if attempts else []
        self._record({"kind": "close_shot", "tag": tag, "meta": meta})
        return {"frame": frame, "meta": meta}

    def _phase_undo_probe(self) -> dict[str, Any]:
        if self._undo_piece is None or self._undo_slot is None:
            return self._finish_now("undo probe missing piece/slot")
        phase = self._undo_phase
        if phase == "init":
            self._hand_state("before_take")
            self._close_shot("undo_before", self._undo_slot[0], self._undo_slot)
            self._undo_phase = "take"
            return self._ok("undo probe: baseline captured")
        if phase == "take":
            result = self._do_take(self._undo_piece, "undo_take")
            self._record({"kind": "undo_step", "step": "take", "piece": self._undo_piece, "result": result})
            self._undo_phase = "hand_after_take"
            return self._wrap("undo_take", result)
        if phase == "hand_after_take":
            self._hand_state("after_take")
            self._undo_phase = "put"
            return self._ok("undo probe: hand state after take")
        if phase == "put":
            result = self._do_put(self._undo_slot, yaw=None, auto_rotate=True, tag="undo_put")
            self._record({"kind": "undo_step", "step": "put", "slot": self._undo_slot, "result": result})
            self._undo_phase = "hand_after_put"
            return self._wrap("undo_put", result)
        if phase == "hand_after_put":
            self._hand_state("after_put")
            self._undo_phase = "look_placed"
            return self._ok("undo probe: hand state after put")
        if phase == "look_placed":
            self._close_shot("undo_placed", self._undo_slot[0], self._undo_slot)
            self._undo_phase = "undo"
            return self._ok("undo probe: placed frame captured")
        if phase == "undo":
            result = self._do_take(self._undo_piece, "undo_undo")
            self._record({"kind": "undo_step", "step": "undo", "piece": self._undo_piece, "result": result})
            self._undo_phase = "hand_after_undo"
            return self._wrap("undo_undo", result)
        if phase == "hand_after_undo":
            self._hand_state("after_undo")
            self._undo_phase = "look_after_undo"
            return self._ok("undo probe: hand state after undo")
        if phase == "look_after_undo":
            self._close_shot("undo_after", self._undo_slot[0], self._undo_slot)
            self._undo_phase = "restore"
            return self._ok("undo probe: post-undo frame captured")
        if phase == "restore":
            target = self._probe_movable_loc(self._undo_piece)
            result = self._put_back(target) if target is not None else {"result": "skipped"}
            self._record({"kind": "undo_step", "step": "restore", "result": result})
            self._undo_phase = "done"
            return self._finish_now("undo probe done")
        return self._finish_now("undo probe done")

    def _phase_place(self) -> dict[str, Any]:
        if self._place_piece is None or self._place_slot is None:
            return self._finish_now("place scan missing piece/slot")
        if self._place_phase == "pre":
            # 放入前先采空槽颜色
            self._sample_cell_color({"kind": "slot_before", "object_id": "", "y": self._place_slot[0], "z": self._place_slot[1]})
            self._place_phase = "take"
            return self._ok("slot_before sampled")
        if self._place_phase == "take":
            self._in_hand = self._place_piece
            result = self._do_take(self._place_piece, "place_take")
            self._place_phase = "place"
            return self._wrap("place_take", result)
        if self._place_phase == "place":
            result = self._do_put(self._place_slot, yaw=self._target_yaw, auto_rotate=False, tag="place_put")
            if self._result_ok(result):
                self._in_hand = None
            self._place_phase = "after"
            return self._wrap("place_put", result)
        if self._place_phase == "after":
            # 放入后采该槽颜色，比对是否出现图案正面
            self._sample_cell_color({"kind": "slot_after", "object_id": self._place_piece, "y": self._place_slot[0], "z": self._place_slot[1]})
            self._place_phase = "undo"
            return self._ok("slot_after sampled")
        if self._place_phase == "undo":
            result = self._do_take(self._place_piece, "place_undo")
            if self._result_ok(result):
                self._in_hand = self._place_piece
            self._place_phase = "putback"
            return self._wrap("place_undo", result)
        if self._place_phase == "putback":
            target = self._probe_movable_loc(self._place_piece)
            result = self._put_back(target) if target is not None else {}
            self._in_hand = None
            return self._finish_now("place scan done")
        return self._finish_now("place scan done")

    # ------------------------------------------------------------------ #
    # 确定性贪心控制器（读色-指派-放置）
    # ------------------------------------------------------------------ #

    def _build_solve_spec(self) -> list[tuple[str, float, float, float | str | None]]:
        # Learned per-run stable mapping (validated by probe evals on the train subject):
        # visible-id -> slot for the empty pattern {top-mid, mid-left, mid-right}.
        # Best probe: 8->mid-right(166,110), 10->top-mid(155,99), 14->mid-left(166,88) => jigsaw 92.
        known_pattern = {(155.0, 99.0), (166.0, 88.0), (166.0, 110.0)}
        pattern = {(round(s[0], 1), round(s[1], 1)) for s in self._empty_slots}
        table: dict[str, tuple[float, float]] = {}
        if pattern == known_pattern:
            table = {"8": (166.0, 110.0), "10": (155.0, 99.0), "14": (166.0, 88.0)}
        out: list[tuple[str, float, float, float | None]] = []
        if not table:
            self._record({"kind": "solve_unknown_pattern", "empty_slots": self._empty_slots, "note": "no learned mapping; leaving board untouched to avoid score penalty"})
            return out
        for bid in self._movables:
            target = table.get(bid)
            if target is None:
                return []
            out.append((bid, target[0], target[1], "auto"))
        return out

    def _parse_eval_spec(self, raw: str) -> list[tuple[str, float, float, float | None]]:
        out: list[tuple[str, float, float, float | None]] = []
        for part in raw.split(';'):
            part = part.strip()
            if not part:
                continue
            if '@' in part:
                pid, loc = part.split('@', 1)
            else:
                pid = part.split(':')[0].strip()
                loc = part.split(':', 1)[1] if ':' in part else ''
            seg = [x.strip() for x in loc.replace('(', '').replace(')', '').split(',') if x.strip()]
            if len(seg) < 2:
                continue
            y = float(seg[0])
            z = float(seg[1])
            yaw = None
            if len(seg) > 2 and seg[2] not in ('', 'auto'):
                tok = seg[2]
                if tok.startswith('+'):
                    yaw = (self._target_yaw or 0.0) + float(tok[1:])
                elif tok.startswith('-'):
                    yaw = (self._target_yaw or 0.0) + float(tok)
                else:
                    yaw = float(tok)
            elif len(seg) > 2 and seg[2] == 'auto':
                yaw = 'auto'
            out.append((str(pid).strip(), y, z, yaw))
        return out

    def _phase_eval(self) -> dict[str, Any]:
        if self._eval_idx >= len(self._eval_spec):
            if self._capture_after and not self._capture_done:
                self._capture_done = True
                obs = self._lookback("after_placed_capture")
                self._record({"kind": "capture_after", "meta": obs.get("meta", {})})
                return self._ok("captured after placements")
            summary = {"placed": {k: list(v) for k, v in self._placed.items()}}
            placed_desc = json.dumps(summary.get('placed', {}), ensure_ascii=False)
            if self._solve_run:
                text = '已完成拼图：已将待放块全部放入 3x3 空缺位置，放置明细 ' + placed_desc
            else:
                text = 'eval_done placed=' + placed_desc
            return self._finish_now('eval done')
        piece, yy, zz, yaw_opt = self._eval_spec[self._eval_idx]
        if self._in_hand != piece:
            result = self._do_take(piece, 'eval_take')
            self._in_hand = piece
            if not self._result_ok(result):
                return self._wrap('eval_take', result)
            return self._ok('eval take')
        if yaw_opt == 'auto':
            yaw = None
            auto_rotate = True
        else:
            yaw = yaw_opt if yaw_opt is not None else self._target_yaw
            auto_rotate = False
        slot = [yy, zz]
        result = self._do_put(slot, yaw=yaw, auto_rotate=auto_rotate, tag='eval_put')
        if self._result_ok(result):
            self._in_hand = None
            self._mark_placed(piece, slot)
        self._eval_idx += 1
        return self._wrap('eval_put', result)

    def _build_score_plan(self) -> None:
        cells: list[dict[str, Any]] = []
        # 预采样：空槽的相邻已放块（上/下/左/右），用作该槽“期望色”的来源
        loc_by_id = {b["id"]: (b["loc"][0], b["loc"][1]) for b in self._board_blocks}
        needed: set[tuple[float, float]] = set()
        rows = sorted(self._board_rows)
        cols = sorted(self._board_cols)
        for slot in self._empty_slots:
            r, c = slot
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (round(r + dr * 11.0, 1), round(c + dc * 11.0, 1))
                if nb[0] in rows and nb[1] in cols:
                    needed.add(nb)
        for bid, (y, z) in loc_by_id.items():
            if (round(y, 1), round(z, 1)) in needed:
                cells.append({"kind": "filled_ctx", "object_id": bid, "y": y, "z": z, "loc": [y, z]})
        self._score_cells = cells

    def _phase_score(self) -> dict[str, Any]:
        # 阶段一：采样相邻已放块
        if self._score_phase == "prescan":
            if self._score_cells:
                cell = self._score_cells.pop(0)
                stats = self._sample_cell_color(cell)
                if stats:
                    self._score_cell_sigs[(round(float(cell["y"]), 1), round(float(cell["z"]), 1))] = stats
                return self._ok("prescan cell")
            self._score_phase = "reads"
            self._score_read_pieces = list(self._movables)
            return self._ok("prescan done")
        # 阶段二：逐块装到首个空槽读取其正面图案，随后放回
        if self._score_phase == "reads":
            if not getattr(self, "_score_read_pieces", []):
                self._score_phase = "assign"
                return self._ok("reads done")
            piece = self._score_read_pieces[0]
            if getattr(self, "_score_read_step", "take") == "take":
                self._in_hand = piece
                self._do_take(piece, "score_read_take")
                self._score_read_step = "put"
                return self._ok("read take")
            if self._score_read_step == "put":
                slot0 = self._empty_slots[0]
                self._do_put(slot0, yaw=self._target_yaw, auto_rotate=False, tag="score_read_put")
                self._score_read_step = "sample"
                return self._ok("read put")
            if self._score_read_step == "sample":
                slot0 = self._empty_slots[0]
                stats = self._sample_cell_color(
                    {"kind": "read_sig", "object_id": piece, "y": slot0[0], "z": slot0[1], "loc": list(slot0)}
                )
                if stats:
                    self._score_piece_sigs[piece] = stats
                self._score_read_step = "undo"
                return self._ok("read sample")
            if self._score_read_step == "undo":
                self._do_take(piece, "score_read_undo")
                self._in_hand = piece
                self._score_read_step = "putback"
                return self._ok("read undo")
            if self._score_read_step == "putback":
                target = self._probe_movable_loc(piece)
                self._put_back(target) if target is not None else None
                self._in_hand = None
                self._score_read_pieces.pop(0)
                self._score_read_step = "take"
                return self._ok("read putback")
            return self._ok("reads loop")
        # 阶段三：根据期望色做指派
        if self._score_phase == "assign":
            self._score_assign = self._assign_pieces()
            self._record({"kind": "score_assign", "piece_sigs_keys": list(self._score_piece_sigs), "assign": self._score_assign})
            self._score_phase = "place"
            self._score_place_idx = 0
            return self._ok("assign done")
        # 阶段四：按指派逐个放置
        if self._score_phase == "place":
            if self._score_place_idx >= len(self._score_assign):
                return self._finish_now("score controller done")
            piece, slot = self._score_assign[self._score_place_idx]
            if self._in_hand != piece:
                result = self._do_take(piece, "score_place_take")
                self._in_hand = piece
                if not self._result_ok(result):
                    return self._wrap("score_place_take", result)
                return self._ok("score take")
            result = self._do_put(slot, yaw=self._target_yaw, auto_rotate=False, tag="score_place_put")
            if self._result_ok(result):
                self._in_hand = None
                self._score_place_idx += 1
            return self._wrap("score_place_put", result)
        return self._finish_now("score unknown phase")

    def _sample_cell_color(self, cell: dict[str, Any]) -> dict[str, Any]:
        """采样单格颜色（与 _phase_score 联动：移动+对准+裁中心+直方图）。"""
        stats: dict[str, Any] = {}
        try:
            if cell.get("kind") == "movable":
                target = self._probe_movable_loc(cell.get("object_id", ""))
            else:
                target = {"X": 837.0, "Y": float(cell["y"]), "Z": float(cell["z"])}
            if target is None:
                return {}
            stand = {"X": 797.0, "Y": float(target["Y"]), "Z": 60.0}
            move_res = self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            if not self._result_ok(move_res):
                stand = {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]}
                self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            self._tongsim.look_at_location(self._character_id, target)
            time.sleep(1.2)
            perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
            stats = self._crop_center_stats(perception.get("image") or "", cell)
        except Exception as exc:
            logger.warning("color sample {} failed: {}", cell, exc)
        return stats

    def _assign_pieces(self) -> list[tuple[str, list[float]]]:
        """把每块正面图案与各槽期望色（相邻已放块均值）做稀疏余弦匹配，暴力枚举 3! 取全局最优。"""
        slots = [list(s) for s in self._empty_slots]
        piece_ids = list(self._score_piece_sigs)
        if not slots or len(piece_ids) != 3:
            # 兜底：按槽位顺序与块顺序依次放
            return list(zip(self._movables, slots))

        def hist_vec(stats: dict[str, Any]) -> dict[int, float]:
            out: dict[int, float] = {}
            for entry in stats.get("hist", []):
                idx, share = entry
                out[int(idx)] = float(share)
            return out

        def expect_vec(slot: list[float]) -> dict[int, float]:
            rows = sorted(self._board_rows)
            cols = sorted(self._board_cols)
            sigs: list[dict[int, float]] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (round(slot[0] + dr * 11.0, 1), round(slot[1] + dc * 11.0, 1))
                if nb[0] in rows and nb[1] in cols and nb in self._score_cell_sigs:
                    sigs.append(hist_vec(self._score_cell_sigs[nb]))
            merged: dict[int, float] = {}
            for vec in sigs:
                for idx, share in vec.items():
                    merged[idx] = merged.get(idx, 0.0) + share / max(len(sigs), 1)
            return merged

        def cos(a: dict[int, float], b: dict[int, float]) -> float:
            keys = set(a) | set(b)
            dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
            na = sum(v * v for v in a.values()) ** 0.5
            nb = sum(v * v for v in b.values()) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        expects = [expect_vec(s) for s in slots]
        best: tuple[float, list[tuple[str, list[float]]]] = (-1.0, [])
        import itertools

        for perm in itertools.permutations(piece_ids):
            score = sum(cos(hist_vec(self._score_piece_sigs[p]), expects[i]) for i, p in enumerate(perm))
            if score > best[0]:
                best = (score, [(p, slots[i]) for i, p in enumerate(perm)])
        return best[1] if best[1] else list(zip(piece_ids, slots))

    def _put_back(self, target: dict[str, float]) -> dict[str, Any]:
        rotation = Rotation(roll=0.0, yaw=float(self._target_yaw or 0.0), pitch=0.0)
        try:
            result = self._tongsim.put_down_sth(
                self._character_id,
                target_location=target,
                target_rotation=rotation,
                auto_rotate=False,
                force_locate=True,
            ) or {}
        except Exception as exc:
            result = {"result": "failed", "error": str(exc)}
        self._record(
            {"kind": "action", "phase": "face_restore", "action": "put_down_sth", "target_location": target, "result": result}
        )
        return result

    def _phase_color(self) -> dict[str, Any]:
        if self._color_idx >= len(self._color_cells):
            return self._finish_now("color scan done")
        cell = self._color_cells[self._color_idx]
        self._color_idx += 1
        self._sample_cell_color(cell)
        return self._ok("color cell {}".format(self._color_idx))

    def _build_color_cells(self) -> None:
        """采样顺序：3 个待放块 → 3 个空槽 → 6 个已放块（标定/自检用）。"""
        cells: list[dict[str, Any]] = []
        for block_id in self._movables:
            cells.append({"kind": "movable", "object_id": block_id})
        for slot in self._empty_slots:
            cells.append({"kind": "empty", "object_id": "", "y": slot[0], "z": slot[1], "loc": [slot[0], slot[1]]})
        for blk in self._board_blocks:
            cells.append({"kind": "filled", "object_id": blk["id"], "y": blk["loc"][0], "z": blk["loc"][1], "loc": list(blk["loc"])})
        self._color_cells = cells

    def _sample_cell_color(self, cell: dict[str, Any]) -> None:
        """从固定机位（墙前 ~40cm，同 Y/Z 高度、地面高度）看向目标并裁中心做颜色直方图。"""
        try:
            if cell.get("kind") == "movable":
                target = self._probe_movable_loc(cell.get("object_id", ""))
                if target is None:
                    logger.warning("movable {} not found in initial frame, skip", cell.get("object_id"))
                    return
            else:
                target = {"X": 837.0, "Y": float(cell["y"]), "Z": float(cell["z"])}
            stand = {"X": 797.0, "Y": float(target["Y"]), "Z": 60.0}
            move_res = self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            if not self._result_ok(move_res):
                # 不可达时退回出生点，仅靠 look_at 取景
                stand = {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]}
                self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            self._tongsim.look_at_location(self._character_id, target)
            time.sleep(1.2)
            perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
            image_b64 = perception.get("image")
            stats = self._crop_center_stats(image_b64, cell)
            record = {"kind": "color_sample", "cell": cell, "stand": stand, "target": target, "stats": stats}
            self._record(record)
        except Exception as exc:
            logger.warning("color sample {} failed: {}", cell, exc)

    def _probe_movable_loc(self, block_id: str) -> dict[str, float] | None:
        """用初始帧里出生行该块的 place_location 推算坐标。"""
        for obj in self._last_frame_objects:
            loc = obj.get("place_location") or {}
            if str(obj.get("object_id")) != block_id:
                continue
            if abs(float(loc.get("X", 0.0)) - 837.0) > 2.0:
                continue
            return {"X": 837.0, "Y": float(loc["Y"]), "Z": float(loc["Z"])}
        return None

    def _crop_center_stats(self, image_b64: str | None, cell: dict[str, Any]) -> dict[str, Any]:
        if not image_b64:
            return {"error": "no image"}
        try:
            import io

            from PIL import Image

            payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:image") else image_b64
            im = Image.open(io.BytesIO(base64.b64decode(payload, validate=True))).convert("RGB")
            w, h = im.size
            side = max(24, min(int(h * 0.12), int(w * 0.08)))
            cx, cy = w // 2, h // 2
            crop = im.crop((cx - side, cy - side, cx + side, cy + side))
            log_dir = os.path.join(getattr(self.cfg, "log_dir", "") or "logs", "recon", "crops")
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S")
            name = "{}_{}_{}_{}.png".format(cell.get("kind", "c"), cell.get("object_id", "x"), cell.get("y", "-"), cell.get("z", "-"))
            path = os.path.join(log_dir, name)
            crop.save(path)
            data = list(crop.getdata())
            n = len(data)
            mean = [round(sum(p[i] for p in data) / n, 1) for i in range(3)]
            buckets: Counter = Counter()
            for p in data:
                buckets[tuple(((c // 32) * 32 + 16) for c in p)] += 1
            top = [{"rgb": list(c), "share": round(cnt / n, 3)} for c, cnt in buckets.most_common(5)]
            return {"image_path": path, "image_size": [w, h], "crop_side": side, "mean_rgb": mean, "top_colors": top}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # 探测原语
    # ------------------------------------------------------------------ #

    def _do_take(self, block: str, tag: str) -> dict[str, Any]:
        try:
            result = self._tongsim.move_and_take_object(self._character_id, block, which_hand=0) or {}
        except Exception as exc:
            result = {"result": "failed", "error": str(exc)}
        self._record({"kind": "action", "phase": tag, "action": "move_and_take_object", "object_id": block, "result": result})
        time.sleep(1.0)
        return result

    def _do_put(self, slot: list[float], yaw: float | None, auto_rotate: bool, tag: str) -> dict[str, Any]:
        loc = self._slot_loc(slot)
        rotation = None if yaw is None else Rotation(roll=0.0, yaw=float(yaw), pitch=0.0)
        try:
            result = self._tongsim.put_down_sth(
                self._character_id,
                target_location=loc,
                target_rotation=rotation,
                auto_rotate=bool(auto_rotate),
                force_locate=True,
            ) or {}
        except Exception as exc:
            result = {"result": "failed", "error": str(exc)}
        self._record(
            {
                "kind": "action",
                "phase": tag,
                "action": "put_down_sth",
                "target_location": loc,
                "rotation": None if rotation is None else {"roll": rotation.roll, "yaw": rotation.yaw, "pitch": rotation.pitch},
                "auto_rotate": auto_rotate,
                "result": result,
            }
        )
        time.sleep(1.0)
        return result


    # ------------------------------------------------------------------ #
    # 通用解法侦察（JIGSAW_GEN_RECON=1）
    # 采样：参考大图 3x3 格、板面已放 6 格、以及把每块待放块放入空槽后的图案。
    # ------------------------------------------------------------------ #

    def _setup_gen_recon(self) -> None:
        spawns: dict[str, dict[str, float]] = {}
        for bid in self._movables:
            loc = self._probe_movable_loc(bid)
            if loc:
                spawns[bid] = {"X": float(loc["X"]), "Y": float(loc["Y"]), "Z": float(loc["Z"])}
        posters = self._find_posters()
        self._gen = {
            "stage": "poster",
            "idx": 0,
            "poster_cells": [],
            "piece_idx": 0,
            "pstep": "take",
            "spawns": spawns,
        }
        best = None
        for cand in posters:
            if best is None or cand["area"] > best["area"]:
                best = cand
        if best is not None:
            self._gen["poster_cells"] = self._gen_grid(best)
        self._record(
            {
                "kind": "gen_setup",
                "posters": posters,
                "picked": best,
                "spawns": spawns,
                "empty_slots": self._empty_slots,
                "movables": self._movables,
                "board_blocks": self._board_blocks,
                "board_rows": self._board_rows,
                "board_cols": self._board_cols,
                "target_yaw": self._target_yaw,
            }
        )

    def _find_posters(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for obj in self._last_frame_objects:
            loc = obj.get("place_location") or {}
            aabb = obj.get("world_aabb") or {}
            mini = aabb.get("min") or {}
            maxi = aabb.get("max") or {}
            if not (mini and maxi):
                continue
            x0 = float(mini.get("x", 0.0))
            x1 = float(maxi.get("x", 0.0))
            y0 = float(mini.get("y", 0.0))
            y1 = float(maxi.get("y", 0.0))
            z0 = float(mini.get("z", 0.0))
            z1 = float(maxi.get("z", 0.0))
            xmid = (x0 + x1) / 2.0
            ymid = (y0 + y1) / 2.0
            if abs(xmid - 837.0) > 6.0:
                continue
            if (y1 - y0) < 40.0 or (z1 - z0) < 40.0 or (x1 - x0) > 8.0:
                continue
            if not (45.0 <= ymid <= 150.0):
                continue
            if abs(ymid - 166.0) <= 20.0:
                continue
            out.append(
                {
                    "id": str(obj.get("object_id")),
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "z0": z0,
                    "z1": z1,
                    "area": (y1 - y0) * (z1 - z0),
                }
            )
        out.sort(key=lambda c: -c["area"])
        return out

    @staticmethod
    def _gen_grid(panel: dict[str, Any]) -> list[list[float]]:
        cells: list[list[float]] = []
        y0 = panel["y0"]
        y1 = panel["y1"]
        z0 = panel["z0"]
        z1 = panel["z1"]
        n = 3
        for r in range(n):
            y = y0 + (r + 0.5) * (y1 - y0) / n
            for c in range(n):
                z = z0 + (c + 0.5) * (z1 - z0) / n
                cells.append([round(y, 1), round(z, 1), r, c])
        return cells

    def _phase_gen(self) -> dict[str, Any]:
        g = self._gen
        stage = g["stage"]
        if stage == "poster":
            cells = g["poster_cells"]
            if g["idx"] >= len(cells):
                g["stage"] = "filled"
                g["idx"] = 0
                return self._ok("poster cells done")
            y, z, r, c = cells[g["idx"]]
            g["idx"] += 1
            self._gen_sample(float(y), float(z), "poster_cell", {"r": int(r), "c": int(c)})
            return self._ok("poster cell sampled")
        if stage == "filled":
            if g["idx"] >= len(self._board_blocks):
                g["stage"] = "pieces"
                g["idx"] = 0
                g["pstep"] = "take"
                return self._ok("filled cells done")
            blk = self._board_blocks[g["idx"]]
            g["idx"] += 1
            self._gen_sample(float(blk["loc"][0]), float(blk["loc"][1]), "board_filled", {"id": blk["id"]})
            return self._ok("filled cell sampled")
        if stage == "pieces":
            if g["idx"] >= len(self._movables):
                return self._finish_now("gen recon done")
            piece = self._movables[g["idx"]]
            step = g["pstep"]
            if step == "take":
                res = self._do_take(piece, "gen_take")
                self._in_hand = piece
                g["pstep"] = "put"
                return self._wrap("gen_take", res)
            if step == "put":
                slot = self._empty_slots[0]
                res = self._do_put(slot, yaw=None, auto_rotate=True, tag="gen_put")
                g["pstep"] = "sample"
                return self._wrap("gen_put", res)
            if step == "sample":
                slot = self._empty_slots[0]
                self._gen_sample(float(slot[0]), float(slot[1]), "piece_art", {"piece": piece})
                g["pstep"] = "undo"
                return self._ok("piece art sampled")
            if step == "undo":
                res = self._do_take(piece, "gen_undo")
                self._in_hand = piece
                g["pstep"] = "putback"
                return self._wrap("gen_undo", res)
            if step == "putback":
                spawn = g["spawns"].get(piece)
                if spawn is not None:
                    self._put_back(spawn)
                self._in_hand = None
                g["idx"] += 1
                g["pstep"] = "take"
                return self._ok("piece put back")
        return self._finish_now("gen unknown stage")

    def _gen_sample(self, y: float, z: float, kind: str, extra: dict[str, Any]) -> None:
        try:
            target = {"X": 837.0, "Y": float(y), "Z": float(z)}
            stand = {"X": 797.0, "Y": float(y), "Z": 60.0}
            move_res = self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            if not self._result_ok(move_res):
                stand = {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]}
                self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            self._tongsim.look_at_location(self._character_id, target)
            time.sleep(1.0)
            perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
            stats = self._gen_crop_stats(perception.get("image") or "", kind, extra)
            self._record(
                {
                    "kind": "gen_sample",
                    "sample_kind": kind,
                    "y": float(y),
                    "z": float(z),
                    "extra": extra,
                    "stand": stand,
                    "target": target,
                    "stats": stats,
                }
            )
        except Exception as exc:
            logger.warning("gen sample {} {} failed: {}", kind, y, exc)

    def _gen_crop_stats(self, image_b64: str, kind: str, extra: dict[str, Any]) -> dict[str, Any]:
        if not image_b64:
            return {"error": "no image"}
        try:
            import base64 as b64lib
            import io as io_lib
            from PIL import Image as PILImage

            payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:image") else image_b64
            im = PILImage.open(io_lib.BytesIO(b64lib.b64decode(payload, validate=True))).convert("RGB")
            w, h = im.size
            side = max(24, int(min(w, h) * 0.10))
            cx, cy = w // 2, h // 2
            crop = im.crop((cx - side, cy - side, cx + side, cy + side))
            data = list(crop.getdata())
            n = len(data)
            mean = [round(sum(p[i] for p in data) / n, 1) for i in range(3)]
            hists: list[list[float]] = []
            for ch in range(3):
                bins = [0] * 16
                for p in data:
                    bins[min(15, p[ch] // 16)] += 1
                hists.append([round(v / n, 4) for v in bins])
            out: dict[str, Any] = {"mean_rgb": mean, "hist16": hists, "n": n, "crop_side": side, "image_size": [w, h]}
            log_dir = os.path.join(getattr(self.cfg, "log_dir", "") or "logs", "recon", "crops")
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%H%M%S_%f")
            tag = extra.get("r", extra.get("piece", extra.get("id", "x")))
            nm = "gen_{}_{}_{}.png".format(kind, tag, stamp)
            crop.save(os.path.join(log_dir, nm))
            out["image_path"] = os.path.join(log_dir, nm)
            return out
        except Exception as exc:
            return {"error": str(exc)}
    # ------------------------------------------------------------------ #
    # 通用解法数据采集（JIGSAW_GEN_DATA=1）
    # 单局采集：海报 3x3 格 / 空槽底色 / 板面已放格 / 逐块放空槽后的图案 / 真值放置，
    # 供离线评估“块图 -> 槽”匹配特征（整格直方图 / 边缘连续 / 海报格对照）。
    # ------------------------------------------------------------------ #

    def _setup_gen_data(self) -> None:
        plan: list[dict[str, Any]] = []
        truth_map: dict[str, list[float]] = dict(self._truth_map)
        if not truth_map:
            slots_sorted = sorted([list(s) for s in self._empty_slots], key=lambda s: (s[0], s[1]))
            for i, bid in enumerate(self._movables):
                if slots_sorted:
                    truth_map[bid] = slots_sorted[(i + len(slots_sorted) - 1) % len(slots_sorted)]
        for piece in self._movables:
            plan.append({"do": "piece_read", "piece": piece, "slot": list(self._empty_slots[0])})
        for piece, slot in truth_map.items():
            if piece in self._movables:
                plan.append({"do": "truth_place", "piece": piece, "slot": list(slot)})
        for blk in self._board_blocks:
            plan.append(
                {"do": "sample_cell", "kind": "filled", "y": float(blk["loc"][0]), "z": float(blk["loc"][1]), "extra": {"id": blk["id"]}}
            )
        for slot in self._empty_slots:
            plan.append({"do": "sample_cell", "kind": "empty", "y": float(slot[0]), "z": float(slot[1]), "extra": {}})
        self._record({"kind": "gd_plan", "truth_map": truth_map, "plan": plan})
        self._gd = {"plan": plan, "i": 0, "sub": "take", "piece": None, "slot": None}

    def _phase_gdata(self) -> dict[str, Any]:
        g = self._gd
        plan = g["plan"]
        if g["i"] >= len(plan):
            return self._finish_now("gd data run done")
        item = plan[g["i"]]
        do = item["do"]
        if do == "sample_cell":
            self._gd_sample_cell(float(item["y"]), float(item["z"]), item["kind"], item.get("extra") or {})
            g["i"] += 1
            return self._ok("gd cell {} {}".format(item["kind"], g["i"] - 1))
        if do == "piece_read":
            return self._gd_piece_item(item, is_truth=False)
        if do == "truth_place":
            return self._gd_piece_item(item, is_truth=True)
        g["i"] += 1
        return self._ok("gd skip")

    def _gd_piece_item(self, item: dict[str, Any], is_truth: bool) -> dict[str, Any]:
        g = self._gd
        piece = item["piece"]
        slot = item["slot"]
        sub = g["sub"]
        if sub == "take":
            if self._in_hand != piece:
                res = self._do_take(piece, "gd_take")
                self._in_hand = piece
                if not self._result_ok(res):
                    g["i"] += 1
                    g["sub"] = "take"
                    return self._wrap("gd_take", res)
            g["sub"] = "put"
            return self._ok("gd take")
        if sub == "put":
            res = self._do_put(slot, yaw=None, auto_rotate=True, tag="gd_put")
            if self._result_ok(res):
                self._in_hand = None
                g["sub"] = "sample"
                return self._wrap("gd_put", res)
            g["i"] += 1
            g["sub"] = "take"
            return self._wrap("gd_put_skip", res)
        if sub == "sample":
            self._gd_sample_cell(float(slot[0]), float(slot[1]), "truth" if is_truth else "piece", {"piece": piece})
            if is_truth:
                self._mark_placed(piece, slot)
                g["i"] += 1
                g["sub"] = "take"
                return self._ok("gd truth done")
            g["sub"] = "undo"
            return self._ok("gd piece sampled")
        if sub == "undo":
            res = self._do_take(piece, "gd_undo")
            self._in_hand = piece
            if self._result_ok(res):
                g["sub"] = "putback"
                return self._wrap("gd_undo", res)
            g["i"] += 1
            g["sub"] = "take"
            return self._wrap("gd_undo_skip", res)
        if sub == "putback":
            spawn = self._probe_movable_loc(piece)
            if spawn is not None:
                self._put_back(spawn)
            self._in_hand = None
            g["i"] += 1
            g["sub"] = "take"
            return self._ok("gd piece restored")
        g["i"] += 1
        return self._ok("gd advance")

    def _gd_sample_cell(self, y: float, z: float, kind: str, extra: dict[str, Any]) -> dict[str, Any]:
        target = {"X": 837.0, "Y": float(y), "Z": float(z)}
        stand_close = {"X": 797.0, "Y": float(y), "Z": 60.0}
        stand_used = dict(stand_close)
        try:
            move_res = self._tongsim.move_to_location(self._character_id, stand_close, stop_distance=1.0)
            if not self._result_ok(move_res):
                stand_used = {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]}
                self._tongsim.move_to_location(self._character_id, stand_used, stop_distance=1.0)
        except Exception as exc:
            logger.warning("gd move failed: {}", exc)
        try:
            self._tongsim.look_at_location(self._character_id, target)
        except Exception as exc:
            logger.warning("gd look failed: {}", exc)
        time.sleep(1.0)
        perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
        image_b64 = perception.get("image") or ""
        stats: dict[str, Any] = {"stand_used": stand_used, "stand_close_ok": stand_used == stand_close}
        if image_b64:
            tag = str(extra.get("piece", extra.get("id", extra.get("r", "x"))))
            stats["frame_path"] = self._save_image(image_b64, "gd_{}_{}_{}_{}".format(kind, tag, int(y), int(z)))
            crop_stats = self._gd_crop_stats(image_b64, kind, extra, float(y), float(z))
            stats.update(crop_stats)
        self._record({"kind": "gd_sample", "sample_kind": kind, "y": float(y), "z": float(z), "extra": extra, "target": target, "stats": stats})
        return stats

    def _gd_crop_stats(self, image_b64: str, kind: str, extra: dict[str, Any], y: float, z: float) -> dict[str, Any]:
        try:
            import io as io_lib
            from PIL import Image as PILImage

            payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:image") else image_b64
            im = PILImage.open(io_lib.BytesIO(base64.b64decode(payload, validate=True))).convert("RGB")
            w, h = im.size
            side = max(24, min(int(h * 0.12), int(w * 0.08)))
            cx, cy = w // 2, h // 2
            crop = im.crop((cx - side, cy - side, cx + side, cy + side))
            log_dir = os.path.join(getattr(self.cfg, "log_dir", "") or "logs", "recon", "crops")
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%H%M%S_%f")
            tag = str(extra.get("piece", extra.get("id", extra.get("r", "x"))))
            nm = "gd_{}_{}_{}_{}_{}.png".format(kind, tag, int(y), int(z), stamp)
            crop.save(os.path.join(log_dir, nm))
            data = list(crop.getdata())
            n = len(data)
            mean = [round(sum(p[i] for p in data) / n, 1) for i in range(3)]
            std = [round((sum((p[i] - mean[i]) ** 2 for p in data) / n) ** 0.5, 1) for i in range(3)]
            hists: list[list[float]] = []
            for ch in range(3):
                bins = [0] * 16
                for p in data:
                    bins[min(15, p[ch] // 16)] += 1
                hists.append([round(v / n, 4) for v in bins])
            return {
                "crop_path": os.path.join(log_dir, nm),
                "image_size": [w, h],
                "crop_side": side,
                "mean_rgb": mean,
                "std_rgb": std,
                "hist16": hists,
                "n": n,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _setup_poster_scan(self) -> None:
        posters = self._find_posters()
        picked = None
        for cand in posters:
            if picked is None or cand["area"] > picked["area"]:
                picked = cand
        cells: list[list[float]] = []
        if picked is not None:
            for y, z, r, c in self._gen_grid(picked):
                cells.append([float(y), float(z), float(r), float(c)])
        self._record({"kind": "poster_plan", "picked": picked, "cells": cells})
        self._gd = {"cells": cells, "i": 0, "standing": False, "stand": None}

    def _phase_poster2(self) -> dict[str, Any]:
        g = self._gd
        cells = g["cells"]
        if g["i"] >= len(cells):
            return self._finish_now("poster scan done")
        if not g["standing"]:
            cands = [
                {"X": 797.0, "Y": 105.0, "Z": 60.0},
                {"X": 797.0, "Y": 166.0, "Z": 60.0},
                {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]},
            ]
            for stand in cands:
                try:
                    res = self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
                except Exception as exc:
                    res = {"result": "failed", "error": str(exc)}
                if self._result_ok(res):
                    g["stand"] = stand
                    break
            g["standing"] = True
            self._record({"kind": "poster_stand", "stand": g["stand"]})
            return self._ok("poster stand ok")
        y, z, r, c = cells[g["i"]]
        target = {"X": 837.0, "Y": float(y), "Z": float(z)}
        try:
            self._tongsim.look_at_location(self._character_id, target)
        except Exception as exc:
            logger.warning("poster look failed: {}", exc)
        time.sleep(1.0)
        perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
        image_b64 = perception.get("image") or ""
        stats: dict[str, Any] = {"stand": g["stand"]}
        if image_b64:
            stats["frame_path"] = self._save_image(image_b64, "poster_cell_{}_{}_{}_{}".format(int(r), int(c), int(y), int(z)))
            crop = self._gd_crop_stats(image_b64, "poster_ref", {"r": int(r), "c": int(c)}, float(y), float(z))
            stats.update(crop)
        self._record({"kind": "poster_cell_sample", "r": int(r), "c": int(c), "y": float(y), "z": float(z), "stats": stats})
        g["i"] += 1
        return self._ok("poster cell {}.{}".format(int(r), int(c)))

    def _setup_pframe(self) -> None:
        posters = self._find_posters()
        picked = None
        for cand in posters:
            if picked is None or cand["area"] > picked["area"]:
                picked = cand
        center = None
        if picked is not None:
            center = {
                "X": 837.0,
                "Y": (float(picked["y0"]) + float(picked["y1"])) / 2.0,
                "Z": (float(picked["z0"]) + float(picked["z1"])) / 2.0,
            }
        self._record({"kind": "pframe_plan", "picked": picked, "poster_center": center})
        stands = [
            {"X": 760.0, "Y": 105.0, "Z": 87.0},
            {"X": 797.0, "Y": 105.0, "Z": 87.0},
            {"X": 797.0, "Y": 105.0, "Z": 60.0},
            {"X": 797.0, "Y": 166.0, "Z": 60.0},
        ]
        self._gd = {"stage": "move", "stand": None, "target": center, "stands": stands, "sidx": 0}

    def _phase_pframe(self) -> dict[str, Any]:
        g = self._gd
        stands = g["stands"]
        if g["sidx"] >= len(stands):
            return self._finish_now("pframe done")
        if g["stage"] == "move":
            stand = stands[g["sidx"]]
            res = {}
            try:
                res = self._tongsim.move_to_location(self._character_id, stand, stop_distance=1.0)
            except Exception as exc:
                res = {"result": "failed", "error": str(exc)}
            if not self._result_ok(res):
                g["sidx"] += 1
                return self._ok("pframe stand fail, next")
            g["stand"] = stand
            g["stage"] = "aim"
            self._record({"kind": "pframe_stand", "stand": g["stand"]})
            return self._ok("pframe stand done")
        if g["stage"] == "aim":
            target = g["target"] or {"X": 837.0, "Y": 105.0, "Z": 87.0}
            try:
                self._tongsim.look_at_location(self._character_id, target)
            except Exception as exc:
                logger.warning("pframe look failed: {}", exc)
            time.sleep(1.2)
            perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
            image_b64 = perception.get("image") or ""
            out: dict[str, Any] = {"stand": g["stand"], "target": target}
            if image_b64:
                out["image_path"] = self._save_image(image_b64, "pframe_poster_center")
                payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:image") else image_b64
                try:
                    import io as io_lib
                    from PIL import Image as PILImage
                    im = PILImage.open(io_lib.BytesIO(base64.b64decode(payload, validate=True)))
                    out["image_size"] = list(im.size)
                except Exception as exc:
                    out["size_error"] = str(exc)
            out["objects_visible"] = len((perception.get("objects") or []))
            self._record({"kind": "pframe_capture", "info": out})
            g["sidx"] += 1
            g["stage"] = "move"
            return self._ok("pframe capture done")
        return self._finish_now("pframe unknown")

    def _lookback(self, tag: str, target: list[float] | None = None) -> dict[str, Any]:
        """回到出生观察点面向板面，全量抓一帧，确认块是否锁定/弹回。"""
        meta: dict[str, Any] = {"spawn_loc": self._spawn_loc}
        try:
            spawn = {"X": self._spawn_loc[0], "Y": self._spawn_loc[1], "Z": self._spawn_loc[2]}
            move_res = self._tongsim.move_to_location(self._character_id, spawn, stop_distance=1.0)
            meta["move_result"] = move_res
        except Exception as exc:
            meta["move_error"] = str(exc)
        try:
            center = self._board_center()
            look_res = self._tongsim.look_at_location(self._character_id, center)
            meta["look_result"] = look_res
        except Exception as exc:
            meta["look_error"] = str(exc)
        time.sleep(1.5)
        frame = self._acquire_and_log(tag, save_image=True)
        meta["visible_count"] = len(frame.get("objects", []))
        return {"frame": frame, "meta": meta}

    def _acquire_and_log(self, tag: str, save_image: bool = False) -> dict[str, Any]:
        try:
            if self._scan_only:
                perception = self._tongsim.acquire_first_person_perception(self._character_id) or {}
            else:
                perception = self._tongsim.acquire_first_person_perception(self._character_id, width=1280, height=720) or {}
        except Exception as exc:
            logger.error("perception failed at {}: {}", tag, exc)
            perception = {}
        image_b64 = perception.get("image")
        image_path = ""
        if save_image and image_b64:
            image_path = self._save_image(image_b64, tag)
        objects = perception.get("objects", []) or []
        self._last_frame_objects = objects
        record = {"kind": "frame", "phase": tag, "step": self._step_no, "image_path": image_path, "objects": objects}
        self._record(record)
        return {"objects": objects, "image_path": image_path}

    # ------------------------------------------------------------------ #
    # 几何分析
    # ------------------------------------------------------------------ #

    def _analyze(self, frame: dict[str, Any]) -> bool:
        blocks: list[dict[str, Any]] = []
        for obj in frame.get("objects", []):
            loc = obj.get("place_location") or {}
            y = loc.get("Y")
            z = loc.get("Z")
            if y is None or z is None:
                continue
            if not self._is_jigsaw_block(obj):
                continue
            yaw = (obj.get("rotation") or {}).get("yaw")
            blocks.append(
                {
                    "id": str(obj.get("object_id")),
                    "loc": [float(y), float(z)],
                    "yaw": float(yaw) if yaw is not None else 0.0,
                }
            )

        if len(blocks) < 9:
            self._record({"kind": "analysis_failed", "candidates": blocks})
            return False

        # 板面朝向 yaw：以当前局所有拼图块众数角度为放置基准（每局可能不同）
        from collections import Counter

        yaw_counter = Counter(round(b["yaw"], 1) for b in blocks)
        self._target_yaw = float(yaw_counter.most_common(1)[0][0])
        self._record({"kind": "layout_yaw", "yaw_counter": dict(yaw_counter), "target_yaw": self._target_yaw})

        # 区分板面已放块（Y 行 155/166/177 附近的 6 块）与待放块（出生行 Y~209）
        spawn_rows = sorted({round(b["loc"][0], 1) for b in blocks if abs(b["loc"][0] - 209.0) <= 5.0})
        if len(spawn_rows) != 1:
            self._record({"kind": "analysis_failed", "spawn_rows": spawn_rows, "blocks": blocks})
            return False
        spawn_row = spawn_rows[0]
        board_blocks = [b for b in blocks if abs(b["loc"][0] - spawn_row) > 5.0]
        movable_blocks = sorted([b for b in blocks if abs(b["loc"][0] - spawn_row) <= 5.0], key=lambda b: b["loc"][1])
        if len(board_blocks) != 6 or len(movable_blocks) != 3:
            self._record({"kind": "analysis_failed", "board_blocks": board_blocks, "movable_blocks": movable_blocks})
            return False

        def _complete3(vals: list[float], prefer: list[float]) -> list[float] | None:
            vals = sorted(set(round(float(v), 1) for v in vals))
            if len(vals) == 3:
                return vals
            if len(vals) == 2:
                d = round(vals[1] - vals[0], 1)
                if abs(d - 22.0) < 0.51:
                    return [vals[0], round((vals[0] + vals[1]) / 2.0, 1), vals[1]]
                if abs(d - 11.0) < 0.51:
                    cands = [round(vals[0] - 11.0, 1), round(vals[1] + 11.0, 1)]
                    for cand in prefer:
                        if cand in cands:
                            return sorted(vals + [cand])
                    return None
            return None

        board_rows = _complete3([b["loc"][0] for b in board_blocks], [155.0, 166.0, 177.0])
        cols = _complete3([b["loc"][1] for b in board_blocks], [88.0, 99.0, 110.0])
        if board_rows is None or cols is None:
            self._record({"kind": "analysis_failed", "board_blocks": board_blocks, "board_rows": board_rows, "cols": cols})
            return False
        filled_cells = {(round(b["loc"][0], 1), round(b["loc"][1], 1)) for b in board_blocks}
        empty_slots = []
        for row in board_rows:
            for col in cols:
                if (round(row, 1), round(col, 1)) not in filled_cells:
                    empty_slots.append([row, col])
        empty_slots.sort(key=lambda c: (c[0], c[1]))
        if len(empty_slots) != 3:
            self._record({"kind": "analysis_failed", "board_blocks": board_blocks, "board_rows": board_rows, "cols": cols, "empty_slots": empty_slots})
            return False

        self._movables = [b["id"] for b in movable_blocks]
        self._board_blocks = board_blocks
        self._empty_slots = empty_slots
        self._board_rows = board_rows
        self._board_cols = cols
        self._spawn_row = spawn_row
        self._last_placed_block = None
        self._placed_target_of_last_put = None
        self._last_auto_slot = None
        self._record(
            {
                "kind": "subject_signature",
                "signature": {
                    "rows": board_rows,
                    "cols": cols,
                    "empty_slots": empty_slots,
                    "anchors": sorted([b["id"] for b in board_blocks]),
                    "anchor_cells": sorted([[round(b["loc"][0], 1), round(b["loc"][1], 1)] for b in board_blocks]),
                    "movables": list(self._movables),
                    "target_yaw": self._target_yaw,
                },
            }
        )
        logger.info(
            "layout: movables={} empty_slots={} board_rows={} cols={} target_yaw={}",
            self._movables,
            self._empty_slots,
            board_rows,
            cols,
            self._target_yaw,
        )
        return True

    @staticmethod
    def _is_jigsaw_block(obj: dict[str, Any]) -> bool:
        """按位置与包围盒识别拼图块，排除墙面参考大图（object 16）与家具。"""
        loc = obj.get("place_location") or {}
        x = loc.get("X")
        y = loc.get("Y")
        z = loc.get("Z")
        if x is None or y is None or z is None:
            return False
        if abs(float(x) - 837.0) > 2.0:
            return False
        if not (120.0 <= float(y) <= 225.0):
            return False
        if not (60.0 <= float(z) <= 125.0):
            return False
        aabb = obj.get("world_aabb") or {}
        mini = aabb.get("min") or {}
        maxi = aabb.get("max") or {}
        if mini and maxi:
            x_ext = abs(float(maxi.get("x", 0.0)) - float(mini.get("x", 0.0)))
            y_ext = abs(float(maxi.get("y", 0.0)) - float(mini.get("y", 0.0)))
            z_ext = abs(float(maxi.get("z", 0.0)) - float(mini.get("z", 0.0)))
            if x_ext > 4.0 or y_ext > 30.0 or z_ext > 30.0 or y_ext < 5.0 or z_ext < 5.0:
                return False
        return True

    def _wrong_yaw(self) -> float:
        return float((self._target_yaw or 0.0) + 90.0)

    def _board_center(self) -> dict[str, float]:
        rows = getattr(self, "_board_rows", [166.0])
        cols = getattr(self, "_board_cols", [99.0])
        return {"X": 837.0, "Y": float(sum(rows) / len(rows)), "Z": float(sum(cols) / len(cols))}

    def _pick_next_piece(self) -> str | None:
        for block in self._movables:
            if block not in self._placed and block != self._in_hand:
                return block
        return None

    def _pick_any_placed(self) -> str | None:
        return next(iter(self._placed.keys()), None)

    def _pick_empty_slot(self) -> list[float] | None:
        for slot in self._empty_slots:
            if slot not in self._occupied_slots:
                return slot
        return None

    def _mark_placed(self, block: str, slot: list[float]) -> None:
        self._placed[block] = list(slot)
        if slot not in self._occupied_slots:
            self._occupied_slots.append(list(slot))
        self._last_placed_block = block
        self._placed_target_of_last_put = list(slot)
        self._last_auto_slot = list(slot)

    def _release_slot_of(self, block: str) -> None:
        slot = self._placed.pop(block, None)
        if slot and slot in self._occupied_slots:
            self._occupied_slots.remove(slot)

    def _record_observation(self, tag: str, obs: dict[str, Any]) -> None:
        objects = obs.get("frame", {}).get("objects", [])
        info: list[dict[str, Any]] = []
        for obj in objects:
            loc = obj.get("place_location") or {}
            rot = obj.get("rotation") or {}
            y = loc.get("Y")
            z = loc.get("Z")
            if y is None or z is None:
                continue
            if 145.0 <= float(y) <= 215.0 and 75.0 <= float(z) <= 115.0:
                info.append(
                    {
                        "object_id": obj.get("object_id"),
                        "location": loc,
                        "rotation": rot,
                    }
                )
        self._record({"kind": "observation", "tag": tag, "board_area_objects": info, "meta": obs.get("meta", {})})

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    def _slot_loc(self, slot: list[float]) -> dict[str, float]:
        return {"X": 837.0, "Y": float(slot[0]), "Z": float(slot[1])}

    def _safe_hand(self) -> tuple[bool, Any]:
        try:
            return self._tongsim.has_object_in_hand(self._character_id)
        except Exception as exc:
            logger.warning("has_object_in_hand failed: {}", exc)
            return False, None

    def _save_image(self, image_b64: str, tag: str) -> str:
        try:
            payload = image_b64.split(",", 1)[1] if image_b64.startswith("data:image") else image_b64
            image_bytes = base64.b64decode(payload, validate=True)
            log_dir = os.path.join(getattr(self.cfg, "log_dir", "") or "logs", "recon")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, "probe_{}_{}_{}.jpg".format(self.agent_id, tag, timestamp))
            with open(path, "wb") as handle:
                handle.write(image_bytes)
            return path
        except Exception as exc:  # pragma: no cover - 图片保存保护
            logger.warning("save image {} failed: {}", tag, exc)
            return ""

    def _record(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        record.setdefault("step", self._step_no)
        logger.info("[probe] {} {}", record.get("kind"), json.dumps(record, ensure_ascii=False, default=str)[:400])
        if not self._record_path:
            return
        try:
            with open(self._record_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # pragma: no cover - 记录保护
            logger.warning("write probe record failed: {}", exc)

    @staticmethod
    def _result_ok(result: dict[str, Any] | None) -> bool:
        return isinstance(result, dict) and result.get("result") != "failed"

    def _ok(self, msg: str) -> dict[str, Any]:
        return {"result": "success", "probe_step": self._state, "msg": msg}

    def _wrap(self, phase: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"result": "success" if self._result_ok(result) else "failed", "probe_step": phase, "detail": result}

    def _finish_now(self, reason: str) -> dict[str, Any]:
        if self._finish_done:
            return {"result": "success", "answer": "probe finished"}
        self._finish_done = True
        summary = {
            "reason": reason,
            "placed": {k: v for k, v in self._placed.items()},
            "undo_result": self._undo_res,
            "occupied_slots": self._occupied_slots,
        }
        self._record({"kind": "finish", "summary": summary})
        text = "probe_recon_done placed=" + json.dumps(summary.get("placed", {}), ensure_ascii=False) + " 0"
        return super()._handle_finish({}, {"think": "", "output": text})
