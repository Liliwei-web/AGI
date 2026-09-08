from __future__ import annotations

import os
import json
from typing import Any

from loguru import logger

from arenaagent.builder import Register
from arenaagent.utils.configclass import configclass
from arenaagent.vlm_agent.vlm_agent import VLMAgent, VLMAgentCfg


@configclass
class PreliminaryBaselineAgentCfg(VLMAgentCfg):
    name: str = "preliminary_baseline_agent"


@Register("preliminary_baseline_agent")
class PreliminaryBaselineAgent(VLMAgent):
    def __init__(
        self,
        stub,
        channel,
        cfg: PreliminaryBaselineAgentCfg | None = None,
        sleep_between_steps: float = 2.0,
    ) -> None:
        super().__init__(
            stub=stub,
            channel=channel,
            cfg=cfg or PreliminaryBaselineAgentCfg(),
            sleep_between_steps=sleep_between_steps,
        )

    def _should_handle_piece_transfer(self) -> bool:
        return False


    def _load_task_spec_prompts(self) -> dict[str, str]:
        if self._task_spec_prompt_cache is not None:
            return self._task_spec_prompt_cache

        try:
            here = os.path.dirname(os.path.abspath(__file__))
            task_spec_prompt_path = os.path.join(here, "prompts", "task_spec_prompt.json")
            with open(task_spec_prompt_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self._task_spec_prompt_cache = {str(key): str(value) for key, value in loaded.items()}
            else:
                self._task_spec_prompt_cache = {}
        except Exception as exc:
            logger.warning(f"加载 task_spec_prompt.json 失败: {exc}")
            self._task_spec_prompt_cache = {}

        return self._task_spec_prompt_cache

    def _build_prompt_variables(
        self,
        subject: Any,
        task_response: dict[str, Any],
        api_info: Any,
        visible_objects_info: list[dict[str, Any]],
        object_in_hand: Any,
    ) -> dict[str, Any]:
        task_type = subject.get("task_type", "") if isinstance(subject, dict) else ""
        task_prompt = self._load_task_spec_prompts().get(task_type, "") if task_type else ""

        if task_type == "jigsaw" and isinstance(subject, dict):
            reference_bounding = subject.get("reference_bounding", [])
            if reference_bounding and len(reference_bounding) >= 4:
                bounding_str = (
                    f"[Y: {reference_bounding[0]:.1f} ~ {reference_bounding[2]:.1f}, "
                    f"Z: {reference_bounding[3]:.1f} ~ {reference_bounding[1]:.1f}]"
                )
                task_prompt = (
                    f"{task_prompt}\n你需要将拼图块放置到{bounding_str}区域内。"
                    if task_prompt
                    else f"你需要将拼图块放置到{bounding_str}区域内。"
                )

        logger.debug("_last_action_res {}", self._last_action_res)

        return {
            "api_info": api_info,
            "example_objects_info": visible_objects_info[0] if len(visible_objects_info) > 0 else {},
            "task_text": subject["subject"],
            "task_prompt": task_prompt,
            "visiable_objects_info": visible_objects_info,
            "object_in_hand": object_in_hand,
            "response": task_response,
            "npc_reply": self._last_npc_reply,
            "npc_subject": self._last_npc_subject or {},
            "action_res": self._serialize_prompt_status(self._last_action_res),
            "apply_resp": self._serialize_prompt_status(self._last_apply_resp),
            "action_histories": self._action_histories,
        }

