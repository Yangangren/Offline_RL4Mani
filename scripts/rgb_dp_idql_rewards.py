"""Shared reward contracts for RGB IDQL dataset builders and trainers."""

from __future__ import annotations


LEGACY_RISE_REWARD_DEFINITION = "expert_transition=1; non_expert_transition=0"

REWARD_DEFINITIONS = {
    "task": "source_task_reward",
    "terminal_success": (
        "successful_episode: truncate_at_first_source_task_reward>0.5, "
        "reward=1_and_done=1_there; failed_episode: reward=0, "
        "done=1_at_source_end"
    ),
    "rise": (
        "successful_episode: truncate_at_first_source_task_reward>0.5, "
        "reward=1_and_done=1_there; failed_episode: reward=-1_and_done=1_"
        "at_source_end; all_nonterminal_rewards=0"
    ),
}

CANONICAL_TERMINAL_REWARD_MODES = frozenset(("terminal_success", "rise"))
