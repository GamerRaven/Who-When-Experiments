
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate three failure-attribution strategies (all_at_once, step_by_step, binary_search)
on the Who_and_When datasets using Anthropic Claude Sonnet 4 instead of OpenAI models.

Requirements:
  pip install anthropic datasets pandas tqdm

Environment:
  export ANTHROPIC_API_KEY=your_key_here
"""

from datasets import load_dataset
from anthropic import Anthropic
from tqdm import tqdm
import pandas as pd
import json, re
from typing import Any, Dict, List, Optional, Tuple

# === Model and decoding params ===
MODEL = "claude-sonnet-4-20250514"  # Claude Sonnet 4 (docs list alias 'claude-sonnet-4')
DETERMINISM = dict(temperature=0, top_p=0)
MAX_TOKENS = 1024  # per call; adjust if you see truncation

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

# === Helpers copied/adapted from original ===
def _extract_json(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("Model output is not text.")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        return json.loads(text[a:b+1])
    raise ValueError("No JSON object found in model output.")

def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "yes", "y", "1"}

def _norm_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r"[\W_]+", "", s).casefold()

def _extract_gold(convo: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    ga = convo.get("mistake_agent")
    gs = convo.get("mistake_step")
    if ga is None:
        for k in ("agent_name", "agent", "who_failed", "failure_agent", "label"):
            v = convo.get(k)
            if isinstance(v, str) and v.strip():
                ga = v
                break
    if gs is None:
        for k in ("step_number", "step", "when", "failure_step"):
            if k in convo:
                gs = convo[k]
                break
    if isinstance(gs, str):
        m = re.search(r"\d+", gs)
        gs = int(m.group(0)) if m else None
    elif gs is not None:
        try:
            gs = int(gs)
        except Exception:
            gs = None
    return ga, gs

def _turn_agent_name(turn: Any) -> Optional[str]:
    if not isinstance(turn, dict):
        return None
    for k in ("agent", "name", "speaker", "role"):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if len(turn) == 1:
        k = next(iter(turn.keys()))
        if isinstance(k, str) and k.strip():
            return k.strip()
    return None

def _turn_text(turn: Any) -> str:
    if not isinstance(turn, dict):
        return str(turn)
    for k in ("content", "message", "text", "assistant_response", "tool_call", "tool_result"):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(turn, ensure_ascii=False)

def filter_agent_messages(history: Any, known_agents: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[int]]:
    lines, agents, idx_map = [], [], []
    known_norm = {_norm_name(a) for a in (known_agents or []) if a}
    for i, turn in enumerate(history if isinstance(history, list) else []):
        name = _turn_agent_name(turn)
        if not name:
            continue
        if known_norm and _norm_name(name) not in known_norm:
            continue
        if name.lower() in {"system", "tool", "observation"}:
            continue
        msg = _turn_text(turn)
        lines.append(f"{name}: {msg}")
        agents.append(name)
        idx_map.append(i)
    return lines, agents, idx_map

def infer_known_agents(history: Any) -> List[str]:
    names, seen = [], set()
    if isinstance(history, list):
        for turn in history:
            n = _turn_agent_name(turn)
            if n and _norm_name(n) not in seen:
                seen.add(_norm_name(n))
                names.append(n)
    return names

# === Anthropic call wrappers ===
def _anthropic_call(system: str, user: str) -> str:
    """Single text response from Claude."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=DETERMINISM.get("temperature", 0),
        top_p=DETERMINISM.get("top_p", 1),
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Message content is a list of content blocks; we expect first to be text
    parts = getattr(msg, "content", [])
    if parts and getattr(parts[0], "type", "") == "text" and getattr(parts[0], "text", None):
        return parts[0].text
    # Fallback to stringifying whole object
    return str(msg)

def ask_json(system: str, user: str) -> dict:
    txt = _anthropic_call(system, user)
    return _extract_json(txt)

def ask_upper_lower(system: str, user: str) -> str:
    txt = _anthropic_call(system, user).strip().lower()
    # accept raw or json
    try:
        d = _extract_json(txt)
        choice = str(d.get("choice", "")).lower()
        if "upper" in choice:
            return "upper"
        if "lower" in choice:
            return "lower"
    except Exception:
        pass
    if "upper" in txt:
        return "upper"
    if "lower" in txt:
        return "lower"
    return "upper"  # deterministic fallback

# === Strategies (unchanged logic, new back-end) ===
def all_at_once(problem: str,
                agent_lines: List[str],
                agent_names: List[str],
                *,
                ground_truth: Optional[str] = None) -> Optional[Dict[str, Any]]:
    convo = "\n".join(agent_lines)
    gt = f'\nGround truth answer: {ground_truth}' if ground_truth else ''
    sys = (
        "You are an expert judge for multi-agent failure attribution. "
        "Return ONLY JSON with keys exactly: agent_name, step_number, reason. "
        "Step numbering is 1-based and counts ONLY agent messages."
    )
    usr = (
        f"Problem: {problem}{gt}\n\n"
        f"Conversation (agent-only, in order, one message per step):\n{convo}\n\n"
        "Decide:\n"
        "1) agent_name: the single agent most directly responsible for the failure\n"
        "2) step_number: the earliest step where that agent makes a decisive error\n"
        "3) reason: concise explanation tied to that step\n"
        'Respond ONLY as JSON: {"agent_name": string, "step_number": integer, "reason": string}'
    )
    try:
        out = ask_json(sys, usr)
        agent = str(out["agent_name"]).strip()
        step = int(re.search(r"\d+", str(out["step_number"])).group(0))
        reason = str(out.get("reason", "")).strip()
        return {"agent_name": agent, "step_number": step, "reason": reason}
    except Exception:
        return None

def step_by_step(problem: str,
                 agent_lines: List[str],
                 agent_names: List[str],
                 *,
                 ground_truth: Optional[str] = None) -> Optional[Dict[str, Any]]:
    gt = f'\nGround truth answer: {ground_truth}' if ground_truth else ''
    judge_sys = (
        "You are an expert judge for multi-agent failure attribution. "
        "Return ONLY JSON with keys exactly: is_error, explanation. "
        "A 'decisive error' is a mistake in the most recent message that derails or makes the correct final answer unreachable unless corrected first."
    )
    for k in range(1, len(agent_lines) + 1):
        prefix = "\n".join(agent_lines[:k])
        usr = (
            f"Problem: {problem}{gt}\n\n"
            f"Conversation prefix (steps 1..{k}):\n{prefix}\n\n"
            f"Does the latest message (step {k}) contain a decisive error as defined? "
            'Respond ONLY as JSON: {"is_error": true|false, "explanation": "brief"}'
        )
        try:
            out = ask_json(judge_sys, usr)
            if _coerce_bool(out.get("is_error", False)):
                return {
                    "agent_name": agent_names[k-1],
                    "step_number": k,
                    "reason": f"First decisive error at step {k} by {agent_names[k-1]}: {str(out.get('explanation','')).strip()}"
                }
        except Exception:
            continue
    return None

def binary_search(problem: str,
                  agent_lines: List[str],
                  agent_names: List[str],
                  *,
                  ground_truth: Optional[str] = None) -> Optional[Dict[str, Any]]:
    n = len(agent_lines)
    if n == 0:
        return None
    gt = f'\nGround truth answer: {ground_truth}' if ground_truth else ''

    lo, hi = 1, n
    choose_sys = (
        "You are an expert judge for multi-agent failure attribution. "
        "Return ONLY a single word or JSON with key 'choice' indicating 'upper' or 'lower'. "
        "You must choose which half is MORE LIKELY to contain the earliest decisive error."
    )
    while lo < hi:
        mid = (lo + hi) // 2
        segment = "\n".join(agent_lines[lo-1:hi])
        usr = (
            f"Problem: {problem}{gt}\n\n"
            f"Conversation segment (steps {lo}..{hi}):\n{segment}\n\n"
            f"Choose which half more likely contains the EARLIEST decisive error:\n"
            f"UPPER = steps {lo}..{mid} ; LOWER = steps {mid+1}..{hi}.\n"
            "Answer 'upper' or 'lower' (or JSON {\"choice\":\"upper|lower\"})."
        )
        choice = ask_upper_lower(choose_sys, usr)
        if choice == "upper":
            hi = mid
        else:
            lo = mid + 1

    k = lo

    reason_sys = (
        "You are an expert judge for multi-agent failure attribution. "
        "Return ONLY JSON with key 'reason'."
    )
    reason_usr = (
        f"Problem: {problem}{gt}\n\n"
        f"The earliest decisive error is hypothesized at step {k} said by {agent_names[k-1]}.\n"
        f"Step {k} content:\n{agent_lines[k-1]}\n\n"
        'Briefly explain why this specific message is a decisive error. '
        'Respond ONLY as JSON: {"reason": "concise explanation"}'
    )
    try:
        r = ask_json(reason_sys, reason_usr).get("reason", "")
        return {"agent_name": agent_names[k-1], "step_number": k, "reason": str(r).strip()}
    except Exception:
        return {"agent_name": agent_names[k-1], "step_number": k, "reason": ""}

def _evaluate_dataset(ds, ds_name: str, *, include_ground_truth: bool) -> pd.DataFrame:
    rows = []
    n = len(ds["train"])
    for method_name in ("all_at_once", "step_by_step", "binary_search"):
        counted = agent_ok = step_ok = joint_ok = 0
        for i in tqdm(range(n), desc=f"{ds_name} | {method_name} | {'withGT' if include_ground_truth else 'noGT'}",
                      ascii=True, leave=False):
            convo = ds["train"][i]
            gold_agent, gold_step = _extract_gold(convo)
            if gold_agent is None or gold_step is None:
                continue
            problem = convo.get("question") or convo.get("query") or ""
            history = convo.get("history") or []
            known = infer_known_agents(history)
            lines, agents, idx_map = filter_agent_messages(history, known_agents=known)
            if not lines:
                continue
            gt = convo.get("answer") or convo.get("final_answer") or convo.get("label") or None
            gt = gt if include_ground_truth else None
            if method_name == "all_at_once":
                pred = all_at_once(problem, lines, agents, ground_truth=gt)
            elif method_name == "step_by_step":
                pred = step_by_step(problem, lines, agents, ground_truth=gt)
            else:
                pred = binary_search(problem, lines, agents, ground_truth=gt)
            if not pred:
                continue
            pa = _norm_name(pred.get("agent_name"))
            ga = _norm_name(gold_agent)
            try:
                ps = int(pred.get("step_number"))
            except Exception:
                ps = None
            a_hit = (pa is not None and ga is not None and pa == ga)
            s_hit = (ps is not None and gold_step is not None and ps == int(gold_step))
            counted += 1
            agent_ok += int(a_hit)
            step_ok += int(s_hit)
            joint_ok += int(a_hit and s_hit)
        rows.append({
            "dataset": ds_name,
            "setting": "with_GT" if include_ground_truth else "without_GT",
            "method": method_name,
            "n_eval": counted,
            "agent_acc": (agent_ok / counted) if counted else 0.0,
            "step_acc": (step_ok / counted) if counted else 0.0,
            "joint_acc": (joint_ok / counted) if counted else 0.0,
            "agent_correct": agent_ok,
            "step_correct": step_ok,
            "joint_correct": joint_ok,
        })
    return pd.DataFrame(rows)

def run_all():
    algo_ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")
    hand_ds = load_dataset("Kevin355/Who_and_When", "Hand-Crafted")
    dfs = []
    for with_gt in (True, False):
        dfs.append(_evaluate_dataset(algo_ds, "Algorithm-Generated", include_ground_truth=with_gt))
        dfs.append(_evaluate_dataset(hand_ds, "Hand-Crafted", include_ground_truth=with_gt))
    df = pd.concat(dfs, ignore_index=True)
    df = df[["dataset", "setting", "method", "n_eval", "agent_acc", "step_acc", "joint_acc",
             "agent_correct", "step_correct", "joint_correct"]]
    df[["agent_acc", "step_acc", "joint_acc"]] = df[["agent_acc", "step_acc", "joint_acc"]].round(4)
    print("\n=== Evaluation Results (with and without GT separately) ===")
    print(df.to_string(index=False))
    avg = (df.groupby(["dataset", "method"], as_index=False)
             .agg(n_eval=("n_eval", "sum"),
                  agent_acc=("agent_acc", "mean"),
                  step_acc=("step_acc", "mean"),
                  joint_acc=("joint_acc", "mean"),
                  agent_correct=("agent_correct", "sum"),
                  step_correct=("step_correct", "sum"),
                  joint_correct=("joint_correct", "sum")))
    avg[["agent_acc", "step_acc", "joint_acc"]] = avg[["agent_acc", "step_acc", "joint_acc"]].round(4)
    print("\n=== Averaged over settings ===")
    print(avg.to_string(index=False))

if __name__ == "__main__":
    run_all()