from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import json, re
from typing import Any, Dict, List, Optional, Tuple

client = OpenAI()
algo_ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")
hand_ds = load_dataset("Kevin355/Who_and_When", "Hand-Crafted")

def render_history(history):
    if isinstance(history, str):
        return history
    if isinstance(history, list):
        return "\n".join(json.dumps(turn, ensure_ascii=False) for turn in history)
    return str(history)

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

def _normalize_and_validate(d: dict) -> dict:
    def norm(k): return k.strip().lower().replace(" ", "_")
    nd = {norm(k): v for k, v in d.items()}
    aliases = {
        "agent": "agent_name", "name": "agent_name", "agentname": "agent_name",
        "step": "step_number", "stepnum": "step_number", "stepnumber": "step_number",
        "reason_for_mistake": "reason", "explanation": "reason",
    }
    for k, v in list(nd.items()):
        t = aliases.get(k)
        if t and t not in nd:
            nd[t] = v
    missing = [k for k in ("agent_name", "step_number", "reason") if k not in nd]
    if missing:
        raise KeyError(f"Missing required keys in model output: {missing}. Got keys: {list(nd.keys())}")
    nd["agent_name"] = str(nd["agent_name"]).strip()
    try:
        nd["step_number"] = int(nd["step_number"])
    except Exception:
        m = re.search(r"\d+", str(nd["step_number"]))
        if not m:
            raise ValueError(f"step_number is not an integer: {nd['step_number']}")
        nd["step_number"] = int(m.group(0))
    nd["reason"] = str(nd["reason"]).strip()
    return {"agent_name": nd["agent_name"], "step_number": nd["step_number"], "reason": nd["reason"]}

def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "yes", "y", "1"}

def _agent_from_turn(turn: Any) -> str:
    if isinstance(turn, dict):
        for k in ("agent", "name", "speaker", "role"):
            if k in turn and isinstance(turn[k], str) and turn[k].strip():
                return turn[k].strip()
        if len(turn) == 1:
            k = next(iter(turn.keys()))
            if isinstance(k, str) and k.strip():
                return k.strip()
    return "Unknown"

def _norm_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    # normalize both gold and pred the same way (case/space/underscore/punct)
    return re.sub(r"[\W_]+", "", s).casefold()

def _extract_gold(convo: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    # primary keys per dataset card
    ga = convo.get("mistake_agent")
    gs = convo.get("mistake_step")
    # fallbacks just in case
    if ga is None:
        for k in ("agent_name", "agent", "who_failed", "failure_agent", "label"):
            if k in convo and isinstance(convo[k], str) and convo[k].strip():
                ga = convo[k]; break
    if gs is None:
        for k in ("step_number", "step", "when", "failure_step"):
            if k in convo:
                gs = convo[k]; break
    if isinstance(gs, str):
        m = re.search(r"\d+", gs)
        gs = int(m.group(0)) if m else None
    elif gs is not None:
        try:
            gs = int(gs)
        except Exception:
            gs = None
    return ga, gs

def call_openai(prompt, model="gpt-4o-mini"):
    system_msg = (
        "You are an expert judge for multi-agent failure attribution. "
        "Always return ONLY a JSON object with keys exactly: agent_name, step_number, reason. "
        "The step numbering is 1-based and each message in the conversation counts as one step in order."
    )
    user_prompt = prompt + (
        "\n\nFORMAT INSTRUCTIONS:\n- Return ONLY JSON (no prose, no markdown) with keys: "
        '{"agent_name": string, "step_number": integer, "reason": string}.'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    parsed = _extract_json(raw)
    return _normalize_and_validate(parsed)

def all_at_once(category, convo_id):
    convo = category["train"][convo_id]
    problem = convo["question"]
    chat = render_history(convo["history"])
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is: {problem}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. "
        "If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        '{\n    "agent a": "xx",\n    "agent b": "xxxx",\n    "agent c": "xxxxx",\n    "agent a": "xxxxxxx"\n}\n'
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. "
        "If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. "
        "Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction.\n"
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )
    return call_openai(prompt)

# Step-wise judge that returns the earliest decisive error encountered
class StepByStepAttributor:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0, include_ground_truth: bool = False):
        self.client = client; self.model = model; self.temperature = temperature; self.include_ground_truth = include_ground_truth
    def attribute_from_dataset(self, category, convo_id: int) -> Dict[str, Any]:
        c = category["train"][convo_id]; gt = c.get("answer") or c.get("final_answer") or c.get("label") or None
        return self.attribute(problem=c["question"], history=c["history"], ground_truth=gt)
    def attribute(self, problem: str, history: List[Any], ground_truth: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(history, list) or not history:
            return {"agent_name": "Unknown", "step_number": 1, "reason": "Empty or invalid history; unable to run step-by-step attribution."}
        for i, turn in enumerate(history):
            partial = render_history(history[: i + 1])
            judge = self._judge_step(problem, partial, ground_truth if self.include_ground_truth else None)
            if judge["is_error"]:
                agent = _agent_from_turn(turn); k = i + 1
                return {"agent_name": agent, "step_number": k, "reason": f"First error detected at step {k} by '{agent}'. {judge['explanation']}".strip()}
        agent = _agent_from_turn(history[-1])
        return {"agent_name": agent, "step_number": len(history), "reason": "No decisive error flagged by the judge at any step; defaulted to the final step/agent as a tie-breaker."}
    def _judge_step(self, problem: str, partial_log: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        sys = ("You are an expert judge for multi-agent failure attribution. "
               "Always return ONLY a JSON object with keys exactly: is_error, explanation. "
               "Define 'error' as a decisive mistake that meaningfully derails solving the problem or makes the correct final answer unreachable.")
        gt = f"\nThe Answer for the problem is: {ground_truth}" if ground_truth else ""
        usr = (f"The problem being addressed is: {problem}.{gt}\n"
               f"Here is the conversation history up to the current step:\n{partial_log}\n\n"
               'Your task: Determine whether the most recent agent’s action contains a decisive error that could hinder or prevent solving the problem. Respond ONLY as JSON:\n{"is_error": true or false, "explanation": "brief rationale"}')
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=self.temperature)
        raw = resp.choices[0].message.content
        try:
            parsed = _extract_json(raw)
        except Exception:
            return {"is_error": False, "explanation": f"Judge returned non-parseable output; treating as no-error. Raw: {raw[:200]}"}
        return {"is_error": _coerce_bool(parsed.get("is_error", False)), "explanation": str(parsed.get("explanation", "")).strip()}

def step_by_step(category, convo_id, *, include_ground_truth=False):
    a = StepByStepAttributor(client=client, model="gpt-4o-mini", temperature=0.0, include_ground_truth=include_ground_truth)
    return a.attribute_from_dataset(category, convo_id)

# Binary search judge to locate earliest decisive error in O(log n)
class BinarySearchAttributor:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0, include_ground_truth: bool = False):
        self.client = client; self.model = model; self.temperature = temperature; self.include_ground_truth = include_ground_truth
    def attribute_from_dataset(self, category, convo_id: int) -> Dict[str, Any]:
        c = category["train"][convo_id]; gt = c.get("answer") or c.get("final_answer") or c.get("label") or None
        return self.attribute(problem=c["question"], history=c["history"], ground_truth=gt)
    def attribute(self, problem: str, history: List[Any], ground_truth: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(history, list) or not history:
            return {"agent_name": "Unknown", "step_number": 1, "reason": "Empty or invalid history; unable to run binary-search attribution."}
        n = len(history)
        full = render_history(history[:n])
        has_full, _ = self._prefix_has_error(problem, full, ground_truth if self.include_ground_truth else None)
        if not has_full:
            agent = _agent_from_turn(history[-1])
            return {"agent_name": agent, "step_number": n, "reason": "No decisive error flagged by the judge in the full conversation; defaulted to the final step/agent as a tie-breaker."}
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = render_history(history[:mid])
            has_err, _ = self._prefix_has_error(problem, prefix, ground_truth if self.include_ground_truth else None)
            if has_err:
                hi = mid
            else:
                lo = mid + 1
        k = lo
        agent = _agent_from_turn(history[k - 1])
        part_k = render_history(history[:k])
        reason = self._explain_step(problem, part_k, k, agent, ground_truth if self.include_ground_truth else None)
        return {"agent_name": agent, "step_number": k, "reason": reason}
    def _prefix_has_error(self, problem: str, partial_log: str, ground_truth: Optional[str] = None):
        sys = ("You are an expert judge for multi-agent failure attribution. "
               "Return ONLY JSON with keys exactly: has_error, rationale. "
               "A 'decisive error' is a mistake in the prefix that, even if all future messages were perfect, would still likely derail or make it impossible to reach the correct final answer without first correcting that mistake.")
        gt = f"\nGround truth answer (if known): {ground_truth}" if ground_truth else ""
        usr = (f"Problem: {problem}.{gt}\n"
               f"Conversation prefix (up to and including the current step):\n{partial_log}\n\n"
               'Question: Does this prefix already contain a decisive, unrecoverable error as defined above?\nRespond ONLY as JSON:\n{"has_error": true|false, "rationale": "brief reason"}')
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=self.temperature)
            raw = resp.choices[0].message.content
            parsed = _extract_json(raw)
            return _coerce_bool(parsed.get("has_error", False)), str(parsed.get("rationale", "")).strip()
        except Exception as e:
            return False, f"Judge parsing issue; treated as no-error. Detail: {e}"
    def _explain_step(self, problem: str, partial_log: str, step_index: int, agent_name: str, ground_truth: Optional[str] = None) -> str:
        sys = "You are an expert judge for multi-agent failure attribution. Return ONLY JSON with keys exactly: reason."
        gt = f"\nGround truth answer (if known): {ground_truth}" if ground_truth else ""
        usr = (f"Problem: {problem}.{gt}\n"
               f"You have identified that the earliest decisive error occurs at step {step_index}, spoken by '{agent_name}'.\n"
               f"Conversation up to and including step {step_index}:\n{partial_log}\n\n"
               f'Briefly explain why the message at this step is a decisive error. Avoid vague wording; explicitly refer to "the message at step {step_index}". Respond ONLY as JSON:\n{{"reason": "concise explanation"}}')
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=self.temperature)
            raw = resp.choices[0].message.content
            parsed = _extract_json(raw)
            r = str(parsed.get("reason", "")).strip()
            if not r:
                r = f"The message at step {step_index} by '{agent_name}' constitutes the earliest decisive error."
            return r.replace("last agent", f"message at step {step_index}")
        except Exception as e:
            return f"The message at step {step_index} by '{agent_name}' is the earliest decisive error, but the judge explanation could not be parsed. Detail: {e}"

def step_by_step(category, convo_id, *, include_ground_truth=False):
    a = StepByStepAttributor(client=client, model="gpt-4o-mini", temperature=0.0, include_ground_truth=include_ground_truth)
    return a.attribute_from_dataset(category, convo_id)

def binary_search(category, convo_id, *, include_ground_truth=False):
    a = BinarySearchAttributor(client=client, model="gpt-4o-mini", temperature=0.0, include_ground_truth=include_ground_truth)
    return a.attribute_from_dataset(category, convo_id)

def _evaluate_dataset(ds, ds_name: str) -> pd.DataFrame:
    methods = {
        "all_at_once": lambda cat, i: all_at_once(cat, i),
        "step_by_step": lambda cat, i: step_by_step(cat, i, include_ground_truth=False),
        "binary_search": lambda cat, i: binary_search(cat, i, include_ground_truth=False),
    }
    rows = []
    n = len(ds["train"])
    for method, fn in methods.items():
        counted = agent_ok = step_ok = joint_ok = 0
        for i in tqdm(range(n), desc=f"{ds_name} | {method}", ascii=True, leave=True):
            convo = ds["train"][i]
            gold_agent, gold_step = _extract_gold(convo)
            if gold_agent is None or gold_step is None:
                continue
            try:
                pred = fn(ds, i)
            except Exception:
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
            "method": method,
            "n_eval": counted,
            "agent_acc": (agent_ok / counted) if counted else 0.0,
            "step_acc": (step_ok / counted) if counted else 0.0,
            "joint_acc": (joint_ok / counted) if counted else 0.0,
            "agent_correct": agent_ok,
            "step_correct": step_ok,
            "joint_correct": joint_ok,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_algo = _evaluate_dataset(algo_ds, "Algorithm-Generated")
    df_hand = _evaluate_dataset(hand_ds, "Hand-Crafted")
    df = pd.concat([df_algo, df_hand], ignore_index=True)
    df = df[["dataset", "method", "n_eval", "agent_acc", "step_acc", "joint_acc", "agent_correct", "step_correct", "joint_correct"]]
    df[["agent_acc", "step_acc", "joint_acc"]] = df[["agent_acc", "step_acc", "joint_acc"]].round(4)
    print("\n=== Evaluation Results ===")
    print(df.to_string(index=False))