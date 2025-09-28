from datasets import load_dataset
from openai import OpenAI
import json
import re

# Load datasets
algo_ds = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated")
hand_ds = load_dataset("Kevin355/Who_and_When", "Hand-Crafted")

def render_history(history):
    """Return the conversation history as a newline-joined string.
    If `history` is already a string, return it.
    If it's a list[dict], JSON-encode each turn so step counting is clear.
    """
    if isinstance(history, str):
        return history
    if isinstance(history, list):
        return "\n".join(json.dumps(turn, ensure_ascii=False) for turn in history)
    return str(history)

def _extract_json(text: str) -> dict:
    """Extract a JSON object from model output, even if wrapped in code fences or extra prose."""
    if not isinstance(text, str):
        raise ValueError("Model output is not text.")

    # 1) Try direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Try fenced block ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))

    # 3) Fallback: slice from first '{' to last '}' and try
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])

    # If all fail:
    raise ValueError("No JSON object found in model output.")

def _normalize_and_validate(d: dict) -> dict:
    """Normalize keys and validate schema."""
    def norm(k):
        return k.strip().lower().replace(" ", "_")
    nd = {norm(k): v for k, v in d.items()}

    aliases = {
        "agent": "agent_name",
        "name": "agent_name",
        "agentname": "agent_name",
        "step": "step_number",
        "stepnum": "step_number",
        "stepnumber": "step_number",
        "reason_for_mistake": "reason",
        "explanation": "reason",
    }
    for k, v in list(nd.items()):
        if k in aliases and aliases[k] not in nd:
            nd[aliases[k]] = v

    missing = [k for k in ("agent_name", "step_number", "reason") if k not in nd]
    if missing:
        raise KeyError(f"Missing required keys in model output: {missing}. Got keys: {list(nd.keys())}")

    # Coerce types
    nd["agent_name"] = str(nd["agent_name"]).strip()
    try:
        nd["step_number"] = int(nd["step_number"])
    except Exception:
        m = re.search(r"\d+", str(nd["step_number"]))
        if not m:
            raise ValueError(f"step_number is not an integer: {nd['step_number']}")
        nd["step_number"] = int(m.group(0))
    nd["reason"] = str(nd["reason"]).strip()

    return {
        "agent_name": nd["agent_name"],
        "step_number": nd["step_number"],
        "reason": nd["reason"],
    }

def all_at_once(category, convo_id):
    convo = category["train"][convo_id]
    problem = convo["question"]
    chat_content = render_history(convo["history"])

    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        '{\n'
        '    "agent a": "xx",\n'
        '    "agent b": "xxxx",\n'
        '    "agent c": "xxxxx",\n'
        '    "agent a": "xxxxxxx"\n'
        '}\n'
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction.\n"
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )

    pred = call_openai(prompt)
    return pred

client = OpenAI()

def call_openai(prompt, model="gpt-4o-mini"):
    system_msg = (
        "You are an expert judge for multi-agent failure attribution. "
        "Always return ONLY a JSON object with keys exactly: agent_name, step_number, reason. "
        "The step numbering is 1-based and each message in the conversation counts as one step in order."
    )
    user_prompt = (
        prompt
        + "\n\nFORMAT INSTRUCTIONS:\n"
          "- Return ONLY JSON (no prose, no markdown) with keys: "
          '{"agent_name": string, "step_number": integer, "reason": string}.'
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    try:
        parsed = _extract_json(raw)
        out = _normalize_and_validate(parsed)
        return out  # <- don't rebuild keys; normalization already handled it
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse/validate model output: {e}\n--- RAW OUTPUT START ---\n{raw}\n--- RAW OUTPUT END ---"
        )

if __name__ == "__main__":
    print(all_at_once(algo_ds, 0)) 