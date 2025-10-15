from openai import OpenAI
import json, subprocess, os, datetime

client = OpenAI()

def spawn_agent(role, goal, context):
    prompt = f"""
You are a specialized AG-1 sub-agent.

Role: {role}
Goal: {goal}
Context: {context}

You must output a JSON object with fields:
- plan: ordered list of concrete steps
- code: full code blocks for any new files
- commands: terminal commands to execute
- next: short description of what to do next
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    data = json.loads(resp.choices[0].message.content)
    timestamp = datetime.datetime.now().isoformat()
    with open(f"../data/{role}_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
    return data
