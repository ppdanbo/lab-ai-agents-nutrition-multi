from agents import FileSearchTool, HostedMCPTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from openai import AsyncOpenAI
from types import SimpleNamespace
from guardrails.runtime import load_config_bundle, instantiate_guardrails, run_guardrails
from pydantic import BaseModel
from openai.types.shared.reasoning import Reasoning

# Tool definitions
file_search = FileSearchTool(
  vector_store_ids=[
    "vs_694fe87330088191af6b15781016b0b3"
  ]
)
mcp = HostedMCPTool(tool_config={
  "type": "mcp",
  "server_label": "exa_search_mcp",
  "allowed_tools": [
    "web_search_exa"
  ],
  "require_approval": "never",
  "server_url": "https://mcp.exa.ai/mcp?exaApiKey=f9e56ddf-911c-4808-ba8e-7e0dba4c2e3d"
})
# Shared client for guardrails and file search
client = AsyncOpenAI()
ctx = SimpleNamespace(guardrail_llm=client)
# Guardrails definitions
guardrails_config = {
  "guardrails": [
    { "name": "Contains PII", "config": { "block": False, "detect_encoded_pii": True, "entities": ["CREDIT_CARD", "US_BANK_NUMBER", "US_PASSPORT", "US_SSN"] } },
    { "name": "Moderation", "config": { "categories": ["sexual/minors", "hate/threatening", "harassment/threatening", "self-harm/instructions", "violence", "violence/graphic", "illicit/violent"] } },
    { "name": "Jailbreak", "config": { "model": "gpt-4.1-mini", "confidence_threshold": 0.7 } }
  ]
}
def guardrails_has_tripwire(results):
    return any((hasattr(r, "tripwire_triggered") and (r.tripwire_triggered is True)) for r in (results or []))

def get_guardrail_safe_text(results, fallback_text):
    for r in (results or []):
        info = (r.info if hasattr(r, "info") else None) or {}
        if isinstance(info, dict) and ("checked_text" in info):
            return info.get("checked_text") or fallback_text
    pii = next(((r.info if hasattr(r, "info") else {}) for r in (results or []) if isinstance((r.info if hasattr(r, "info") else None) or {}, dict) and ("anonymized_text" in ((r.info if hasattr(r, "info") else None) or {}))), None)
    if isinstance(pii, dict) and ("anonymized_text" in pii):
        return pii.get("anonymized_text") or fallback_text
    return fallback_text

async def scrub_conversation_history(history, config):
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return
        pii_only = {"guardrails": [pii]}
        for msg in (history or []):
            content = (msg or {}).get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    res = await run_guardrails(ctx, part["text"], "text/plain", instantiate_guardrails(load_config_bundle(pii_only)), suppress_tripwire=True, raise_guardrail_errors=True)
                    part["text"] = get_guardrail_safe_text(res, part["text"])
    except Exception:
        pass

async def scrub_workflow_input(workflow, input_key, config):
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return
        if not isinstance(workflow, dict):
            return
        value = workflow.get(input_key)
        if not isinstance(value, str):
            return
        pii_only = {"guardrails": [pii]}
        res = await run_guardrails(ctx, value, "text/plain", instantiate_guardrails(load_config_bundle(pii_only)), suppress_tripwire=True, raise_guardrail_errors=True)
        workflow[input_key] = get_guardrail_safe_text(res, value)
    except Exception:
        pass

async def run_and_apply_guardrails(input_text, config, history, workflow):
    results = await run_guardrails(ctx, input_text, "text/plain", instantiate_guardrails(load_config_bundle(config)), suppress_tripwire=True, raise_guardrail_errors=True)
    guardrails = (config or {}).get("guardrails") or []
    mask_pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII" and ((g or {}).get("config") or {}).get("block") is False), None) is not None
    if mask_pii:
        await scrub_conversation_history(history, config)
        await scrub_workflow_input(workflow, "input_as_text", config)
        await scrub_workflow_input(workflow, "input_text", config)
    has_tripwire = guardrails_has_tripwire(results)
    safe_text = get_guardrail_safe_text(results, input_text)
    fail_output = build_guardrail_fail_output(results or [])
    pass_output = {"safe_text": (get_guardrail_safe_text(results, input_text) or input_text)}
    return {"results": results, "has_tripwire": has_tripwire, "safe_text": safe_text, "fail_output": fail_output, "pass_output": pass_output}

def build_guardrail_fail_output(results):
    def _get(name: str):
        for r in (results or []):
            info = (r.info if hasattr(r, "info") else None) or {}
            gname = (info.get("guardrail_name") if isinstance(info, dict) else None) or (info.get("guardrailName") if isinstance(info, dict) else None)
            if gname == name:
                return r
        return None
    pii, mod, jb, hal, nsfw, url, custom, pid = map(_get, ["Contains PII", "Moderation", "Jailbreak", "Hallucination Detection", "NSFW Text", "URL Filter", "Custom Prompt Check", "Prompt Injection Detection"])
    def _tripwire(r):
        return bool(r.tripwire_triggered)
    def _info(r):
        return r.info
    jb_info, hal_info, nsfw_info, url_info, custom_info, pid_info, mod_info, pii_info = map(_info, [jb, hal, nsfw, url, custom, pid, mod, pii])
    detected_entities = pii_info.get("detected_entities") if isinstance(pii_info, dict) else {}
    pii_counts = []
    if isinstance(detected_entities, dict):
        for k, v in detected_entities.items():
            if isinstance(v, list):
                pii_counts.append(f"{k}:{len(v)}")
    flagged_categories = (mod_info.get("flagged_categories") if isinstance(mod_info, dict) else None) or []
    
    return {
        "pii": { "failed": (len(pii_counts) > 0) or _tripwire(pii), "detected_counts": pii_counts },
        "moderation": { "failed": _tripwire(mod) or (len(flagged_categories) > 0), "flagged_categories": flagged_categories },
        "jailbreak": { "failed": _tripwire(jb) },
        "hallucination": { "failed": _tripwire(hal), "reasoning": (hal_info.get("reasoning") if isinstance(hal_info, dict) else None), "hallucination_type": (hal_info.get("hallucination_type") if isinstance(hal_info, dict) else None), "hallucinated_statements": (hal_info.get("hallucinated_statements") if isinstance(hal_info, dict) else None), "verified_statements": (hal_info.get("verified_statements") if isinstance(hal_info, dict) else None) },
        "nsfw": { "failed": _tripwire(nsfw) },
        "url_filter": { "failed": _tripwire(url) },
        "custom_prompt_check": { "failed": _tripwire(custom) },
        "prompt_injection": { "failed": _tripwire(pid) },
    }
class TopicClassifierSchema(BaseModel):
  topic: str


class SportsCoachSchema(BaseModel):
  pass


calories_agent = Agent(
  name="Calories Agent",
  instructions="""You are a a calorie look up agent that takes ingredients on it's input looks 
If you are asked about calorie information, always read the data from the Calorie Database even if you know the calorie by heart as the Calorie Database has the most up-to-date information.

You strictly follow these steps: 
1) You always read the calorie information for the ingredients using the 'Calorie Database'  File Search tool, even if you know the calorie by heart. The 'Calorie Database' File search tool has the most update-to-date information. 
2)Once you have the calorie for each ingredients, you add them up to calculate the total calories. 
3)Then you fill in every part of the template below and respond.

This is how you response must look (md format):
'''
# {{meal name}}

_Serving size: {{serving size}}

_Total calories: {{total_calories}}
'''

\"\"\"""",
  model="gpt-5-nano",
  tools=[
    file_search
  ],
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="medium"
    )
  )
)


ingredients_agent = Agent(
  name="Ingredients Agent",
  instructions="""*Always use the 'web_search_exa'  tool to search the web for the ingredients for the meal before you search for calories in the Calorie Database. Use the tool even if you know the ingredients as the tool might have more  update-to-date information. 

*Given the name and the size for each ingredients for a two person serving

*Your ONLY job is to figure out the ingredients. Subsequent agents will take care of the calorie calculation. 

This is the response format you want to use (markdown). Don't add anything else to the response:
\"\"\"
Mean name: {{meal name}}

Ingredient list for two servings:
*{{ingredient name}}: {{ingredient size}}
\"\"\"""",
  model="gpt-5-nano",
  tools=[
    mcp
  ],
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low"
    )
  )
)


topic_classifier = Agent(
  name="Topic Classifier",
  instructions="""Classify the subject of the input which can be \"calories\", \"sport\" or \"other\". 

You need to  output a JSON object with the key \"topic\" based on the following conditions: 

If the input is about calories,  set to \"calories\" 

if the input is about sport, set to \"sport\"

If the input is neither calories nor sport,  set to \"other\" """,
  model="gpt-5-nano",
  output_type=TopicClassifierSchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="medium"
    )
  )
)


no_offtopic_agent = Agent(
  name="No-offtopic Agent",
  instructions="Politely inform the user that this workflow is focused exclusively on nutrition topics or sport exercise and cannot assist with other non-related inquires.",
  model="gpt-5-nano",
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="medium"
    )
  )
)


sports_coach = Agent(
  name="Sports Coach",
  instructions="You will be an excellent Sport Coach, and you will get simple info about the person e.g. age and male or female.  Based on those info, you will develop a 30 minutes morning exercise plan",
  model="gpt-5-nano",
  output_type=SportsCoachSchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low"
    )
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("3-multi-agents"):
    state = {

    }
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    guardrails_input_text = workflow["input_as_text"]
    guardrails_result = await run_and_apply_guardrails(guardrails_input_text, guardrails_config, conversation_history, workflow)
    guardrails_hastripwire = guardrails_result["has_tripwire"]
    guardrails_anonymizedtext = guardrails_result["safe_text"]
    guardrails_output = (guardrails_hastripwire and guardrails_result["fail_output"]) or guardrails_result["pass_output"]
    if guardrails_hastripwire:
      no_offtopic_agent_result_temp = await Runner.run(
        no_offtopic_agent,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
        })
      )

      conversation_history.extend([item.to_input_item() for item in no_offtopic_agent_result_temp.new_items])

      no_offtopic_agent_result = {
        "output_text": no_offtopic_agent_result_temp.final_output_as(str)
      }
    else:
      topic_classifier_result_temp = await Runner.run(
        topic_classifier,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
        })
      )

      conversation_history.extend([item.to_input_item() for item in topic_classifier_result_temp.new_items])

      topic_classifier_result = {
        "output_text": topic_classifier_result_temp.final_output.json(),
        "output_parsed": topic_classifier_result_temp.final_output.model_dump()
      }
      if topic_classifier_result["output_parsed"]["topic"] == 'calories':
        ingredients_agent_result_temp = await Runner.run(
          ingredients_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
          })
        )

        conversation_history.extend([item.to_input_item() for item in ingredients_agent_result_temp.new_items])

        ingredients_agent_result = {
          "output_text": ingredients_agent_result_temp.final_output_as(str)
        }
        calories_agent_result_temp = await Runner.run(
          calories_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
          })
        )

        conversation_history.extend([item.to_input_item() for item in calories_agent_result_temp.new_items])

        calories_agent_result = {
          "output_text": calories_agent_result_temp.final_output_as(str)
        }
      elif topic_classifier_result["output_parsed"]["topic"] == 'sport':
        sports_coach_result_temp = await Runner.run(
          sports_coach,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
          })
        )

        conversation_history.extend([item.to_input_item() for item in sports_coach_result_temp.new_items])

        sports_coach_result = {
          "output_text": sports_coach_result_temp.final_output.json(),
          "output_parsed": sports_coach_result_temp.final_output.model_dump()
        }
      else:
        no_offtopic_agent_result_temp = await Runner.run(
          no_offtopic_agent,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
          })
        )

        conversation_history.extend([item.to_input_item() for item in no_offtopic_agent_result_temp.new_items])

        no_offtopic_agent_result = {
          "output_text": no_offtopic_agent_result_temp.final_output_as(str)
        }
