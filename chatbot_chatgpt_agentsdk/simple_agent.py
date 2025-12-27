from agents import FileSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel

# Tool definitions
file_search = FileSearchTool(
  vector_store_ids=[
    "vs_694fe87330088191af6b15781016b0b3"
  ]
)
nutrition_agent_dw = Agent(
  name="Nutrition Agent DW",
  instructions="""You are a helpful assistant giving out nutrition advice. You give concise answers.

If you are asked about calorie information, always read the data from the Calorie Database even if you know the calorie by heart as the Calorie Database has the most up-to-date information.

If you are asked about nutrition and health benefits information, always read the data from the Nutrition Database even if you know the benefits by heart as the Nutrition Database has the most up-to-date information.""",
  model="gpt-5-nano",
  tools=[
    file_search
  ],
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
  with trace("1-simple-nutrition-agent"):
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
    nutrition_agent_dw_result_temp = await Runner.run(
      nutrition_agent_dw,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_694fe56b2b988190a2009f31b7b28082057b4bb9032aa813"
      })
    )

    conversation_history.extend([item.to_input_item() for item in nutrition_agent_dw_result_temp.new_items])

    nutrition_agent_dw_result = {
      "output_text": nutrition_agent_dw_result_temp.final_output_as(str)
    }
