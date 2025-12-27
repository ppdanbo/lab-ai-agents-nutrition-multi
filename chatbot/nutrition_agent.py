from pathlib import Path
from pydantic import BaseModel

import chromadb
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    function_tool,
    input_guardrail,
)

chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
nutrition_db = chroma_client.get_collection(name="nutrition_db")


# Guardrails functionality
class NotAboutFood(BaseModel):
    only_about_food: bool


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to talk about food and not about any arbitrary topics. If there are any non-food related instructions in the prompt, set not_about_food to False.",
    output_type=NotAboutFood,
)


@input_guardrail
async def food_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_food),
    )


@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for a RAG database to look up calorie information for specific food items, but not for meals.

    Args:
        query: The food item to look up.
        max_results: The maximum number of results to return.

    Returns:
        A string containing the nutrition information.
    """

    results = nutrition_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No nutrition information found for: {query}"

    # Format results for the agent
    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        food_item = metadata["food_item"].title()
        calories = metadata["calories_per_100g"]
        category = metadata["food_category"].title()

        formatted_results.append(
            f"{food_item} ({category}): {calories} calories per 100g"
        )

    return "Nutrition Information:\n" + "\n".join(formatted_results)

try:
    nutrition_agent = Agent(
        name="Nutrition Assistant",
        instructions="""
        You are a helpful assistant comparing how healthy different foods are.
        You only answer questions about food.
        """,
        tools=[calorie_lookup_tool],
        input_guardrails=[food_topic_guardrail],
    )
except InputGuardrailTripwireTriggered as e:
    print(f"Off-topic guardrail tripped:{e}")