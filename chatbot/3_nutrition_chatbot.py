import chainlit as cl
import dotenv

dotenv.load_dotenv()

from openai.types.responses import ResponseTextDeltaEvent
from agents import Runner, trace
from nutrition_agent import nutrition_agent


@cl.on_message
async def on_message(message: cl.Message):
    with trace("Nutrition Agent via chainlit"):
        result = Runner.run_streamed(
        nutrition_agent,
        message.content,
        )

        msg = cl.Message(content="")
        async for event in result.stream_events():
            # Stream final message text to screen
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                await msg.stream_token(token=event.data.delta)
                print(event.data.delta, end="", flush=True)

            elif (
                event.type == "raw_response_event"
                and hasattr(event.data, "item")
                and hasattr(event.data.item, "type")
                and event.data.item.type == "function_call"
                and len(event.data.item.arguments) > 0
            ):
                with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                    step.input = event.data.item.arguments
                    print(
                        f"\nTool call: {
                            event.data.item.name} with args: {
                            event.data.item.arguments}"
                    )

        await msg.update()
    