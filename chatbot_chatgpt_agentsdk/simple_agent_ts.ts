import { fileSearchTool, Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";


// Tool definitions
const fileSearch = fileSearchTool([
  "vs_694fe87330088191af6b15781016b0b3"
])
const nutritionAgentDw = new Agent({
  name: "Nutrition Agent DW",
  instructions: `You are a helpful assistant giving out nutrition advice. You give concise answers.

If you are asked about calorie information, always read the data from the Calorie Database even if you know the calorie by heart as the Calorie Database has the most up-to-date information.

If you are asked about nutrition and health benefits information, always read the data from the Nutrition Database even if you know the benefits by heart as the Nutrition Database has the most up-to-date information.`,
  model: "gpt-5-nano",
  tools: [
    fileSearch
  ],
  modelSettings: {
    reasoning: {
      effort: "low"
    },
    store: true
  }
});

type WorkflowInput = { input_as_text: string };


// Main code entrypoint
export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("1-simple-nutrition-agent", async () => {
    const state = {

    };
    const conversationHistory: AgentInputItem[] = [
      { role: "user", content: [{ type: "input_text", text: workflow.input_as_text }] }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_694fe56b2b988190a2009f31b7b28082057b4bb9032aa813"
      }
    });
    const nutritionAgentDwResultTemp = await runner.run(
      nutritionAgentDw,
      [
        ...conversationHistory
      ]
    );
    conversationHistory.push(...nutritionAgentDwResultTemp.newItems.map((item) => item.rawItem));

    if (!nutritionAgentDwResultTemp.finalOutput) {
        throw new Error("Agent result is undefined");
    }

    const nutritionAgentDwResult = {
      output_text: nutritionAgentDwResultTemp.finalOutput ?? ""
    };
  });
}
