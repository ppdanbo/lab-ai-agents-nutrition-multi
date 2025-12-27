import { fileSearchTool, hostedMcpTool, Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";
import { OpenAI } from "openai";
import { runGuardrails } from "@openai/guardrails";
import { z } from "zod";


// Tool definitions
const fileSearch = fileSearchTool([
  "vs_694fe87330088191af6b15781016b0b3"
])
const mcp = hostedMcpTool({
  serverLabel: "exa_search_mcp",
  allowedTools: [
    "web_search_exa"
  ],
  requireApproval: "never",
  serverUrl: "https://mcp.exa.ai/mcp?exaApiKey=f9e56ddf-911c-4808-ba8e-7e0dba4c2e3d"
})

// Shared client for guardrails and file search
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Guardrails definitions
const guardrailsConfig = {
  guardrails: [
    { name: "Contains PII", config: { block: false, detect_encoded_pii: true, entities: ["CREDIT_CARD", "US_BANK_NUMBER", "US_PASSPORT", "US_SSN"] } },
    { name: "Moderation", config: { categories: ["sexual/minors", "hate/threatening", "harassment/threatening", "self-harm/instructions", "violence", "violence/graphic", "illicit/violent"] } },
    { name: "Jailbreak", config: { model: "gpt-4.1-mini", confidence_threshold: 0.7 } }
  ]
};
const context = { guardrailLlm: client };

function guardrailsHasTripwire(results: any[]): boolean {
    return (results ?? []).some((r) => r?.tripwireTriggered === true);
}

function getGuardrailSafeText(results: any[], fallbackText: string): string {
    for (const r of results ?? []) {
        if (r?.info && ("checked_text" in r.info)) {
            return r.info.checked_text ?? fallbackText;
        }
    }
    const pii = (results ?? []).find((r) => r?.info && "anonymized_text" in r.info);
    return pii?.info?.anonymized_text ?? fallbackText;
}

async function scrubConversationHistory(history: any[], piiOnly: any): Promise<void> {
    for (const msg of history ?? []) {
        const content = Array.isArray(msg?.content) ? msg.content : [];
        for (const part of content) {
            if (part && typeof part === "object" && part.type === "input_text" && typeof part.text === "string") {
                const res = await runGuardrails(part.text, piiOnly, context, true);
                part.text = getGuardrailSafeText(res, part.text);
            }
        }
    }
}

async function scrubWorkflowInput(workflow: any, inputKey: string, piiOnly: any): Promise<void> {
    if (!workflow || typeof workflow !== "object") return;
    const value = workflow?.[inputKey];
    if (typeof value !== "string") return;
    const res = await runGuardrails(value, piiOnly, context, true);
    workflow[inputKey] = getGuardrailSafeText(res, value);
}

async function runAndApplyGuardrails(inputText: string, config: any, history: any[], workflow: any) {
    const guardrails = Array.isArray(config?.guardrails) ? config.guardrails : [];
    const results = await runGuardrails(inputText, config, context, true);
    const shouldMaskPII = guardrails.find((g) => (g?.name === "Contains PII") && g?.config && g.config.block === false);
    if (shouldMaskPII) {
        const piiOnly = { guardrails: [shouldMaskPII] };
        await scrubConversationHistory(history, piiOnly);
        await scrubWorkflowInput(workflow, "input_as_text", piiOnly);
        await scrubWorkflowInput(workflow, "input_text", piiOnly);
    }
    const hasTripwire = guardrailsHasTripwire(results);
    const safeText = getGuardrailSafeText(results, inputText) ?? inputText;
    return { results, hasTripwire, safeText, failOutput: buildGuardrailFailOutput(results ?? []), passOutput: { safe_text: safeText } };
}

function buildGuardrailFailOutput(results: any[]) {
    const get = (name: string) => (results ?? []).find((r: any) => ((r?.info?.guardrail_name ?? r?.info?.guardrailName) === name));
    const pii = get("Contains PII"), mod = get("Moderation"), jb = get("Jailbreak"), hal = get("Hallucination Detection"), nsfw = get("NSFW Text"), url = get("URL Filter"), custom = get("Custom Prompt Check"), pid = get("Prompt Injection Detection"), piiCounts = Object.entries(pii?.info?.detected_entities ?? {}).filter(([, v]) => Array.isArray(v)).map(([k, v]) => k + ":" + v.length), conf = jb?.info?.confidence;
    return {
        pii: { failed: (piiCounts.length > 0) || pii?.tripwireTriggered === true, detected_counts: piiCounts },
        moderation: { failed: mod?.tripwireTriggered === true || ((mod?.info?.flagged_categories ?? []).length > 0), flagged_categories: mod?.info?.flagged_categories },
        jailbreak: { failed: jb?.tripwireTriggered === true },
        hallucination: { failed: hal?.tripwireTriggered === true, reasoning: hal?.info?.reasoning, hallucination_type: hal?.info?.hallucination_type, hallucinated_statements: hal?.info?.hallucinated_statements, verified_statements: hal?.info?.verified_statements },
        nsfw: { failed: nsfw?.tripwireTriggered === true },
        url_filter: { failed: url?.tripwireTriggered === true },
        custom_prompt_check: { failed: custom?.tripwireTriggered === true },
        prompt_injection: { failed: pid?.tripwireTriggered === true },
    };
}
const TopicClassifierSchema = z.object({ topic: z.enum(["calories", "sport", "other"]) });
const SportsCoachSchema = z.object({});
const caloriesAgent = new Agent({
  name: "Calories Agent",
  instructions: `You are a a calorie look up agent that takes ingredients on it's input looks 
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

\"\"\"`,
  model: "gpt-5-nano",
  tools: [
    fileSearch
  ],
  modelSettings: {
    reasoning: {
      effort: "medium"
    },
    store: true
  }
});

const ingredientsAgent = new Agent({
  name: "Ingredients Agent",
  instructions: `*Always use the 'web_search_exa'  tool to search the web for the ingredients for the meal before you search for calories in the Calorie Database. Use the tool even if you know the ingredients as the tool might have more  update-to-date information. 

*Given the name and the size for each ingredients for a two person serving

*Your ONLY job is to figure out the ingredients. Subsequent agents will take care of the calorie calculation. 

This is the response format you want to use (markdown). Don't add anything else to the response:
\"\"\"
Mean name: {{meal name}}

Ingredient list for two servings:
*{{ingredient name}}: {{ingredient size}}
\"\"\"`,
  model: "gpt-5-nano",
  tools: [
    mcp
  ],
  modelSettings: {
    reasoning: {
      effort: "low"
    },
    store: true
  }
});

const topicClassifier = new Agent({
  name: "Topic Classifier",
  instructions: `Classify the subject of the input which can be \"calories\", \"sport\" or \"other\". 

You need to  output a JSON object with the key \"topic\" based on the following conditions: 

If the input is about calories,  set to \"calories\" 

if the input is about sport, set to \"sport\"

If the input is neither calories nor sport,  set to \"other\" `,
  model: "gpt-5-nano",
  outputType: TopicClassifierSchema,
  modelSettings: {
    reasoning: {
      effort: "medium"
    },
    store: true
  }
});

const noOfftopicAgent = new Agent({
  name: "No-offtopic Agent",
  instructions: "Politely inform the user that this workflow is focused exclusively on nutrition topics or sport exercise and cannot assist with other non-related inquires.",
  model: "gpt-5-nano",
  modelSettings: {
    reasoning: {
      effort: "medium"
    },
    store: true
  }
});

const sportsCoach = new Agent({
  name: "Sports Coach",
  instructions: "You will be an excellent Sport Coach, and you will get simple info about the person e.g. age and male or female.  Based on those info, you will develop a 30 minutes morning exercise plan",
  model: "gpt-5-nano",
  outputType: SportsCoachSchema,
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
  return await withTrace("3-multi-agents", async () => {
    const state = {

    };
    const conversationHistory: AgentInputItem[] = [
      { role: "user", content: [{ type: "input_text", text: workflow.input_as_text }] }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_695012547d7c8190949ea05e4289c59c08a10cd53b5b5926"
      }
    });
    const guardrailsInputText = workflow.input_as_text;
    const { hasTripwire: guardrailsHasTripwire, safeText: guardrailsAnonymizedText, failOutput: guardrailsFailOutput, passOutput: guardrailsPassOutput } = await runAndApplyGuardrails(guardrailsInputText, guardrailsConfig, conversationHistory, workflow);
    const guardrailsOutput = (guardrailsHasTripwire ? guardrailsFailOutput : guardrailsPassOutput);
    if (guardrailsHasTripwire) {
      const noOfftopicAgentResultTemp = await runner.run(
        noOfftopicAgent,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...noOfftopicAgentResultTemp.newItems.map((item) => item.rawItem));

      if (!noOfftopicAgentResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const noOfftopicAgentResult = {
        output_text: noOfftopicAgentResultTemp.finalOutput ?? ""
      };
    } else {
      const topicClassifierResultTemp = await runner.run(
        topicClassifier,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...topicClassifierResultTemp.newItems.map((item) => item.rawItem));

      if (!topicClassifierResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const topicClassifierResult = {
        output_text: JSON.stringify(topicClassifierResultTemp.finalOutput),
        output_parsed: topicClassifierResultTemp.finalOutput
      };
      if (topicClassifierResult.output_parsed.topic == 'calories') {
        const ingredientsAgentResultTemp = await runner.run(
          ingredientsAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...ingredientsAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!ingredientsAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const ingredientsAgentResult = {
          output_text: ingredientsAgentResultTemp.finalOutput ?? ""
        };
        const caloriesAgentResultTemp = await runner.run(
          caloriesAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...caloriesAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!caloriesAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const caloriesAgentResult = {
          output_text: caloriesAgentResultTemp.finalOutput ?? ""
        };
      } else if (topicClassifierResult.output_parsed.topic == 'sport') {
        const sportsCoachResultTemp = await runner.run(
          sportsCoach,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...sportsCoachResultTemp.newItems.map((item) => item.rawItem));

        if (!sportsCoachResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const sportsCoachResult = {
          output_text: JSON.stringify(sportsCoachResultTemp.finalOutput),
          output_parsed: sportsCoachResultTemp.finalOutput
        };
      } else {
        const noOfftopicAgentResultTemp = await runner.run(
          noOfftopicAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...noOfftopicAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!noOfftopicAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const noOfftopicAgentResult = {
          output_text: noOfftopicAgentResultTemp.finalOutput ?? ""
        };
      }
    }
  });
}
