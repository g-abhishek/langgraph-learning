import { AIMessage, initChatModel, tool } from "langchain";
import {
  END,
  MemorySaver,
  MessagesValue,
  START,
  StateGraph,
  StateSchema,
} from "@langchain/langgraph";
import { TavilySearch } from "@langchain/tavily";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import z from "zod";

// LLM used inside the graph node. LangChain converts a plain string input
// like "Hi, I am Abhishek" into a HumanMessage automatically.
const model = await initChatModel("openai:gpt-5.5");

const addNumbersTool = tool(({ a, b }) => {
    return a + b;
  },
  {
    name: "add_numbers",
    description: "This tool is used to add two numbers",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
})


// Tavily web search tool.
// `topic: "news"` + `timeRange` helps the model fetch recent news.
const tavilySearchTool = new TavilySearch({
  maxResults: 2,
  timeRange: "week",
});

// Bind the tool to the model so it can decide when to search.
const modelWithTools = model.bindTools([tavilySearchTool, addNumbersTool]);

// Define the shape of graph state.
// `messages` uses MessagesValue so new messages are APPENDED (reduced),
// not replaced, when a node returns { messages: [...] }.
const State = new StateSchema({
  messages: MessagesValue,
});

// A graph node: LangGraph calls this with the current state and merges
// the return value back into state via the reducer above.
async function chatBotNode(state) {
  // Pass full conversation history to the model for this turn.
  const response = await modelWithTools.invoke(state.messages);
  // Return only the new AI message; MessagesValue appends it to history.
  return { messages: [response] };
}

const toolNode = new ToolNode([tavilySearchTool, addNumbersTool])

// Persists graph state between separate invoke() calls.
// Without this, each invoke starts from scratch even with StateSchema.
const checkpointer = new MemorySaver();

function shouldContinue(state) {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage && AIMessage.isInstance(lastMessage) && lastMessage.tool_calls?.length) {
    return "tool_node";
  }

  return END
}


// Build the graph: one node, linear flow START -> chat -> END.
// .compile() returns the runnable graph (the object that has .invoke()).
const graph = new StateGraph(State)
  .addNode("chat_bot_node", chatBotNode)
  .addNode("tool_node", toolNode)
  .addEdge(START, "chat_bot_node")
  .addConditionalEdges("chat_bot_node", shouldContinue)
  .addEdge("tool_node", "chat_bot_node")
  .compile({ checkpointer });

// Same thread_id = same conversation across multiple invokes.
// Different thread_id = separate conversation / memory.
const config = {
  configurable: { thread_id: "chat-1" },
};

// Turn 1: saves checkpoint for thread "chat-1" after this run.
const result1 = await graph.invoke(
  { messages: "Hi, I am Abhishek" },
  config,
);
console.log("result1 >>", result1.messages);

// Turn 2: loads prior state for "chat-1", appends the new user message,
// so the model can see "Hi, I am Abhishek" from turn 1.
const result2 = await graph.invoke(
  { messages: "What is my name?" },
  config,
);
console.log("result2 >>", result2.messages);

const result3 = await graph.invoke({ messages: "what is the weather in mumbai today?" }, config);
console.log("result3 >>", result3.messages[result3.messages.length - 1].content);

const result4 = await graph.invoke({ messages: "what will be the weather in mumbai tomorrow?" }, config);
console.log("result4 >>", result4.messages[result4.messages.length - 1].content);

const result5 = await graph.invoke({ messages: "add 10 and 20" }, config);
console.log("result5 >>", result5.messages[result5.messages.length - 1].content);

const result6 = await graph.invoke({ messages: "what is latest AI news? and add 10 and 20" }, config);
console.log("result6 >>", result6.messages);