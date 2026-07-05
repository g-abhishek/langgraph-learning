import { initChatModel } from "langchain";
import {
  END,
  MemorySaver,
  MessagesValue,
  START,
  StateGraph,
  StateSchema,
} from "@langchain/langgraph";

// LLM used inside the graph node. LangChain converts a plain string input
// like "Hi, I am Abhishek" into a HumanMessage automatically.
const model = await initChatModel("openai:gpt-5.5");

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
  const response = await model.invoke(state.messages);
  // Return only the new AI message; MessagesValue appends it to history.
  return { messages: [response] };
}

// Persists graph state between separate invoke() calls.
// Without this, each invoke starts from scratch even with StateSchema.
const checkpointer = new MemorySaver();

// Build the graph: one node, linear flow START -> chat -> END.
// .compile() returns the runnable graph (the object that has .invoke()).
const graph = new StateGraph(State)
  .addNode("chat_bot_node", chatBotNode)
  .addEdge(START, "chat_bot_node")
  .addEdge("chat_bot_node", END)
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