from typing import TypedDict
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from tenacity import retry, wait_exponential, stop_after_attempt

from config import Config
from tools import search_documents, get_full_document, get_knowledge_base


class SynthesisState(TypedDict):
    question: str
    research: str
    draft: str
    final: str
    review_feedback: str
    review_passed: bool
    attempts: int


class ResearchPipeline:
    """Orchestrates the Researcher -> Writer -> Reviewer agent workflow."""

    def __init__(self):
        self._model = ChatGroq(model=Config.MODEL_NAME, temperature=Config.MODEL_TEMPERATURE)
        self._kb = get_knowledge_base()

        self._research_agent = create_agent(
            self._model,
            tools=[search_documents, get_full_document],
            system_prompt=(
                "You are a research specialist working with an ALREADY-LOADED knowledge base.\n\n"
                "Choose the RIGHT tool:\n"
                "- Use get_full_document when asked to review, analyze, summarize, or evaluate an "
                "ENTIRE document (e.g. 'how strong is this CV', 'summarize this report').\n"
                "- Use search_documents for SPECIFIC fact-finding questions about details.\n\n"
                "Never tell the user to upload or share a document — it is already in the knowledge base."
            )
        )

        self._graph = self._build_graph()

    @retry(wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
           stop=stop_after_attempt(Config.RETRY_MAX_ATTEMPTS))
    def _researcher_node(self, state: SynthesisState) -> dict:
        doc_list = self._kb.list_document_names()
        context_note = f"Available documents in knowledge base: {doc_list}\n\n" if doc_list else ""
        result = self._research_agent.invoke({"messages": [("human", context_note + state["question"])]})
        return {"research": result["messages"][-1].content}

    @retry(wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
           stop=stop_after_attempt(Config.RETRY_MAX_ATTEMPTS))
    def _writer_node(self, state: SynthesisState) -> dict:
        feedback_note = f"\n\nAddress this feedback: {state['review_feedback']}" if state.get("review_feedback") else ""
        response = self._model.invoke([
            ("system", "You are a clear, concise writer. Turn research findings into a well-structured "
                       "answer. Keep source citations from the research."),
            ("human", f"Question: {state['question']}\nResearch:\n{state['research']}{feedback_note}")
        ])
        return {"draft": response.content}

    @retry(wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
           stop=stop_after_attempt(Config.RETRY_MAX_ATTEMPTS))
    def _reviewer_node(self, state: SynthesisState) -> dict:
        response = self._model.invoke([
            ("system", "You are a strict reviewer. Check if the answer is accurate, well-cited, and "
                       "clearly written.\nReply in this EXACT format:\nPASS: yes or no\n"
                       "FEEDBACK: brief feedback if no, or \"Looks good\" if yes"),
            ("human", state["draft"])
        ])
        content = response.content
        passed = "pass: yes" in content.lower()
        feedback_line = [l for l in content.split("\n") if "feedback" in l.lower()]
        feedback = feedback_line[0].split(":", 1)[1].strip() if feedback_line else ""

        return {
            "final": state["draft"],
            "review_passed": passed,
            "review_feedback": feedback,
            "attempts": state["attempts"] + 1
        }

    @staticmethod
    def _route_after_review(state: SynthesisState) -> str:
        if state["review_passed"] or state["attempts"] >= Config.MAX_REVIEW_ATTEMPTS:
            return "done"
        return "revise"

    def _build_graph(self):
        builder = StateGraph(SynthesisState)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("writer", self._writer_node)
        builder.add_node("reviewer", self._reviewer_node)

        builder.add_edge(START, "researcher")
        builder.add_edge("researcher", "writer")
        builder.add_edge("writer", "reviewer")
        builder.add_conditional_edges("reviewer", self._route_after_review, {
            "revise": "writer",
            "done": END
        })
        return builder.compile()

    def run(self, question: str) -> dict:
        """Run the full research pipeline for a given question."""
        return self._graph.invoke({
            "question": question,
            "research": "",
            "draft": "",
            "final": "",
            "review_feedback": "",
            "review_passed": False,
            "attempts": 0
        })