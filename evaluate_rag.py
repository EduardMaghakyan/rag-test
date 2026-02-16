import logging
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from ingest import _get_embeddings, get_llm
from rag import SYSTEM_PROMPT, RAGChain
import random

logger = logging.getLogger(__name__)

TEST_CASES = [
    {
        "question": "What is the core problem AGENT+P addresses in LLM-based UI agents?",
        "answer": "AGENT+P addresses the issue where LLM-based UI agents suffer from hallucinations in long-horizon tasks because they do not understand the global UI transition structure.",
        "references": [
            "Large Language Model (LLM)-based UI agents show great promise for UI automation but often hallucinate in long-horizon tasks due to their lack of understanding of the global UI transition structure. To address this, we introduce AGENT+P..."
        ],
        "ground_truth": "AGENT+P addresses the problem of LLM-based UI agents hallucinating in long-horizon tasks due to a lack of understanding of the global UI transition structure.",
    },
    {
        "question": "How does AGENT+P formulate the UI automation problem to solve it?",
        "answer": "It formulates the task as a pathfinding problem on a UI Transition Graph (UTG). This allows it to use symbolic planners to generate optimal plans.",
        "references": [
            "Specifically, we model an app's UI transition structure as a UI Transition Graph (UTG), which allows us to reformulate the UI automation task as a pathfinding problem on the UTG.",
            "This further enables an off-the-shelf symbolic planner to generate a provably correct and optimal high-level plan...",
        ],
        "ground_truth": "It reformulates the UI automation task as a pathfinding problem on a UI Transition Graph (UTG), enabling the use of symbolic planners.",
    },
    {
        "question": "What are the four beneficial reasoning behaviors identified for agentic search in the paper 'Beneficial Reasoning Behaviors'?",
        "answer": "The four behaviors are Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery.",
        "references": [
            "Through this process, we identify four reasoning behaviors critical for agentic search: Information Verification (validating results across sources), Authority Evaluation (assessing reliability...), Adaptive Search (modifying strategies dynamically), and Error Recovery (detecting and correcting mistakes)."
        ],
        "ground_truth": "The four beneficial reasoning behaviors are Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery.",
    },
    {
        "question": "What is 'Behavior Priming' in the context of agentic search?",
        "answer": "Behavior Priming is a technique where the model is fine-tuned using trajectories that demonstrate beneficial reasoning behaviors before running reinforcement learning.",
        "references": [
            "We propose a technique called Behavior Priming... It synthesizes agentic search trajectories that exhibit these four behaviors and integrates them into the agentic search model through supervised fine-tuning (SFT), followed by standard reinforcement learning (RL)."
        ],
        "ground_truth": "Behavior Priming is a method of instilling beneficial reasoning behaviors into an agent via supervised fine-tuning (SFT) on curated trajectories prior to reinforcement learning.",
    },
    {
        "question": "What are the three core components of the TOOLMEM framework?",
        "answer": "TOOLMEM consists of a structured capability memory, a feedback generation process, and a dynamic memory update mechanism.",
        "references": [
            "TOOLMEM is built upon three core components: (1) a structured capability memory initialized with a taxonomy... (2) a feedback generation process that evaluates tool outputs... and (3) a dynamic memory update mechanism that incorporates new experiences..."
        ],
        "ground_truth": "The three core components are: 1) a structured capability memory, 2) a feedback generation process, and 3) a dynamic memory update mechanism.",
    },
    {
        "question": "How does TOOLMEM improve tool selection at inference time?",
        "answer": "At inference, TOOLMEM retrieves relevant memory entries about tool capabilities, allowing the agent to choose the best tool for the specific task.",
        "references": [
            "During inference, TOOLMEM-augmented agents retrieve relevant memory entries based on the current task and inject them into the input context, enabling more accurate tool selection and solution generation."
        ],
        "ground_truth": "It retrieves relevant memory entries regarding tool strengths and weaknesses based on the current task, enabling the agent to select the best-performing tool.",
    },
    {
        "question": "What is the AJAN framework used for?",
        "answer": "AJAN is used to engineer multi-agent systems where knowledge is stored in RDF/OWL and behaviors are defined using SPARQL Behavior Trees.",
        "references": [
            "The AJAN framework allows to engineer multi-agent systems based on these standards. In particular, agent knowledge is represented in RDF/RDFS and OWL, while agent behavior models are defined with Behavior Trees and SPARQL..."
        ],
        "ground_truth": "AJAN is a framework for engineering semantic Web-enabled multi-agent systems, utilizing RDF for knowledge representation and SPARQL Behavior Trees for agent behavior.",
    },
    {
        "question": "What function does the 'Behaviors tab' serve in the AJAN-Editor?",
        "answer": "The Behaviors tab provides a graphical editor for creating SPARQL Behavior Trees (SBTs) via drag-and-drop.",
        "references": [
            "The Behaviors tab (cf. Fig. 1) offers a graphical editor for modeling SBTs using drag-and-drop of SBT nodes like composites, decorators, and leaves."
        ],
        "ground_truth": "It offers a graphical editor for modeling SPARQL Behavior Trees (SBTs) using drag-and-drop functionality.",
    },
    {
        "question": "What is the core innovation of the ProSEA framework regarding exploration?",
        "answer": "ProSEA uses two-dimensional exploration: a manager agent explores the problem in breadth (strategy), while expert agents explore in depth (reasoning).",
        "references": [
            "ProSEA's core innovation lies in its ability to explore solution spaces through both breadth and depth simultaneously.",
            "The design will allow ProSEA to tackle problems through two-dimensional exploration: the manager explores solution spaces in breadth through task decomposition... while experts explore in depth through iterative reasoning...",
        ],
        "ground_truth": "ProSEA's core innovation is simultaneous breadth and depth exploration, where a manager handles breadth (task decomposition/planning) and experts handle depth (domain reasoning).",
    },
    {
        "question": "How does ProSEA handle human collaboration?",
        "answer": "ProSEA supports human collaboration by allowing expert agents to request human assistance during exploration without changing the architecture.",
        "references": [
            "ProSEA incorporates human collaboration as an integral part of the exploration process.",
            "In collaborative mode, Expert Agents can seamlessly integrate human expertise by requesting assistance when encountering particularly challenging decisions...",
        ],
        "ground_truth": "It supports human-in-the-loop collaboration where agents can proactively seek human assistance or guidance during the exploration process.",
    },
    {
        "question": "What is the primary function of the AgentAsk module?",
        "answer": "AgentAsk is a clarification module that detects potential errors in inter-agent messages and asks minimal questions to stop the errors from spreading.",
        "references": [
            "We propose AgentAsk, a lightweight and plug-and-play clarification module that treats every inter-agent message as a potential failure point and inserts minimally necessary questions to arrest error propagation."
        ],
        "ground_truth": "AgentAsk monitors inter-agent messages (handoffs) and inserts minimal clarifying questions to prevent error propagation.",
    },
    {
        "question": "What are the four types of edge-level errors identified in the AgentAsk taxonomy?",
        "answer": "The four error types are Data Gap, Referential Drift, Signal Corruption, and Capability Gap.",
        "references": [
            "In our annotated corpus... Data Gap 29.1%, Referential Drift 27.3%, Signal Corruption 36.8%, and Capability Gap 6.8%.",
            "Figure 5: The case of our error taxonomy... DG, RD, SC, CG",
        ],
        "ground_truth": "The four error types are Data Gap (DG), Signal Corruption (SC), Referential Drift (RD), and Capability Gap (CG).",
    },
    {
        "question": "What does the 'blame attribution' methodology allow in the Traceability and Accountability study?",
        "answer": "It allows the system to determine which agent repaired an error or harmed a correct solution by monitoring correctness at every stage of the pipeline.",
        "references": [
            "We employ a blame attribution methodology to monitor the correctness of a solution as it passes through a Planner -> Executor -> Critic sequence.",
            "This allows us to quantify novel, role-specific behaviors such as repair... and harm...",
        ],
        "ground_truth": "It allows the quantification of role-specific behaviors, specifically identifying when an agent repairs an error from a previous stage or harms a previously correct output.",
    },
    {
        "question": "In the Traceability study, which role was found to be the primary source of unrecoverable pipeline failures?",
        "answer": "The Planner role was the primary source of unrecoverable failures.",
        "references": [
            "Our analysis reveals that the vast majority of unrecoverable pipeline failures originate from a flawed plan created in the very first step.",
            "Planner error rate is the strongest predictor of pipeline failure.",
        ],
        "ground_truth": "The Planner role (unrecoverable failures largely originate from a flawed initial plan).",
    },
    {
        "question": "What is the 'hierarchical citation graph' in SurveyG?",
        "answer": "It is a graph where nodes are papers and edges represent citations and semantic similarity. It is organized into three layers: Foundation, Development, and Frontier.",
        "references": [
            "SurveyG... integrates hierarchical citation graph, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness...",
            "The graph is organized into three layers: Foundation, Development, and Frontier...",
        ],
        "ground_truth": "It is a graph representation where nodes are papers and edges represent citation/semantic links, organized into three layers: Foundation, Development, and Frontier.",
    },
    {
        "question": "How does SurveyG differ from existing survey generation frameworks?",
        "answer": "Unlike other frameworks that summarize flat lists of papers, SurveyG uses a hierarchical graph to capture the structural and evolutionary relationships between papers.",
        "references": [
            "Existing approaches typically extract content... and prompt LLMs to summarize them directly... overlooking the structural relationships...",
            "SurveyG... embedding structural and contextual knowledge... By combining horizontal search within layers and vertical depth traversal...",
        ],
        "ground_truth": "Existing frameworks typically summarize flat collections of papers, whereas SurveyG uses a hierarchical citation graph to model structural relationships and research evolution.",
    },
    {
        "question": "What is the ACONIC framework?",
        "answer": "ACONIC is a framework that decomposes complex LLM tasks by modeling them as constraint satisfaction problems.",
        "references": [
            "Analysis of CONstraint-Induced Complexity (ACONIC), which models the task as a constraint problem and leveraging formal complexity measures to guide decomposition."
        ],
        "ground_truth": "ACONIC is a framework for systematic task decomposition that models tasks as constraint satisfaction problems (CSPs) and uses complexity measures to guide the decomposition.",
    },
    {
        "question": "What metric does ACONIC use to quantify task complexity?",
        "answer": "ACONIC uses the treewidth of the induced constraint graph to measure complexity.",
        "references": [
            "We use properties of the induced constraint graph (graph size and treewidth) as measures of task complexity."
        ],
        "ground_truth": "It uses the treewidth (and graph size) of the induced constraint graph.",
    },
    {
        "question": "In the Agent+P paper, what is a UTG?",
        "answer": "A UTG is a UI Transition Graph that models an app's structure, where nodes are UI screens and edges are user actions.",
        "references": [
            "Definition 3 (UI Transition Graph). A UTG for an app is a directed graph G=(U, T, E) that models the transition structure of the app... nodes representing UIs and edges representing user-triggered UI transitions."
        ],
        "ground_truth": "UTG stands for UI Transition Graph. It is a directed graph modeling an app's structure with nodes as UI states and edges as user-triggered transitions.",
    },
    {
        "question": "What is E-GRPO used for in AgentAsk?",
        "answer": "E-GRPO is a reinforcement learning objective used to train AgentAsk to balance accuracy with cost and latency.",
        "references": [
            "...optimizing online with E-GRPO, a reinforcement learning objective that balances accuracy, latency, and cost."
        ],
        "ground_truth": "E-GRPO is a reinforcement learning objective used to optimize the AgentAsk clarifier to balance accuracy, latency, and cost.",
    },
]


def collect_samples(test_cases: list[dict[str, str]]) -> list[SingleTurnSample]:
    rag_chain = RAGChain()
    samples = []

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        logger.info("Processing question %d/%d: %s", i, len(test_cases), question)
        try:
            docs = rag_chain.retriever.invoke(question)
            retrieved_contexts = [doc.page_content for doc in docs]
            context = RAGChain.format_context(docs)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
                HumanMessage(content=question),
            ]
            response = str(rag_chain.llm.invoke(messages).content)

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=retrieved_contexts,
                    response=response,
                    reference=case["ground_truth"],
                )
            )
        except Exception:
            logger.exception("Failed on question %d, skipping: %s", i, question)

    return samples


def run_evaluation() -> None:
    sample_test_cases = random.choices(TEST_CASES, 5)
    
    logger.info("Collecting RAG responses for %d test cases...", len(sample_test_cases))
    samples = collect_samples(sample_test_cases)

    if not samples:
        print("No samples collected — all test cases failed. Exiting.")
        return

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        # Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        # FactualCorrectness(),
    ]

    logger.info("Running Ragas evaluation...")
    run_config = RunConfig(max_workers=2, timeout=300)
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=get_llm(),
            embeddings=_get_embeddings(),
            run_config=run_config,
        )
    except Exception:
        logger.exception("Ragas evaluation failed")
        return

    print("\n=== Aggregate Scores ===")
    print(f"  {result}")

    print("\n=== Per-Question Breakdown ===")
    df = result.to_pandas()
    print(df.to_string(index=False))

    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_evaluation()
