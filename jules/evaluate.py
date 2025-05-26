import argparse
import os
import sys

# DeepEval imports
try:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import DAGMetric
    from deepeval.metrics.dag import DeepAcyclicGraph, BinaryJudgementNode, NonBinaryJudgementNode, VerdictNode
except ImportError as e:
    print(f"ERROR: Failed to import from deepeval: {e}")
    print("Please ensure DeepEval is installed correctly (e.g., 'pip install deepeval').")
    sys.exit(1)

# Local app imports
try:
    from app import SummaryWriter # TopicResearcher is not run, PREDEFINED_RESEARCH_TEXT is used directly
    # RESEARCH_FINDINGS from app.py is equivalent to PREDEFINED_RESEARCH_TEXT here.
    # We define it locally for clarity and to ensure it's exactly as specified.
except ImportError as e:
    print(f"ERROR: Failed to import from local 'app' module: {e}")
    print("Ensure 'app.py' exists in the same directory and is importable.")
    sys.exit(1)

# Agent runner import
try:
    from agents import Runner
except ImportError as e:
    print(f"ERROR: Failed to import Runner from 'agents': {e}")
    print("Ensure the 'openai-agents-python' SDK is installed and importable.")
    sys.exit(1)

# Predefined text, identical to RESEARCH_FINDINGS in app.py
PREDEFINED_RESEARCH_TEXT = """
The field of quantum computing has seen significant advancements in recent years.
Researchers are exploring various qubit technologies, including superconducting circuits,
trapped ions, and photonic systems. Quantum algorithms, such_as Shor's algorithm for
factoring and Grover's algorithm for search, promise to solve certain problems
intractable for classical computers. However, building fault-tolerant quantum computers
remains a major challenge, requiring breakthroughs in error correction and qubit coherence.
The potential applications span drug discovery, materials science, financial modeling,
and cryptography.
"""

# 1. Define DAGMetric Nodes

# Node for Summarization Quality
SummarizationQualityNode = NonBinaryJudgementNode(
    name="SummarizationQuality", # Name for the node
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    criteria="Evaluate the quality of the `actual_output` (a summary) in relation to the `expected_output` (the original text). Consider accuracy (does the summary misrepresent or add new info?), completeness (does it cover the main points of original text?), conciseness (is it free of fluff?), and readability (is it well-written and easy to understand?).",
    children=[
        VerdictNode(verdict="Excellent summary, captures all key points concisely and accurately.", score=10),
        VerdictNode(verdict="Good summary, captures most key points, minor omissions or verbosity.", score=7),
        VerdictNode(verdict="Fair summary, captures some key points but misses important details or is unclear.", score=4),
        VerdictNode(verdict="Poor summary, significant inaccuracies, omissions, or very difficult to understand.", score=1),
        VerdictNode(verdict="No meaningful summary provided or completely irrelevant.", score=0)
    ]
)

# Node for Initial Output Check
InitialOutputCheckNode = BinaryJudgementNode(
    name="InitialOutputRelevance", # Name for the node
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    criteria="Given the research topic in `input`, does the `expected_output` (the researcher's text that was provided for summarization) appear to be relevant to the topic and is it substantial enough for summarization? For example, does it mention key terms from the `input` topic or generally align with it?",
    children=[
        VerdictNode(verdict=True, child=SummarizationQualityNode), # If True, proceed to SummarizationQualityNode
        VerdictNode(verdict=False, score=0)                      # If False, assign score 0 and terminate this path
    ]
)

# 2. Define the DAG and the Metric
dag = DeepAcyclicGraph(root_nodes=[InitialOutputCheckNode])
agent_quality_metric = DAGMetric(
    name="MultiAgentSystemQuality", 
    dag=dag, 
    model="gpt-4o" # Specify a capable model for judgment. "gpt-3.5-turbo" can also be used.
    # Ensure the model used here has access via the API key.
)

# 3. Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the multi-agent system using DeepEval.")
    parser.add_argument(
        "--topic", 
        type=str, 
        default="Quantum Computing", 
        help="The research topic to simulate."
    )
    args = parser.parse_args()
    research_topic_input = args.topic

    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("DeepEval's DAGMetric requires an OpenAI API key to function as it uses an LLM for evaluation.")
        sys.exit(1)
    else:
        print("INFO: OPENAI_API_KEY is set. Proceeding with evaluation.")

    # Instantiate SummaryWriter agent from app.py
    # TopicResearcher is not run; PREDEFINED_RESEARCH_TEXT is used directly as its output.
    summary_writer_agent = SummaryWriter # Imported directly

    # Use PREDEFINED_RESEARCH_TEXT directly as the output of the (notional) TopicResearcher
    research_text = PREDEFINED_RESEARCH_TEXT
    print(f"\nUsing predefined research text for topic '{research_topic_input}'. Length: {len(research_text)} chars.")

    # Run SummaryWriter agent
    print("\nRunning SummaryWriter agent...")
    final_summary = None
    try:
        # The input to SummaryWriter is the research_text
        writer_agent_output = Runner.run_sync(summary_writer_agent, research_text)
        
        if hasattr(writer_agent_output, 'final_output') and isinstance(writer_agent_output.final_output, str):
            final_summary = writer_agent_output.final_output
        elif isinstance(writer_agent_output, str):
            final_summary = writer_agent_output
        else:
            print(f"Warning: Output from SummaryWriter is of unexpected type: {type(writer_agent_output)}. Attempting to convert to string.")
            final_summary = str(writer_agent_output)

        if not final_summary or not final_summary.strip():
            print("ERROR: SummaryWriter produced no meaningful output.")
            sys.exit(1)
            
        print("SummaryWriter Output:")
        print(final_summary)

    except Exception as e:
        print(f"ERROR: An exception occurred while running SummaryWriter agent: {e}")
        sys.exit(1)

    # Create LLMTestCase
    # input: The initial research topic.
    # actual_output: The summary generated by SummaryWriter.
    # expected_output: The predefined research text that was supposed to be summarized.
    test_case = LLMTestCase(
        input=research_topic_input,         # Topic given to the system
        actual_output=final_summary,        # The final summary from SummaryWriter
        expected_output=research_text       # The original text fed into SummaryWriter
    )
    print("\nCreated LLMTestCase for DeepEval.")

    # Measure with the DAGMetric
    print("\nMeasuring with DAGMetric (MultiAgentSystemQuality)...")
    try:
        agent_quality_metric.measure(test_case)
    except Exception as e:
        print(f"ERROR: An exception occurred during DAGMetric.measure: {e}")
        print("This could be due to API issues, DeepEval configuration, or the metric definition.")
        sys.exit(1)

    # Print results
    print(f"\n--- DeepEval DAGMetric Results ---")
    print(f"Score: {agent_quality_metric.score}")
    if agent_quality_metric.reason: # Reason might not always be populated by all metrics/nodes
        print(f"Reason: {agent_quality_metric.reason}")
    
    # For more detailed output, print the metric object itself or iterate through its graph structure if needed.
    # The default __str__ representation of DAGMetric might provide a summary.
    print("\nDetailed Metric Information:")
    print(agent_quality_metric) 
    
    # Example of accessing node-specific results if available and needed (pseudo-code, API may vary)
    # for node_result in agent_quality_metric.dag_results: # Fictional attribute, check DeepEval docs
    #    print(f"Node {node_result.name}: Score={node_result.score}, Reason={node_result.reason}")

    print("\nEvaluation complete.")
