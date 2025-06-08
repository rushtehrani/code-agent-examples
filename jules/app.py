import argparse
import os
import sys
import uuid
from zep_cloud.client import ZepClient
from zep_cloud.types import Message

# Attempt to import Agent and Runner from the 'agents' library
try:
    from agents import Agent, Runner
except ImportError:
    print("ERROR: Failed to import Agent or Runner from 'agents'.")
    print("Please ensure the 'openai-agents-python' SDK is installed correctly.")
    print("For example, run: pip install openai-agents-python")
    print("If the library name or structure is different, this import statement will need to be updated.")
    sys.exit(1)

# Predefined text for the TopicResearcher
RESEARCH_FINDINGS = """
The field of quantum computing has seen significant advancements in recent years.
Researchers are exploring various qubit technologies, including superconducting circuits,
trapped ions, and photonic systems. Quantum algorithms, such_as Shor's algorithm for
factoring and Grover's algorithm for search, promise to solve certain problems
intractable for classical computers. However, building fault-tolerant quantum computers
remains a major challenge, requiring breakthroughs in error correction and qubit coherence.
The potential applications span drug discovery, materials science, financial modeling,
and cryptography.
"""

def main():
    parser = argparse.ArgumentParser(description="Run research and summarization agents.")
    parser.add_argument(
        "--topic", 
        type=str, 
        default="Quantum Computing", 
        help="The research topic for the TopicResearcher agent."
    )
    args = parser.parse_args()
    ZEP_USER_ID = "default_jules_user"

    # Print message about OPENAI_API_KEY as requested
    print("INFO: This script uses agents that may interact with OpenAI services.")
    print("Please ensure the OPENAI_API_KEY environment variable is set if required by the SDK for LLM operations.")
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set. LLM-based agent operations might fail.")

    print("INFO: This script uses Zep for memory.")
    print("Please ensure the ZEP_API_KEY environment variable is set.")
    zep_api_key = os.getenv("ZEP_API_KEY")
    if not zep_api_key:
        print("Warning: ZEP_API_KEY environment variable is not set. Zep operations will fail.")
        # Decide if exiting is necessary, for now, we'll just print a warning like for OPENAI_API_KEY
        # sys.exit(1) # Optionally exit if Zep is critical

    try:
        zep_client = ZepClient(api_key=zep_api_key)
        print("Successfully initialized ZepClient.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ZepClient: {e}")
        print("Please check your ZEP_API_KEY and network connection.")
        sys.exit(1)

    original_researcher_instructions = f"You are a research assistant. Your task is to find information on the given topic. The topic is: {{topic}}. Since you cannot browse the web, you will acknowledge the topic and provide the following pre-defined block of text as your research findings:\n\n{RESEARCH_FINDINGS}"
    topic_researcher = Agent(
        name="TopicResearcher",
        instructions=original_researcher_instructions
    )

    # Define the SummaryWriter agent
    # The placeholder {text_to_summarize} is intended to be filled by the input (output of TopicResearcher).
    original_summary_writer_instructions = "You are a writing assistant. Your task is to write a concise summary of the provided text: {text_to_summarize}"
    summary_writer = Agent(
        name="SummaryWriter",
        instructions=original_summary_writer_instructions
    )

    session_id = str(uuid.uuid4())
    print(f"INFO: Created Zep session ID: {session_id} for user ID: {ZEP_USER_ID}")

    # Add initial topic to Zep memory as a user message
    try:
        initial_topic_message = Message(role_type="user", role="User", content=args.topic)
        zep_client.memory.add(session_id, [initial_topic_message], user_id=ZEP_USER_ID) # Explicitly pass user_id
        print(f"INFO: Added initial topic to Zep memory for session {session_id}.")
    except Exception as e:
        print(f"ERROR: Failed to add initial topic to Zep memory: {e}")
        # Continue without Zep memory if this fails, or sys.exit(1) if critical

    # Retrieve memory context from Zep
    zep_memory_context = ""
    try:
        print(f"INFO: Retrieving Zep memory for session {session_id}...")
        memory_response = zep_client.memory.get(session_id, user_id=ZEP_USER_ID) # Explicitly pass user_id
        if memory_response and memory_response.context:
            zep_memory_context = memory_response.context
            print(f"INFO: Retrieved Zep memory context successfully.")
        else:
            print("INFO: No specific memory context returned by Zep.")
    except Exception as e:
        print(f"ERROR: Failed to retrieve Zep memory: {e}")
        # Continue without Zep memory if this fails

    # Augment TopicResearcher instructions
    augmented_researcher_instructions = original_researcher_instructions
    if zep_memory_context:
        augmented_researcher_instructions = f"Relevant conversation history (use this for context if helpful):\n{zep_memory_context}\n\n-----\n\n{original_researcher_instructions}"

    topic_researcher.instructions = augmented_researcher_instructions

    print(f"\nRunning TopicResearcher for topic: \"{args.topic}\"...")
    
    research_output_text = None
    try:
        # Run the TopicResearcher agent with the command-line topic
        # The SDK example `Runner.run_sync(agent, "Input string.")` suggests the input string
        # will be made available to the agent, likely for template substitution in instructions.
        research_result = Runner.run_sync(topic_researcher, args.topic)
        
        # Extract the text output, checking for a 'final_output' attribute as per prompt
        if hasattr(research_result, 'final_output') and isinstance(research_result.final_output, str):
            research_output_text = research_result.final_output
        elif isinstance(research_result, str):
            research_output_text = research_result
        else:
            print(f"Warning: Output from TopicResearcher is of unexpected type: {type(research_result)}. Attempting to convert to string.")
            research_output_text = str(research_result)
            # If the agent truly outputs the RESEARCH_FINDINGS as instructed, this should ideally be that string.
            # If it's not, the LLM might be adding conversational fluff. The prompt for the agent is specific.
            # A more robust check would be to see if RESEARCH_FINDINGS is a substring of research_output_text if it's not an exact match.
            # For now, we assume the agent's output *is* the findings.

        if research_output_text:
            try:
                researcher_output_message = Message(role_type="assistant", role=topic_researcher.name, content=research_output_text)
                zep_client.memory.add(session_id, [researcher_output_message], user_id=ZEP_USER_ID) # Explicitly pass user_id
                print(f"INFO: Added TopicResearcher output to Zep memory for session {session_id}.")
            except Exception as e:
                print(f"ERROR: Failed to add TopicResearcher output to Zep memory: {e}")

    except Exception as e:
        print(f"ERROR: An exception occurred while running TopicResearcher: {e}")
        print("This could be due to issues with the 'agents' SDK, OPENAI_API_KEY, network, or input handling.")
        sys.exit(1)
        
    print("\nOutput from TopicResearcher:")
    if research_output_text:
        print(research_output_text)
    else:
        print("ERROR: TopicResearcher did not produce any text output.")
        sys.exit(1) # Exit if no text to summarize

    print("\nRunning SummaryWriter...")

    # Retrieve memory context from Zep (it will now include TopicResearcher's output)
    zep_summary_memory_context = ""
    try:
        print(f"INFO: Retrieving Zep memory for SummaryWriter, session {session_id}...")
        summary_memory_response = zep_client.memory.get(session_id, user_id=ZEP_USER_ID)
        if summary_memory_response and summary_memory_response.context:
            zep_summary_memory_context = summary_memory_response.context
            print(f"INFO: Retrieved Zep memory context for SummaryWriter successfully.")
        else:
            print("INFO: No specific memory context returned by Zep for SummaryWriter.")
    except Exception as e:
        print(f"ERROR: Failed to retrieve Zep memory for SummaryWriter: {e}")

    # Augment SummaryWriter instructions
    augmented_summary_writer_instructions = original_summary_writer_instructions
    if zep_summary_memory_context:
        augmented_summary_writer_instructions = f"Relevant conversation history (use this for context if helpful):\n{zep_summary_memory_context}\n\n-----\n\n{original_summary_writer_instructions}"

    summary_writer.instructions = augmented_summary_writer_instructions

    summary_output_text = None
    try:
        # Run the SummaryWriter agent, using the output from TopicResearcher as input
        summary_result = Runner.run_sync(summary_writer, research_output_text)

        # Extract the text output, checking for a 'final_output' attribute
        if hasattr(summary_result, 'final_output') and isinstance(summary_result.final_output, str):
            summary_output_text = summary_result.final_output
        elif isinstance(summary_result, str):
            summary_output_text = summary_result
        else:
            print(f"Warning: Output from SummaryWriter is of unexpected type: {type(summary_result)}. Attempting to convert to string.")
            summary_output_text = str(summary_result)

        if summary_output_text:
            try:
                summary_writer_output_message = Message(role_type="assistant", role=summary_writer.name, content=summary_output_text)
                zep_client.memory.add(session_id, [summary_writer_output_message], user_id=ZEP_USER_ID)
                print(f"INFO: Added SummaryWriter output to Zep memory for session {session_id}.")
            except Exception as e:
                print(f"ERROR: Failed to add SummaryWriter output to Zep memory: {e}")
            
    except Exception as e:
        print(f"ERROR: An exception occurred while running SummaryWriter: {e}")
        sys.exit(1)

    print("\nFinal Summary from SummaryWriter:")
    if summary_output_text:
        print(summary_output_text)
    else:
        print("ERROR: SummaryWriter did not produce any text output.")
        sys.exit(1)

if __name__ == "__main__":
    main()
