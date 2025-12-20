#!/usr/bin/env python3
"""
LangGraph Multi-Agent Research Assistant
Entry point and CLI interface

This module provides the command-line interface for interacting
with the Research Assistant in both interactive and single-query modes.
"""

import argparse
import sys
import logging
from typing import Optional

from .app import ResearchAssistantApp
from .config import settings
from .utils.logging import setup_logging


def setup_cli() -> argparse.ArgumentParser:
    """Configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant powered by LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python -m src.research_assistant.main

  # Single query mode
  python -m src.research_assistant.main -q "Tell me about Apple's latest products"

  # Verbose logging
  python -m src.research_assistant.main -v

Environment Variables:
  ANTHROPIC_API_KEY    Required. Your Anthropic API key.
  USE_MOCK_DATA        Use mock data (default: true)
  LOG_LEVEL            Logging level (default: INFO)
"""
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query to process (non-interactive mode)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive conversation mode (default if no query provided)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Research Assistant v1.0.0"
    )

    return parser


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     LangGraph Multi-Agent Research Assistant                  ║
║     ─────────────────────────────────────────                ║
║     Powered by Claude & LangGraph                            ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help_commands():
    """Print available commands for interactive mode."""
    print("""
Commands:
  - Type your question to get company research
  - 'new'  : Start a fresh conversation
  - 'state': Show current conversation state
  - 'help' : Show this help message
  - 'quit' or 'exit': End the session

Tips:
  - Ask about companies like Apple, Tesla, Microsoft, Google, Amazon
  - Follow-up questions maintain context (e.g., "What about their competitors?")
  - The assistant will ask for clarification if your query is unclear
""")


def run_interactive_mode(app: ResearchAssistantApp):
    """Run the assistant in interactive conversation mode."""
    print_banner()
    print_help_commands()

    current_thread_id: Optional[str] = None

    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye! Thanks for using Research Assistant.\n")
                break

            if user_input.lower() == "new":
                current_thread_id = None
                print("\n🔄 Starting fresh conversation...\n")
                continue

            if user_input.lower() == "help":
                print_help_commands()
                continue

            if user_input.lower() == "state":
                if current_thread_id:
                    state = app.get_conversation_state(current_thread_id)
                    if state:
                        print(f"\n📊 Current State:")
                        print(f"   Thread: {current_thread_id}")
                        print(f"   Company: {state.get('detected_company', 'N/A')}")
                        print(f"   Clarity: {state.get('clarity_status', 'N/A')}")
                        print(f"   Attempts: {state.get('research_attempts', 0)}/3")
                    else:
                        print("   No state available")
                else:
                    print("\n📊 No active conversation")
                continue

            # Process query
            print("\n⏳ Processing your request...\n")

            if current_thread_id is None:
                result = app.start_conversation(user_input)
                current_thread_id = result["thread_id"]
            else:
                result = app.continue_conversation(current_thread_id, user_input)

            # Handle errors
            if result.get("error"):
                print(f"\n❌ Error: {result['error']}")
                continue

            # Handle interrupt (clarification needed)
            while result.get("interrupted"):
                interrupt_info = result.get("interrupt_info", {})
                question = interrupt_info.get("question", "Could you please clarify?")

                print(f"\n🤔 Clarification needed: {question}")
                clarification = input("\n📝 Your clarification: ").strip()

                if clarification.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!\n")
                    return

                if not clarification:
                    print("   Please provide a clarification to continue.")
                    continue

                print("\n⏳ Processing clarification...\n")
                result = app.resume_with_clarification(
                    current_thread_id,
                    clarification
                )

                if result.get("error"):
                    print(f"\n❌ Error: {result['error']}")
                    break

            # Display final response
            if result.get("final_response"):
                print("─" * 60)
                print(f"\n🤖 Assistant:\n")
                print(result["final_response"])
                print("\n" + "─" * 60)
            elif not result.get("error") and not result.get("interrupted"):
                print("\n⚠️ No response generated. Please try rephrasing your question.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"\n❌ An error occurred: {e}")
            print("   Please try again or start a new conversation.\n")


def run_single_query(app: ResearchAssistantApp, query: str):
    """Process a single query and display result."""
    print(f"\n📝 Query: {query}\n")
    print("⏳ Processing...\n")

    result = app.start_conversation(query)
    thread_id = result["thread_id"]

    # Handle errors
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        sys.exit(1)

    # Handle interrupts
    while result.get("interrupted"):
        interrupt_info = result.get("interrupt_info", {})
        question = interrupt_info.get("question", "Could you please clarify?")

        print(f"🤔 Clarification needed: {question}")
        clarification = input("\nYour response: ").strip()

        if not clarification:
            print("No clarification provided. Exiting.")
            sys.exit(1)

        result = app.resume_with_clarification(thread_id, clarification)

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            sys.exit(1)

    # Display result
    if result.get("final_response"):
        print("═" * 60)
        print("RESPONSE:")
        print("═" * 60)
        print(result["final_response"])
        print("═" * 60)
    else:
        print("⚠️ No response generated.")
        sys.exit(1)


def validate_environment():
    """Validate required environment configuration."""
    if not settings.validate_api_key():
        print("\n❌ Error: ANTHROPIC_API_KEY not configured!")
        print("\nPlease set your API key:")
        print("  1. Edit .env file and set ANTHROPIC_API_KEY=your_key_here")
        print("  2. Or export ANTHROPIC_API_KEY=your_key_here")
        print("\nGet your API key at: https://console.anthropic.com/")
        sys.exit(1)


def main():
    """Main entry point for the Research Assistant CLI."""
    parser = setup_cli()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Validate environment
    validate_environment()

    # Initialize application
    try:
        app = ResearchAssistantApp()
    except Exception as e:
        print(f"\n❌ Failed to initialize application: {e}")
        sys.exit(1)

    # Run appropriate mode
    if args.query:
        run_single_query(app, args.query)
    else:
        run_interactive_mode(app)


if __name__ == "__main__":
    main()
