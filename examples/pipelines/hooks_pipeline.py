#!/usr/bin/env python3
"""Hooks pipeline demonstrating context injection."""

import sys
from pathlib import Path

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from relais import Pipeline, PipelineStep  # noqa: E402

from config import DB_PATH, INSTRUCTIONS_DIR  # noqa: E402
from hooks import get_current_time, get_user_preferences, get_system_status  # noqa: E402
from tools import contextual_greeting, report_status  # noqa: E402


def main():
    print("=" * 60)
    print("Hooks Pipeline")
    print("=" * 60)
    print("\nThis pipeline uses hooks to inject dynamic context")
    print("(time, preferences, system status) into each step.\n")

    # Create pipeline
    pipeline = Pipeline.create(
        name="hooks_example",
        steps={
            "process_with_context": PipelineStep(
                name="process_with_context",
                instruction="greet",
                max_turns=3,
                tools=[contextual_greeting],
                hooks=[get_current_time, get_user_preferences],
                next={"default": "status_report"}
            ),
            "status_report": PipelineStep(
                name="status_report",
                instruction="chat",
                max_turns=2,
                tools=[report_status],
                hooks=[get_system_status, get_current_time],
                next={"default": None}
            )
        },
        start_step="process_with_context",
        instructions_dir=INSTRUCTIONS_DIR,
        db_config=DB_PATH
    )
    pipeline.initialize_db()

    # Run
    run_id = pipeline.run(
        "Please greet me and then give me a status report",
        args={"user_id": "demo_user"}
    )

    state = pipeline.get_run(run_id)
    print(f"\nCompleted! Status: {state.status}")
    print(f"Steps: {list(state.step_results.keys())}")

    for step_name, result in state.step_results.items():
        print(f"\n--- {step_name} output ---")
        if "final_response" in result:
            print(result["final_response"])

    print("\nHook data was injected into the context for each step,")
    print("allowing the agent to use real-time information.")


if __name__ == "__main__":
    main()
