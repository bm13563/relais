#!/usr/bin/env python3
"""Simple greeting pipeline definition."""

import sys
from pathlib import Path

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from relais import Pipeline, PipelineStep  # noqa: E402

from config import DB_PATH, INSTRUCTIONS_DIR  # noqa: E402
from tools import send_greeting  # noqa: E402


def main():
    print("=" * 60)
    print("Simple Greeting Pipeline")
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline.create(
        name="simple_greeting",
        steps={
            "greet": PipelineStep(
                name="greet",
                instruction="greet",
                response_tool="send_greeting",
                max_turns=3,
                tools=[send_greeting],
                next={"default": None}
            )
        },
        start_step="greet",
        instructions_dir=INSTRUCTIONS_DIR,
        db_config=DB_PATH
    )
    pipeline.initialize_db()

    # Run
    print("\nSending greeting request...")
    run_id = pipeline.run("Please greet Developer!")

    state = pipeline.get_run(run_id)
    print(f"\nCompleted! Status: {state.status}")
    print(f"Steps: {list(state.step_results.keys())}")

    for step_name, result in state.step_results.items():
        print(f"\n--- {step_name} output ---")
        if "final_response" in result:
            print(result["final_response"])


if __name__ == "__main__":
    main()
