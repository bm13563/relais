#!/usr/bin/env python3
"""Research pipeline with subagent for isolated research."""

import sys
from pathlib import Path

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from relais import Pipeline, PipelineStep  # noqa: E402

from config import DB_PATH, INSTRUCTIONS_DIR  # noqa: E402
from tools.search import search  # noqa: E402
from tools.summary import create_summary  # noqa: E402


def main():
    print("=" * 60)
    print("Research Pipeline (with Subagent)")
    print("=" * 60)
    print("\nThis pipeline uses an isolated subagent for research,")
    print("then summarizes the findings in the main agent context.\n")

    # Create pipeline
    pipeline = Pipeline.create(
        name="research_pipeline",
        steps={
            "research": PipelineStep(
                name="research",
                instruction="research",
                response_tool="search",
                max_turns=5,
                tools=[search],
                subagent=True,
                next={"default": "summarize"}
            ),
            "summarize": PipelineStep(
                name="summarize",
                instruction="summarize",
                response_tool="create_summary",
                max_turns=3,
                tools=[create_summary],
                next={"default": None}
            )
        },
        start_step="research",
        instructions_dir=INSTRUCTIONS_DIR,
        db_config=DB_PATH,
        grounded=True
    )
    pipeline.initialize_db()

    # Run
    run_id = pipeline.run("Research the history of Python programming language")

    state = pipeline.get_run(run_id)
    print(f"\nCompleted! Status: {state.status}")
    print(f"Steps: {list(state.step_results.keys())}")

    for step_name, result in state.step_results.items():
        print(f"\n--- {step_name} output ---")
        if "final_response" in result:
            print(result["final_response"])

    print("\nNote: The research step ran in an isolated subagent")
    print("with its own conversation context, separate from the main agent.")


if __name__ == "__main__":
    main()
