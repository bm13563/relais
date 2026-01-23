#!/usr/bin/env python3
"""Routing pipeline with conditional branching based on classification."""

import sys
from pathlib import Path

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from relais import Pipeline, PipelineStep  # noqa: E402

from config import DB_PATH, INSTRUCTIONS_DIR  # noqa: E402
from tools import classify_request, answer, execute_task, chat_response  # noqa: E402


def main():
    print("=" * 60)
    print("Routing Pipeline")
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline.create(
        name="routing_example",
        steps={
            "analyze": PipelineStep(
                name="analyze",
                instruction="analyze",
                max_turns=2,
                tools=[classify_request],
                next={
                    "field": "category",
                    "routes": [
                        {"equals": "question", "goto": "answer_question"},
                        {"equals": "task", "goto": "perform_task"},
                        {"equals": "chat", "goto": "chat"},
                    ],
                    "default": "chat"
                }
            ),
            "answer_question": PipelineStep(
                name="answer_question",
                instruction="answer_question",
                max_turns=3,
                tools=[answer],
                next={"default": None}
            ),
            "perform_task": PipelineStep(
                name="perform_task",
                instruction="perform_task",
                max_turns=5,
                tools=[execute_task],
                next={"default": None}
            ),
            "chat": PipelineStep(
                name="chat",
                instruction="chat",
                max_turns=2,
                tools=[chat_response],
                next={"default": None}
            )
        },
        start_step="analyze",
        instructions_dir=INSTRUCTIONS_DIR,
        db_config=DB_PATH
    )
    pipeline.initialize_db()

    # Test cases
    test_inputs = [
        ("What is the capital of France?", "question"),
        ("Please calculate 25 * 4 for me", "task"),
        ("Hey, how's it going?", "chat"),
    ]

    for i, (input_text, expected_type) in enumerate(test_inputs, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}: {input_text}")
        print(f"Expected route: {expected_type}")
        print("=" * 60)

        run_id = pipeline.run(input_text, args={"test_number": i})
        state = pipeline.get_run(run_id)

        print(f"\nCompleted via route: {list(state.step_results.keys())}")

        for step_name, result in state.step_results.items():
            print(f"\n--- {step_name} output ---")
            if "final_response" in result:
                print(result["final_response"])


if __name__ == "__main__":
    main()
