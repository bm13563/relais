"""Answer tool for question-answering in routing pipelines."""

from relais import tool, Annotated


@tool("answer", "Provide an answer to a question")
async def answer(
    question: Annotated[str, "The question being answered"],
    answer: Annotated[str, "The answer to provide"],
) -> dict:
    """Answer a question."""
    return {
        "content": [{
            "type": "text",
            "text": f"Q: {question}\nA: {answer}"
        }]
    }
