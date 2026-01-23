# Analysis Step

Analyze the user's request and classify it using these EXACT rules:

1. If the input contains words like "calculate", "compute", "add", "subtract", "multiply", "divide", or similar math operations → classify as "task"
2. If the input contains a question mark (?) → classify as "question"
3. Otherwise → classify as "chat"

You MUST call the `classify_request` tool exactly once with:
- category: One of "question", "task", or "chat" based on the rules above
- confidence: A value between 0 and 1 (use 0.95 for clear matches)
