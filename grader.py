grader = """You are a relevance grader for a retrieval system.

Your job: decide if the provided passages contain information that helps answer the question.

Respond with EXACTLY one word: "relevant" or "not_relevant". No explanation, no punctuation, no other text.

PASSAGES:
{context}

QUESTION: {question}

VERDICT:"""