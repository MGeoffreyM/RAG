import dspy


class AnswerBot(dspy.Signature):
    """
    You are a coding expert. Answer the provided questions based on the
    corresponding sources. If you do not know the answer, do not make one up
    and instead, reply "I don't know!

    Inputs:
        question: str,
        sources: list[MinimalSource]

    Output:
        answer: str
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Formatted sources containing context")
    answer: str = dspy.OutputField()
