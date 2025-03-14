BBH_ANSWER_FORMATS = """
- If the answer is not multiple choice, [answer] should be the decided answer. (For eg: Q: not True or False. A: False)
- If the answer is multiple choice,
    - and the given choices are unlabelled options, [answer] should be the chosen option (For eg: Q: Where does the sun rise from? Options: - East, - West, - North. A: East)
    - and the given choices are labelled options, [answer] should be the letter corresponding to the chosen option (For eg: Q: Where does the sun rise from? Options: - A. West, - B. East, - C. North. A: B)"""

T4D_ANSWER_FORMATS = """
- should be complete with the letter and correct answer from the list of given choices (Example answer:  K. Ananda))"""

MATH_ANSWER_FORMATS = """
- should be the final answer based on calculations formatted in Latex style"""