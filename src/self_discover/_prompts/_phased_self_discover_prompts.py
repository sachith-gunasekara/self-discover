## SELECT PROMPT ##

SELECT_PROMPT = """Select several reasoning modules that are crucial to utilize in order to solve the given task.

All reasoning module descriptions:
1 How could I devise an experiment to help solve that problem?
2 Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.
3 How could I measure progress on this problem?
4 How can I simplify the problem so that it is easier to solve?
5 What are the key assumptions underlying this problem?
6 What are the potential risks and drawbacks of each solution?
7 What are the alternative perspectives or viewpoints on this problem?
8 What are the long-term implications of this problem and its solutions?
9 How can I break down this problem into smaller, more manageable parts?
10 Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.
11 Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.
12 Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.
13 Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.
14 Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.
15 Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.
16 What is the core issue or problem that needs to be addressed?
17 What are the underlying causes or factors contributing to the problem?
18 Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?
19 What are the potential obstacles or challenges that might arise in solving this problem?
20 Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?
21 Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?
22 What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?
23 How can progress or success in solving the problem be measured or evaluated?
24 What indicators or metrics can be used?
25 Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?
26 Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?
27 Is the problem related to human behavior, such as a social, cultural, or psychological issue?
28 Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?
29 Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?
30 Is the problem a design challenge that requires creative solutions and innovation?
31 Does the problem require addressing systemic or structural issues rather than just individual instances?
32 Is the problem time-sensitive or urgent, requiring immediate attention and action?
33 What kinds of solution typically are produced for this kind of problem specification?
34 Given the problem specification and the current best solution, have a guess about other possible solutions.
35 Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?
36 What is the best way to modify this current best solution, given what you know about these kinds of problem specification?
37 Ignoring the current best solution, create an entirely new solution to the problem.
38 Let's think step by step.
39 Let's make a step by step plan and implement it with good notion and explanation.

Task: 
{task_description}

Select several modules are crucial for solving the task above. 
Your response should only be the list of select modules(module number and description), no explanations or solutions are required."""


## ADAPT PROMPT ##

ADAPT_PROMPT = """Rephrase and specify each reasoning module so that it better helps solving the task.

SELECTED module descriptions:
{selected_modules}

Task:
{task_description}

Adapt each reasoning module description to better solve the task.
Your response should only be the list of adapted modules, no explanations or solutions are required."""


### REASON ###

## PLANNING PROMPT ##

# SRUCTURED PLANNING #

STRUCTURED_PLANNING_PROMPT = """Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format.

Here's an example:

Example task:

If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:

{{
    "Position after instruction 1":
    "Position after instruction 2":
    "Position after instruction n":
    "Is final position the same as starting position":
}}

Adapted module description:
{adapted_modules}

Task:
{task_description}

Implement a reasoning structure for solvers to follow step-by-step and arrive at correct answer.

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can fill it out and arrive at the correct conclusion for tasks like this"""


# UNSTRUCTURED PLANNING #

UNSTRUCTURED_PLANNING_PROMPT = """Operationalize the reasoning modules into a step-by-step reasoning plan in plain English to solve the given task.
Make sure the plan is concrete, intuitive, and unambigous.
The reasoning plan should help an AI agent follow it and be able to derive a solution to the given task.

Here's an example:
Example task:
If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:
Find position after instruction 1.
FInd position after instruction 2.
Find position after instruction n.
Is final position the same as starting position?

Reasoning Module description:
{adapted_modules}

Task:
{task_description}

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN that can be followed to arrive at the correct answer the given task."""


## APPLICATION PROMPT ##

# STRUCTURED APPLICATION #

STRUCTURED_APPLICATION_PROMPT = """Follow the step-by-step reasoning plan in JSON to correctly solve the task filling in the values for the corresponding keys.
Do not simply rephrase the keys.
Phrase your final answer always as "The final answer is [answer]".

[answer] should be in one of the following formats:
{answer_formats}
    
Reasoning Structure:
{reasoning_plan}

Correctly follow the above JSON reasoning structure to solve the given task below. Your response should be the filled JSON for the above reasoning structure.

Task:
{task_description}"""


# UNSTRUCTURED APPLICATION

UNSTRUCTURED_APPLICATION_PROMPT = """Follow the reasoning plan step-by-step to arrive at the correct answer
Your response should only contain the reasoning process for the given task.
Phrase your final answer always as "The final answer is [answer]".

[answer] should be in one of the following formats:
{answer_formats}

Reasoning Plan:
{reasoning_plan}

Task:
{task_description}"""


### Few Shot Examples Prompt

FEW_SHOT_EXAMPLES_PROMPT = """Here are some other examples like the above task:
<examples>
{few_shot_examples_str}
</examples>"""
