# ============================================================
# [BRIDGE] 2024 NAACL
# Bridging the Novice-Expert Gap via Models of Decision-Making:
# A Case Study on Remediating Math Mistakes
# ============================================================

## No Decision-Making Prompt for gpt-4 and gpt-3.5-turbo
BRIDGE_TUTOR_RESPONSE_NO_DECISION_GPT = """
You are an experienced elementary math teacher and you are going to respond to a
student’s mistake in a useful and caring way. The problem your student is solving is
on topic: {lesson_topic}.
{c_h}
tutor (maximum one sentence):
"""


## No Decision-Making Prompt for llama-2
BRIDGE_TUTOR_RESPONSE_NO_DECISION_LLAMA2 = """
### System:
You are an experienced elementary math teacher and you are going to respond to a
student’s mistake in a useful and caring way.
### User:
Lesson topic: {lesson_topic}
Conversation:
{c_h}
### Assistant:
tutor (maximum one sentence):
"""


## Decision-Making Prompt for gpt-4 and gpt-3.5-turbo
BRIDGE_TUTOR_RESPONSE_WITH_DECISION_GPT = """
You are an experienced elementary math teacher and you are going to respond to a
student’s mistake in a useful and caring way. The problem your student is solving is
on topic: {lesson_topic}. {e} {z_what} in order to {z_why}.
{c_h}
tutor (maximum one sentence):
"""


## Decision-Making Prompt for llama-2
BRIDGE_TUTOR_RESPONSE_WITH_DECISION_LLAMA2 = """
### System:
You are an experienced elementary math teacher and you are going to respond to a
student’s mistake in a useful and caring way.
### User:
{e} {z_what} in order to {z_why}.
Lesson topic: {lesson_topic}
Conversation:
{c_h}
### Assistant:
tutor (maximum one sentence):
"""


## Determine Error (e) with gpt-4 and gpt-3.5-turbo.
## 튜터 응답 생성 전 Step A 프롬프트
BRIDGE_DETERMINE_ERROR_GPT = """
You are an experienced elementary math teacher. Your task is to read a conversation
snippet of a tutoring session between a student and tutor, and determine what type
of error the student makes in the conversation. We have a list of common errors that
students make in math, which you can pick from. We also give you the option to write in
your own error type if none of the options apply.
Error list:
0. Student does not seem to understand or guessed the answer.
1. Student misinterpreted the question.
2. Student made a careless mistake.
3. Student has the right idea, but is not quite there.
4. Student’s answer is not precise enough or the tutor is being too picky about the form
of the student’s answer.
5. None of the above, but I have a different description (please specify in your
reasoning).
6. Not sure, but I’m going to try to diagnose the student.
Here is the conversation snippet:
Lesson topic: {lesson_topic}
Conversation:
{c_h}
Why do you think the student made this mistake? Pick an option number from the error
list and provide the reason behind your choice. Format your answer as: [{"answer": #,
"reason": "write out your reason for picking # here"}]
"""


## Determine Strategy and Intention (zwhat, zwhy) with gpt-4 and gpt-3.5-turbo.
## 튜터 응답 생성 전 Step B + Step C 프롬프트
BRIDGE_DETERMINE_STRATEGY_INTENTION_GPT = """
You are an experienced elementary math teacher. Your task is to read a conversation
snippet of a tutoring session between a student and tutor where a student makes a mistake.
You should then determine what strategy you want to use to remediate the student’s error,
and state your intention in using that strategy. We have a list of common strategies and
intentions that teachers use, which you can pick from. We also give you the option to
write in your own strategy or intention if none of the options apply.

Strategies:
0. Explain a concept
1. Ask a question
2. Provide a hint
3. Provide a strategy
4. Provide a worked example
5. Provide a minor correction
6. Provide a similar problem
7. Simplify the question
8. Affirm the correct answer
9. Encourage the student
10. Other (please specify in your reasoning)

Intentions:
0. Motivate the student
1. Get the student to elaborate their answer
2. Correct the student’s mistake
3. Hint at the student’s mistake
4. Clarify a student’s misunderstanding
5. Help the student understand the lesson topic or solution strategy
6. Diagnose the student’s mistake
7. Support the student in their thinking or problem-solving
8. Explain the student’s mistake (eg. what is wrong in their answer or why is it incorrect)
9. Signal to the student that they have solved or not solved the problem
10. Other (please specify in your reasoning)

Here is the conversation snippet:
Lesson topic: {lesson_topic}
Conversation:
{c_h}

How would you remediate the student’s error and why? Pick the option number from the
list of strategies and intentions and provide the reason behind your choices. Format
your answer as: [{"strategy": #, "intention": #, "reason": "write out your reason for
picking that strategy and intention"}]
"""


# ============================================================
# [STEPWISE] 2024 EMNLP
# Stepwise Verification and Remediation of Student Reasoning Errors
# with Large Language Model Tutors
# ============================================================

## Response generation prompt for the direct baseline
STEPWISE_TUTOR_RESPONSE_DIRECT_BASELINE = """
You are an experienced teacher and you are going to respond to a student. The problem your student is solving is on
topic: {topic}.
Problem: {problem}
{conversation}
Teacher (maximum two sentences):
"""


## Verification for Error reason baseline
## 튜터 응답 생성 전 Error Reason 생성 프롬프트
STEPWISE_ERROR_REASON_BASELINE = """
You are an experienced teacher. Your task is to read a conversation snippet of a tutoring session between a student and
tutor, and determine what type of error the student makes in the conversation. We have a list of common errors that
students make in math, which you can pick from. We also give you the option to write in your own error type if none of
the options apply.
Error list:
0. Student does not seem to understand or guessed the answer.
1. Student misinterpreted the question.
2. Student made a careless mistake.
3. Student has the right idea, but is not quite there.
4. Student’s answer is not precise enough or the tutor is being too picky about the form of the student’s answer.
5. None of the above, but I have a different description (please specify in your reasoning).
6. Not sure, but I’m going to try to diagnose the student.
Here is the conversation snippet:
Lesson topic: {topic}.
Problem: {problem}
{conversation}
Why do you think the student made this mistake? Pick an option number from the error list and provide the reason
behind your choice. Format your answer as: {"answer": , "reason": "write out your reason for picking here"}
"""


## Verification prompt for Error description of the first student error
## 튜터 응답 생성 전 Error Description 생성 프롬프트
STEPWISE_ERROR_DESCRIPTION = """
You are an experienced math teacher. Your goal is to identify the correctness of the Student’s Solution to a Problem.
Problem: {problem}
Expected reference solution: {solution}
{conversation}
Q: Find the first error in the student solution compared to the expected reference solution and write a one line description.
If no error, write "Student’ solution is Correct".
A:
"""


## Response generation for Error reason baseline, Error description, and Alignment generation
STEPWISE_TUTOR_RESPONSE_WITH_VERIFICATION = """
You are an experienced teacher and you are going to respond to a student. The problem your student is solving is on
topic: {topic}.
Problem: {problem}
Assessment of student solution: {description}
{conversation}
Teacher (maximum two sentences):
"""


## Prompt for the chain-of-thought reference solution generation
## Error Description에서 {solution}을 만들 때 사용
STEPWISE_REFERENCE_SOLUTION_COT = """
Question: Maila read 12 pages of a book yesterday and twice as many pages today. If the book has 120 pages,
If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Maila read 12 x 2 = «12*2=24»24 pages today. So she was able to read a total of 12 + 24 = «12+24=36»36
pages since yesterday. There are 120 - 36 = «120-36=84»84 pages left to be read. Since she wants to read half of the
remaining pages tomorrow, then she should read 84/2 = «84/2=42»42 pages. The answer is 42
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she
earn?
Answer: Weng earns 12/60 = «12/60=0.2»0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = «0.2*50=10»10.
The answer is 10
Question: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5
respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs
$200?
Answer: According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts Since Johnson got $2500, each
part is therefore $2500/5 = $«2500/5=500»500 Mike will get 2*$500 = $«2*500=1000»1000. After buying the shirt he
will have $1000-$200 = $«1000-200=800»800 left. The answer is 800
Question: Ralph is going to practice playing tennis with a tennis ball machine that shoots out tennis balls for Ralph to
hit. He loads up the machine with 175 tennis balls to start with. Out of the first 100 balls, he manages to hit 2/5 of them.
Of the next 75 tennis balls, he manages to hit 1/3 of them. Out of all the tennis balls, how many did Ralph not hit?
Answer: Out of the first 100 balls, Ralph was able to hit 2/5 of them and not able to hit 3/5 of them, 3/5 x 100 = 60
tennis balls Ralph didn’t hit. Out of the next 75 balls, Ralph was able to hit 1/3 of them and not able to hit 2/3 of them,
2/3 x 75 = 50 tennis balls that Ralph didn’t hit. Combined, Ralph was not able to hit 60 + 50 = «60+50=110»110 tennis
balls Ralph didn’t hit. The answer is 110
Question: {problem}
Answer:
"""


# ============================================================
# [MATHDIAL] 2023 Findings EMNLP
# MATHDIAL: A Dialogue Tutoring Dataset with Rich Pedagogical
# Properties Grounded in Math Reasoning Problems
# ============================================================

## ChatGPT teacher model prompt in interactive tutoring scenario
## 논문 Appendix E에 제시된 ChatGPT teacher prompt
MATHDIAL_TUTOR_RESPONSE_CHATGPT_INTERACTIVE = """
A tutor and a student work together to
solve the following math word problem.\n
Math problem: (MATH PROBLEM)\n
The correct solution is as follows:
(CORRECT SOLUTION)\n
Your role is tutor.               The tutor is
a soft-spoken empathetic person who
dislikes giving out direct answers to
students and instead likes to answer
with other questions that would help
the student understand the concepts
so students can solve the problem
themselves.
"""


## Human evaluation setting note from the paper:
## "ChatGPT prompt is the same as in the interactive tutoring scenario (Section E)
## with an additional section containing student solution."
##
## 논문에는 이 추가 section이 완성된 prompt template 형태로 별도 출력되어 있지는 않음.
## 따라서 아래는 논문에 인쇄된 문장만 반영한 이름 지정용 상수이며,
## exact full prompt로 공개된 것은 위 MATHDIAL_TUTOR_RESPONSE_CHATGPT_INTERACTIVE임.
MATHDIAL_TUTOR_RESPONSE_CHATGPT_HUMAN_EVAL_NOTE = """
ChatGPT prompt is the same as in the interactive
tutoring scenario (Section E) with an additional
section containing student solution.
"""


## Teacher first utterance hardcoded in interactive evaluation
MATHDIAL_FIRST_TEACHER_UTTERANCE = """
Hi Kayla, could you walk me through your solution?
"""


## NextStep baseline in interactive evaluation
MATHDIAL_NEXTSTEP_BASELINE = """
What is the next step?
"""


# ============================================================
# [LLM CANNOT] 2025 EMNLP
# LLMs cannot spot math errors, even when allowed to peek into the solution
# ============================================================

## IMPORTANT:
## 이 논문은 tutor response generation 논문이 아니라 first error step localization 논문임.
## 따라서 논문 안에 "Teacher response:" 또는 "Tutor response:"를 생성하는 프롬프트는 없음.
## 아래 프롬프트들은 튜터 응답 생성 전에 사용할 수 있는 진단용 프롬프트임.

## Problem-solving prompt: initial prompt
LLMCANNOT_PROBLEM_SOLVING_INITIAL = """
You are experienced at solving math word problems. Solve
the following problem to the best of your ability in a
stepwise manner.
Clearly specify your final answer at the end of your solution
in a new line.
Problem: <<PROBLEM>>
"""


## Problem-solving prompt: follow-up prompt
LLMCANNOT_PROBLEM_SOLVING_FOLLOWUP = """
You are experienced at solving math word problems. Solve
the following problem to the best of your ability in a
stepwise manner.
Clearly specify your final answer at the end of your solution
in a new line.
Problem: <<PROBLEM>>
<<RESPONSE_TO_INITIAL_PROMPT>>
Therefore, the final answer is:
"""


## Without Solution (w/o-S) Prompt
LLMCANNOT_FIRST_ERROR_WITHOUT_SOLUTION = """
You are an experienced math teacher. Your goal is to identify the
step of the first mistake in the Student's Solution to a Problem.
Problem: Natalia sold clips to 48 of her friends in April, and then
she sold half as many clips in May. How many clips did Natalia
sell altogether in April and May?
Student Solution:
Step 1 - Natalia sold 48 clips in April.
Step 2 - She sold 48*2 = 96 clips in May.
Step 3 - She sold 48+96 = 144 clips in April and May together.
Q: Write only the step number with the first error.
A: 2
Problem: Natalia sold clips to 48 of her friends in April, and then
she sold half as many clips in May. How many clips did Natalia
sell altogether in April and May?
Student Solution:
Step 1 - She sold 48/2 = 16 clips in May.
Step 2 - Natalia sold 48+16 = 64 clips in April and May together.
Q: Write only the step number with the first error.
A: 1
Problem: <<PROBLEM>>
Student Solution:
<<STUDENT_STEPS>>
Q: Write only the step number with the first error.
A:
"""


## With Gold Solution (w-GS) Prompt
LLMCANNOT_FIRST_ERROR_WITH_GOLD_SOLUTION = """
You are an experienced math teacher. Your goal is to identify the
step of the first mistake in the Student's Solution to a Problem.
Problem: Natalia sold clips to 48 of her friends in April, and then
she sold half as many clips in May. How many clips did Natalia
sell altogether in April and May?
Expected Answer:
Step 1 - Natalia sold 48/2 = 24 clips in May.
Step 2 - Natalia sold 48+24 = 72 clips altogether in April and May.
Student Solution:
Step 1 - Natalia sold 48 clips in April.
Step 2 - She sold 48*2 = 96 clips in May.
Step 3 - She sold 48+96 = 144 clips in April and May together.
Q: Write only the step number with the first error.
A: 2
Problem: Natalia sold clips to 48 of her friends in April, and then
she sold half as many clips in May. How many clips did Natalia
sell altogether in April and May?
Expected Answer:
Step 1 - Natalia sold 48/2 = 24 clips in May.
Step 2 - Natalia sold 48+24 = 72 clips altogether in April and May.
Student Solution:
Step 1 - She sold 48/2 = 16 clips in May.
Step 2 - Natalia sold 48+16 = 64 clips in April and May together.
Q: Write only the step number with the first error.
A: 1
Problem: <<PROBLEM>>
Expected Answer:
<<GOLD_STEPS>>
Student Solution:
<<STUDENT_STEPS>>
Q: Write only the step number with the first error.
A:
"""


## Corrected Student Solution generation prompt
LLMCANNOT_GENERATE_CORRECTED_STUDENT_SOLUTION = """
Here's a problem: <<PROBLEM>>

Here's the problem's correct reference solution:
<<GOLD_SOLUTION>>

Here's a stepwise candidate solution to the same problem:
<<STUDENT_STEPS>>

Based on the problem and the reference solution, correct and
rewrite the candidate solution.
Change only the portions that are incorrect and need edits.
"""


## With Corrected Student Solution (w-Cor) Prompt
LLMCANNOT_FIRST_ERROR_WITH_CORRECTED_STUDENT_SOLUTION = """
You are an experienced math teacher. Your goal is to identify the
step of the first mistake in the Student's Solution to a Problem.
Problem: Natalia sold clips to 48 of her friends in April, and then
she sold half as many clips in May. How many clips did Natalia
sell altogether in April and May?
Expected Answer:
Step 1 - Natalia sold 48/2 = 24 clips in May.
Step 2 - Natalia sold 48+24 = 72 clips altogether in April and May.
Student Solution:
Step 1 - Natalia sold 48 clips in April.
Step 2 - She sold 48*2 = 96 clips in May.
Step 3 - She sold 48+96 = 144 clips in April and May together.
Q: Write only the step number with the first error.
A: 2
Problem: Natalia sold clips to 48 of her friends in April, and then
she sold half as many clips in May. How many clips did Natalia
sell altogether in April and May?
Expected Answer:
Step 1 - Natalia sold 48/2 = 24 clips in May.
Step 2 - Natalia sold 48+24 = 72 clips altogether in April and May.
Student Solution:
Step 1 - She sold 48/2 = 16 clips in May.
Step 2 - Natalia sold 48+16 = 64 clips in April and May together.
Q: Write only the step number with the first error.
A: 1
Problem: <<PROBLEM>>
Expected Answer:
<<CORRECTED_STUDENT_STEPS>>
Student Solution:
<<STUDENT_STEPS>>
Q: Write only the step number with the first error.
A:
"""