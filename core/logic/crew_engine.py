from crewai import Agent, Task, Crew, LLM

def run_mcq_pipeline(text: str):
    # Gemini model setup with response as JSON
    llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.7,
    )

    # Agent 1: Generates Arabic MCQs
    mcq_generator = Agent(
        role="Arabic MCQ Generator",
        goal="Generate valid Arabic MCQs from provided text in JSON format",
        backstory="Expert in Arabic language and curriculum-based educational content.",
        llm=llm,
        verbose=True
    )

    # Agent 2: Arabic Question Validator
    question_validator = Agent(
        role="Arabic Question Validator",
        goal="Check the correctness and quality of Arabic multiple-choice questions.",
        backstory="Arabic linguist and curriculum reviewer responsible for ensuring the validity of exam questions.",
        llm=llm,
        verbose=True
    )

    # Agent 3: Difficulty Level Assigner
    difficulty_assessor = Agent(
        role="Arabic Question Difficulty Assessor",
        goal="Determine the difficulty level of each question as 'easy', 'normal', or 'hard'.",
        backstory="Experienced educator who assesses question difficulty based on language complexity and content.",
        llm=llm,
        verbose=True
    )

    # Task 1: Generate MCQs
    generate_task = Task(
        description=(
            "Generate Arabic MCQs from the input: {text}.\n"
            "The response must be a valid JSON object. The JSON must have a key 'questions', "
            "which is a list of objects, each containing 'question', 'answers', and 'correct_answer'."
        ),
        expected_output="A JSON object with 'questions' as a list of question objects, each containing 'question', 'answers', and 'correct_answer'.",
        agent=mcq_generator,
    )

    # Task 2: Validate the MCQs generated
    validate_task = Task(
        description=(
            "You will receive a JSON list of Arabic MCQs from the generate_task.\n\n"
            "Your task is to validate each question from the original input.\n"
            "- Check for grammar, logic, and relevance.\n"
            "- Respond with the **same JSON** structure, but for each question, add:\n"
            "  - `is_valid`: true or false\n"
            "  - `feedback`: short feedback in **Arabic** only if `is_valid` is false\n\n"
            "Return the modified JSON."
        ),
        context=[generate_task],
        expected_output="Same JSON structure with added 'is_valid' (boolean) and optional 'feedback' (Arabic string) per question.",
        agent=question_validator
    )

    # Task 3: Assign difficulty levels
    difficulty_task = Task(
        description=(
            "You will receive the validated Arabic MCQs from validate_task and the original text.\n\n"
            "Original Text:\n{text}\n\n"
            "Your task is to determine the difficulty level of each question based on how hard it is to answer from the original text.\n"
            "For each question, add a new field `difficulty` with one of these Arabic values: 'easy', 'noraml', or 'hard'.\n"
            "Keep the existing structure and return the updated JSON."
        ),
        context=[validate_task],
        expected_output="Same JSON structure with a new 'difficulty' field for each question.",
        agent=difficulty_assessor
    )


    # Define the crew with all three tasks
    crew = Crew(
        agents=[mcq_generator, question_validator, difficulty_assessor],
        tasks=[generate_task, validate_task, difficulty_task],
        verbose=True
    )

    # Run the process
    result = crew.kickoff({"text": text})
    print(result)
    return result
