from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm = init_chat_model("gpt-5-nano-2025-08-07", model_provider="openai")

summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes text concisely."),
    ("user", "Summarize the following topic: {topic}")
])
summarizer_chain = summarizer_prompt | llm | StrOutputParser()

questionnaire_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that creates questionnaires."),
    ("user", "Create a questionnaire based on the following information: {topic}")
])
questionnaire_chain = questionnaire_prompt | llm | StrOutputParser()

parallel_runnable = RunnableParallel({
    "summary": summarizer_chain,
    "questionnaire": questionnaire_chain,
    "topic": RunnablePassthrough()
})

synthesizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that synthesizes information.
     Based on the summary: {summary} and the questionnaire: {questionnaire}, provide a final synthesis."""),
    ("user", "original topic: {topic}")
])
synthesizer_chain = synthesizer_prompt | llm | StrOutputParser()
full_chain = parallel_runnable | synthesizer_chain

async def run_parallelization(topic: str):
    """Runs the parallelization chain."""
    try:
        response = await full_chain.ainvoke({"topic": topic})
        print("Final Response:", response)
    except Exception as e:
        print("Error during invocation:", e)


if __name__ == "__main__":
    import asyncio
    test_topic = "The impact of artificial intelligence on modern healthcare."
    asyncio.run(run_parallelization(test_topic))