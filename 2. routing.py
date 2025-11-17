"""This script is to implement the routing pattern in agentic AI applications."""
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv
load_dotenv()
llm = init_chat_model("gpt-5-nano-2025-08-07", model_provider="openai")

def booking_handler(user_input: str) -> str:
    """Handles booking related queries."""
    print("Routing to booking handler...")
    return "Booking handler processed the input: " + user_input

def info_handler(user_input: str) -> str:
    """Handles information related queries."""
    print("Routing to information handler...")
    return "Information handler processed the input: " + user_input

def unclear_handler(user_input: str) -> str:
    """Handles unclear queries."""
    print("Routing to unclear handler...")
    return "Unclear handler processed the input: " + user_input

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing agent that directs user queries to the appropriate handler based on their content.
                If the query is about booking, route to 'booking_handler'.\n"
                If the query is about information, route to 'info_handler'.\n"
                If the query is unclear, route to 'unclear_handler'.
                Output only the handler name as a single word response."""),
    ("user", "Route the following user query to the correct handler: {request}.\n")
])

branches = {
    "booking_handler": RunnablePassthrough.assign(output=lambda x: booking_handler(x["request"]["request"])),
    "info_handler": RunnablePassthrough.assign(output=lambda x: info_handler(x["request"]["request"])),
    "unclear_handler": RunnablePassthrough.assign(output=lambda x: unclear_handler(x["request"]["request"]))    
}
delegation_branch = RunnableBranch(
(lambda x: x["decision"]=="info_handler", branches["info_handler"]),
(lambda x: x["decision"]=="booking_handler", branches["booking_handler"]), 
branches["unclear_handler"],
)

# Runnable pass through is used to add additional variables to the input dict
# Runnable branch is used to route based on the decision made by the classifier
classifier_chain = classifier_prompt | llm | StrOutputParser() 
coordinator_chain = {"decision": classifier_chain} | {"request": RunnablePassthrough()} | delegation_branch | (lambda x: x["output"])

result = coordinator_chain.invoke({"request": "I would like to book a flight to New York next week."})
print("Final Result 1:", result)
print("--" * 20)
result = coordinator_chain.invoke({"request": "Can you provide information about the weather in Paris?"})
print("Final Result 2:", result)
print("--" * 20)
result = coordinator_chain.invoke({"request": "Blah blah random text"})
print("Final Result 3:", result)