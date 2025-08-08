import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational",
    temperature=0.3,
    max_new_tokens=200,
)

llm = ChatHuggingFace(llm=base_llm)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Keep in mind that the system name is Hugging Moon"),
        MessagesPlaceholder(variable_name="question")
    ]
)



