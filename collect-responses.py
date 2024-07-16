from langchain_openai import ChatOpenAI
from typing import *
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from os import getenv
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# === basic data collection config ===
API_URL = "https://openrouter.ai/api/v1"
models = {
    'gemma': "google/gemma-7b-it:free",
    'mistral': 'mistralai/mistral-7b-instruct:free',
}

# === load prompts ===
situations = ['relocation', 'tourism', 'opening_business']
# load prompts for each situation
situation_prompts = {}
for sit in situations:
    df = pd.read_csv(f"prompts/{sit}.csv")
    situation_prompts[sit] = df['text'].tolist()

# === set up prompt template and output parser ===
class CitiesResponse(BaseModel):
    cities: List[str] = Field(description="list of recommended town or city names")
    reason: List[List[str]] = Field(description="for each city/town in `cities`, a list of reasons for recommending the town")
city_parser = PydanticOutputParser(pydantic_object=CitiesResponse)

prompt_template = PromptTemplate(
    template="{query}\nnCan you recommend 5 cities or towns with multiple reasons for each recommendation?\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": city_parser.get_format_instructions()},
)

TEMP = 0
N_SAMPLES = 3
all_responses = []
for mname in models:
    model = ChatOpenAI(
        model=mname,
        temperature=TEMP,
        openai_api_key=getenv("OPENROUTER_API_KEY"),
    )
    chain = prompt_template | model | city_parser
    for sit in situation_prompts:
        for prompt in situation_prompts[sit]:
            for _ in range(N_SAMPLES):
                response = chain.invoke({"query": prompt})
                all_responses.append([mname, sit, prompt, response])
                breakpoint()

df = pd.DataFrame(all_responses, columns=["model", "situation", "prompt", "response"])
df.to_csv("responses.csv", index=False)