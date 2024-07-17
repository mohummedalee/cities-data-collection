from langchain_openai import ChatOpenAI
from typing import *
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.exceptions import OutputParserException

from os import getenv
import re
import pandas as pd
import logging

logfile = 'collection.log'
log_max_str_len = 15
logging.basicConfig(level=logging.INFO, encoding='utf-8')


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
    reasons: List[List[str]] = Field(description="for each city/town in `cities`, a list of reasons for recommending the town")
city_parser = PydanticOutputParser(pydantic_object=CitiesResponse)

prompt_template = PromptTemplate(
    template="{query}\nCan you recommend 5 cities or towns with multiple reasons for each recommendation?\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": city_parser.get_format_instructions(), "pattern": re.compile(r"\`\`\`\n\`\`\`")},
)

TEMP = 0
N_SAMPLES = 3
all_responses = []

prompt_id = 1
for mname in models:
    model = ChatOpenAI(
        model=models[mname],
        temperature=TEMP,
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base = API_URL,
    )
    chain = prompt_template | model | city_parser
    for sit in situation_prompts:
        for prompt in situation_prompts[sit]:
            samples = 0
            while samples < N_SAMPLES:                
                try:
                    response = chain.invoke({"query": prompt})
                except OutputParserException as e:
                    logging.warning(f"Ill formed response: {e}; trying again")
                    continue

                samples += 1
                logging.info(f"""prompt ID: {prompt_id}; prompt: {prompt[:log_max_str_len]}...; model: {mname}; situation: {sit}; cities: {response.cities}""")
                # normalize dataframe i.e., only one reason per row
                for i, city in enumerate(response.cities):
                    for j, reason in enumerate(response.reasons[i]):
                        all_responses.append([prompt_id, mname, sit, prompt, city, reason])
                prompt_id += 1

df = pd.DataFrame(all_responses, columns=["pid", "model", "situation", "prompt", "rec_cities", "rec_reasons"])
df.to_csv("responses/gemma-mistral.csv", index=False)