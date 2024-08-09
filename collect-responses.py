from langchain_openai import ChatOpenAI
from typing import *
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import argparse

import os
from os import getenv
import csv
import re
import pandas as pd
import logging

# error handling
from langchain_core.exceptions import OutputParserException
from json.decoder import JSONDecodeError


# === logging config ===
log_max_str_len = 20
logging.basicConfig(level=logging.INFO, encoding='utf-8')

# === basic data collection config ===
API_URL = "https://openrouter.ai/api/v1"
models = {
    'mistral': 'mistralai/mistral-nemo',
    'gemma': 'google/gemma-7b-it',
    'gemini': 'google/gemini-pro-1.5-exp',
    'anthropic': 'anthropic/claude-3.5-sonnet:beta',
    'llama': 'meta-llama/llama-3.1-405b-instruct',
    'gpt4o': 'openai/gpt-4o',
    'gpt35': 'openai/gpt-3.5-turbo',
    'nvidia': 'nvidia/nemotron-4-340b-instruct',
}

# === set up prompt template and output parser ===
class CitiesResponse(BaseModel):
    cities: List[str] = Field(description="list of recommended town or city names")
    reasons: List[List[str]] = Field(description="for each city/town in `cities`, a list of reasons for recommending the town")
city_parser = PydanticOutputParser(pydantic_object=CitiesResponse)

additional_format_instructions = "Please do not provide any text in addition to the specified JSON response format. Please do not add formatting or indentation to the JSON response."
prompt_template = PromptTemplate(
    template="{query}\nCan you recommend 5 cities or towns with multiple reasons for each recommendation?\n\n{format_instructions}\n{additional_format_instructions}",
    input_variables=["query"],
    partial_variables={
        "format_instructions": city_parser.get_format_instructions(),        
        "additional_format_instructions": additional_format_instructions,
        # pattern to remove extra code blocks
        "pattern": re.compile(r"\`\`\`\n\`\`\`")
    },
)

# === config for data collection, mostly as arguments ===
TEMP = 0
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=models.keys(), help="model to use for data collection")
parser.add_argument("--prompt-type", type=str, required=True,
                    choices=['generic', 'no-constraint', 'single-constraint'],
                    help="folder to read prompts from")
parser.add_argument("--n_samples", type=int, required=False, default=3, help="number of samples to collect per prompt per model")
args = parser.parse_args()
mname = args.model
N_SAMPLES = args.n_samples
prompt_type = args.prompt_type
OUTFILE = f'{prompt_type}/{mname}.csv'
if prompt_type not in os.listdir("responses"):
    os.mkdir(f"responses/{prompt_type}")

# === load prompts ===
situations = ['relocation', 'tourism', 'opening_business']
# load prompts for each situation
situation_prompts = {}
for sit in situations:
    df = pd.read_csv(f"prompts/{prompt_type}/{sit}.csv")
    situation_prompts[sit] = df['text'].tolist()

# === save responses to csv as you go ===
writer = csv.writer(open(f"responses/{OUTFILE}", 'w'))
header = [
    "pid", "model", "situation", "prompt",
    "rec_city1", "rec_reasons1",
    "rec_city2", "rec_reasons2",
    "rec_city3", "rec_reasons3",
    "rec_city4", "rec_reasons4",
    "rec_city5", "rec_reasons5"
]
writer.writerow(header)

if not getenv("OPENROUTER_API_KEY"):
    raise Exception("Please set the OPENROUTER_API_KEY environment variable via `export OPENROUTER_API_KEY=<api-key>`")

# === collect responses ===
prompt_id = 1
model = ChatOpenAI(
    model=models[mname],
    temperature=TEMP,
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base = API_URL,
)
chain = prompt_template | model | city_parser
for sit in situation_prompts:
    for prompt in situation_prompts[sit]:
        orig_prompt = prompt
        samples = 0
        while samples < N_SAMPLES:                
            try:
                logging.info(f"\tQUERY --- prompt ID: {prompt_id}; prompt: {prompt[:log_max_str_len]}...; model: {mname}; situation: {sit}")
                response = chain.invoke({"query": prompt})
            except (OutputParserException, JSONDecodeError) as e:
                # error check 1 -- ill-formed output
                logging.error(f"Ill formed response: {e}; trying again")
                prompt += "\nYour output format was incorrect earlier. Please precisely adhere to the JSON format instructions."
                continue
            # error check 2 -- reasons not provided for all cities
            if len(response.cities) != len(response.reasons):
                logging.error(f"Response has unequal number of cities and reasons; trying again")
                prompt += "\nYou did not provide reasons for some of your recommendations. Please list reasons for recommending each city/town."
                continue

            samples += 1
            logging.info(f"""\tRESPONSE --- cities: {response.cities}; reasons: {response.reasons}""")
            # normalize dataframe i.e., only one reason per row
            row = [prompt_id, mname, sit, orig_prompt]
            for i, city in enumerate(response.cities):
                row.append(city)
                row.append(';'.join(response.reasons[i]))
            
            writer.writerow(row)                    
            prompt_id += 1
