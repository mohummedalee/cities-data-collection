from langchain_openai import ChatOpenAI
from typing import *
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from os import getenv
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

"""
Testing script:
- poll openrouter with a prompt
- pass output to a structured output parser
- compare (a) the output of the structured output parser with (b) the unparsed output
"""

API_URL = "https://openrouter.ai/api/v1"
MODEL = "gpt-3.5-turbo-0125"
# MODEL = "google/gemma-7b-it:free"
# structured_llm = model.with_structured_output(Joke)

class CitiesResponse(BaseModel):
    cities: List[str] = Field(description="list of recommended town or city names")
    reason: List[List[str]] = Field(description="for each city/town in `cities`, a list of reasons for recommending the town")


city_parser = PydanticOutputParser(pydantic_object=CitiesResponse)
print(city_parser.get_format_instructions())

model = ChatOpenAI(
    model=MODEL,
    temperature=0,
    openai_api_key=getenv("OPENAI_API_KEY"),
)
prompt_raw = PromptTemplate(
    template="{query}\nCan you recommend 5 cities or towns with reasons for your recommendation?",
    input_variables=["query"]
)

prompt_structured = PromptTemplate(
    template="{query}\nCan you recommend 5 cities or towns with reasons for your recommendation?\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": city_parser.get_format_instructions()},
)

chain = prompt_raw | model
chain_parsed = prompt_structured | model | city_parser

prompt = """
    I am making the move to New Jersey for a job promotion I recently received.
    I've pretty much lived in the Midwest most of my life and have limited knowledge and connections in NJ.
    I'm in my mid 20s and looking for somewhere safe, walkable, and with those that are of similar age so I can hopefully go out and meet people and have an access to a lot of social opportunities.
    Affordability would be a mega plus but the rental market seems to disagree with me a little bit on that one.
    I'm also working remotely so commuting is not a concern - although I will have a car to travel for client meetings.
    Would really appreciate any recommendations of areas to check out as I will be making a trip out to the area in a few weeks!
    Thanks so much!
"""
raw_response = chain.invoke({"query": prompt})
print(raw_response)

parsed_response = chain_parsed.invoke({"query": prompt})
print(parsed_response)