import pandas as pd
from pydantic import BaseModel, validator, ValidationError

# Define the Pydantic model
class DataFrameSchema(BaseModel):
    model: str
    situation: str
    prompt: str
    rec_city1: str
    rec_reasons1: str
    rec_city2: str
    rec_reasons2: str
    rec_city3: str
    rec_reasons3: str
    rec_city4: str
    rec_reasons4: str
    rec_city5: str
    rec_reasons5: str
    
    @validator('*')
    def check_is_string(cls, v):
        if not isinstance(v, str):
            raise ValueError('Must be a string')
        return v

# Function to validate the DataFrame
def validate_dataframe(df: pd.DataFrame):
    # Check if the dataframe has the right columns
    required_columns = {"model", "situation", "prompt", "rec_city1", "rec_reasons1", \
                        "rec_city2", "rec_reasons2", "rec_city3", "rec_reasons3", "rec_city4", "rec_reasons4"\
                            "rec_city5", "rec_reasons5"}
    if set(df.columns) != required_columns:
        raise ValueError(f"DataFrame must contain the columns: {required_columns}")

    # Validate each row
    for idx, row in df.iterrows():
        try:
            DataFrameSchema(
                model=row['model'],
                situation=row['situation'],
                prompt=row['prompt'],
                rec_city1=row['rec_city1'],
                city_1_rec=row['rec_reasons1'],
                city_1_name=row['rec_city2'],
                city_1_rec=row['rec_reasons2'],
                city_1_name=row['rec_city3'],
                city_1_rec=row['rec_reasons3'],
                city_1_name=row['rec_city4'],
                city_1_rec=row['rec_reasons4'],
                city_1_name=row['rec_city5'],
                city_1_rec=row['rec_reasons5']
            )
        except ValidationError as e:
            print(f"Row {idx} is invalid: {e}")

# Example DataFrame
data = {
    "model": ["sys1", "sys2", "sys3"],
    "situation": ["sit1", "sit2", "sit3"],
    "prompt": ["query1", "query2", "query3"],
    "rec_city1": ["city1", "city2", "city3"],
    "rec_reasons1": ["rec1", "rec2", "rec3"],
    "rec_city2": ["city1", "city2", "city3"],
    "rec_reasons2": ["rec1", "rec2", "rec3"],
    "rec_city3": ["city1", "city2", "city3"],
    "rec_reasons3": ["rec1", "rec2", "rec3"],
    "rec_city4": ["city1", "city2", "city3"],
    "rec_reasons4": ["rec1", "rec2", "rec3"],
    "rec_city5": ["city1", "city2", "city3"],
    "rec_reasons5": ["rec1", "rec2", "rec3"]
}
df = pd.DataFrame(data)

# Validate the DataFrame
validate_dataframe(df)
