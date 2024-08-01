import pandas as pd
from pydantic import BaseModel, validator, ValidationError

# Define the Pydantic model
class DataFrameSchema(BaseModel):
    system: str
    query: str
    city_1_name: str
    city_1_rec: str
    city_2_name: str
    city_2_rec: str
    city_3_name: str
    city_3_rec: str
    city_4_name: str
    city_4_rec: str
    city_5_name: str
    city_5_rec: str
    
    @validator('*')
    def check_is_string(cls, v):
        if not isinstance(v, str):
            raise ValueError('Must be a string')
        return v

# Function to validate the DataFrame
def validate_dataframe(df: pd.DataFrame):
    # Check if the dataframe has the right columns
    required_columns = {"system", "query", "city 1 name", "city 1 rec", \
                        "city 2 name", "city 2 rec", "city 3 name", "city 3 rec", "city 4 name", "city 4 rec"\
                            "city 5 name", "city 5 rec"}
    if set(df.columns) != required_columns:
        raise ValueError(f"DataFrame must contain the columns: {required_columns}")

    # Validate each row
    for idx, row in df.iterrows():
        try:
            DataFrameSchema(
                system=row['system'],
                query=row['query'],
                city_1_name=row['city 1 name'],
                city_1_rec=row['city 1 rec'],
                city_1_name=row['city 2 name'],
                city_1_rec=row['city 2 rec'],
                city_1_name=row['city 3 name'],
                city_1_rec=row['city 3 rec'],
                city_1_name=row['city 4 name'],
                city_1_rec=row['city 4 rec'],
                city_1_name=row['city 5 name'],
                city_1_rec=row['city 5 rec']
            )
        except ValidationError as e:
            print(f"Row {idx} is invalid: {e}")

# Example DataFrame
data = {
    "system": ["sys1", "sys2", "sys3"],
    "query": ["query1", "query2", "query3"],
    "city 1 name": ["city1", "city2", "city3"],
    "city 1 rec": ["rec1", "rec2", "rec3"],
    "city 2 name": ["city1", "city2", "city3"],
    "city 2 rec": ["rec1", "rec2", "rec3"],
    "city 3 name": ["city1", "city2", "city3"],
    "city 3 rec": ["rec1", "rec2", "rec3"],
    "city 4 name": ["city1", "city2", "city3"],
    "city 4 rec": ["rec1", "rec2", "rec3"],
    "city 5 name": ["city1", "city2", "city3"],
    "city 5 rec": ["rec1", "rec2", "rec3"]
}
df = pd.DataFrame(data)

# Validate the DataFrame
validate_dataframe(df)