 ################ GENERATE METHODS ################
def generate_prompt_pred(input_):
    return f"""
### Input:
{input_}
### Answer:"""

def generate_prompt_str(input_):
    return f"""
### Input:
{input_}
### Answer:"""

def generate_prompt(data_point):
    return f"""
### Input:
{data_point["input"]}
### Answer:
{data_point["output"]}"""