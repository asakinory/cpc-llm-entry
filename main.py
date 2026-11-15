"""Send one live request for a single trial; a quick check of the setup.

    python main.py            # trial in row 20 of the raw data, first condition
"""

import pandas as pd

from cpc18_row_to_english import cpc18_row_to_english, get_cpc18_row_and_history
from experiment_config import RAW_DATA_PATH, conditions
from predict_request_openai import choice_request

ROW_NUMBER = 20  # one-based row of rawData_Comp_All.csv
condition = conditions()[0]

data = pd.read_csv(RAW_DATA_PATH)
history, row = get_cpc18_row_and_history(data, ROW_NUMBER)
prompt = cpc18_row_to_english(row, history_rows=history)
result = choice_request(prompt, condition)

print(prompt)
print()
print("Condition:        ", condition.id)
print("Model answer:     ", result.answer)
print("Participant chose:", "B" if row["B"] else "A")
print("log p(A), log p(B):", result.log_likelihood_a, result.log_likelihood_b)
print("Cost (USD):       ", result.cost)
print("Model:            ", result.model)
