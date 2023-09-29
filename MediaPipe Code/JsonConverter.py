import pandas as pd
import json

# Load JSON data
with open('index_finger_data.json', 'r') as json_file:
    data = json.load(json_file)

# Convert JSON data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save DataFrame to Excel
df.to_excel('index_finger_data.xlsx', index=False)
