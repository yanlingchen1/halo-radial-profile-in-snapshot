import pandas as pd
import numpy as np

# Create a sample multidimensional NumPy array
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Convert the NumPy array to a DataFrame
df = pd.DataFrame(array)

# Specify the file path for the CSV file
csv_file_path = 'output.csv'

# Define column names
column_names = ['Column 1', 'Column 2', 'Column 3']

# Save the DataFrame to a CSV file with a header
df.to_csv(csv_file_path, index=False, header=column_names)
