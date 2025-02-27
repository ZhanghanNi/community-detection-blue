import pandas as pd
import ast

"""
Simple script to merge all non PPI benchmarking results together 
into one csv 
"""


file_paths = [
    "../c_elegans/final_celegans_neural_benchmark_results.csv",
    "../college_football/final_college_football_benchmark_results.csv",
    "../karate_club/final_karate_club_benchmark_results_3.csv",
    "../urban_movement/final_urban_movement_synthetic_benchmark_results.csv",
]

dataframes = []

for file in file_paths:
    df = pd.read_csv(file)
    df["Source File"] = file
    
    # Reorder columns to make Source File the first column
    cols = ["Source File"] + [col for col in df.columns if col != "Source File"]
    df = df[cols]

    # After you've already loaded the CSV into your DataFrame, add something like this:
    df['GN_Community_Count'] = df['GN_Final_Communities'].apply(lambda x: len(ast.literal_eval(x)))
    df['LV_Community_Count'] = df['LV_Final_Communities'].apply(lambda x: len(ast.literal_eval(x)))
    df['BVNS_Community_Count'] = df['BVNS_Final_Communities'].apply(lambda x: len(ast.literal_eval(x)))

    dataframes.append(df)

# Concatenate all data frames into one and output it to a csv
merged_df = pd.concat(dataframes, ignore_index=True)

merged_df.to_csv("merged_raw_final_results.csv", index=False)

print("CSV files merged successfully into merged_raw_final_results.csv")
