import pandas as pd

def submission_output(df_test, prediction):
    df_ids = df_test["id"]
    result = df_ids.to_frame()
    result["target"] = prediction
    return result
    