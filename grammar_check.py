import pandas as pd
import language_tool_python as ltp


# load the data
def read_data(file_name):
    data = pd.read_csv(file_name, usecols=['text'])
    return data


# grammar check function
def grammar_check(df):
    tool = ltp.LanguageTool('en-US')
    for i in df.index:
        text = df['text'][i]
        matches = tool.check(text)
        count = len(matches)
        if count == 0:
            print(text, " -- No mistakes")
        else:
            print(text, " -- Mistakes found, ", count, " mistakes")


data = read_data("review_data.csv")
grammar_check(data)