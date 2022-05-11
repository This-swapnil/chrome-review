from email import header
from wsgiref.headers import Headers
import pandas as pd
import language_tool_python as ltp


# load the data
def read_data(file_name):
    data = pd.read_csv(file_name, usecols=['text'])
    data.dropna(inplace=True)
    return data


# grammar check function
def grammar_check(df):
    """Grammar check function to check the grammar of the text and return the list of errors and also provide the right suggestions.

    Args:
        df (_type_): dataset to check the grammar
    Returns:
        new csv file, if there is any error it show the suggestions and if not it will show No Corrections
    """
    df1 = pd.DataFrame()
    tool = ltp.LanguageTool('en-US')
    for i in df.index:
        text = df['text'][i]
        matches = tool.check(text)
        count = len(matches)
        if count == 0:
            print(text, " -- No mistakes")
            df1 = df1.append([[text, " No Mistak Found", "No Correction"]])
        else:
            print(text, " -- Mistakes found, ", count, " mistakes")
            df1 = df1.append(
                [[text, str(count) + " Mistak Found",
                  tool.correct(text)]])
    df1.to_csv('Correction.csv', header=['text', 'mistek', 'correction'])


if __name__ == "__main__":
    data = read_data("review_data.csv")
    grammar_check(data)
