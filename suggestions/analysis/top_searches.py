import pandas as pd
from nltk import word_tokenize
file_name="/home/tiina/Downloads/hypersonica_daily_report_2018-11-20.xlsx"
xl_file = pd.ExcelFile(file_name)
stopwords = [ "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours    ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"];

def has_lower(s_string):
    tokenized=word_tokenize(s_string)
    for token in tokenized:
        if token[0].islower() and not token[0]=="i" and not token in stopwords and not token =="vs":
            return True
    return False

dfs = {sheet_name: xl_file.parse(sheet_name)
          for sheet_name in xl_file.sheet_names}
print(dfs["Top Keywords Report"].columns.values)
print(dfs["Top Keywords Report"].sort_values(by=['Impressions']))
lowers=[]
for search in dfs["Top Keywords Report"]["Top Keywords by Paid Clicks"]:
    lowers.append(str(has_lower(search)))


with open("has_lower_20", "w" ) as f:
    for l in lowers:
        f.write(l+'\n')