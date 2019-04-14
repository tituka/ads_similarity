import re
import pickle
from nltk import word_tokenize
import time


with open("sports.pickle", 'rb') as f:
    sports = pickle.load(f)
searchstring="Shanghai East v. Tianjin Teda"
teams=re.split("\sversus\s|\sv\s|\sv\s|\sv.\s|\svs.\s", searchstring, flags=re.IGNORECASE)
teams=(teams[0], teams[1])

sports1=dict()
for sport in sports:
    sd=dict()
    for league in sports[sport]:
        new_teams=[]
        for team in sports[sport][league]:
            new_teams.append([x.lower() for x in word_tokenize(team)])
        sd.update({league:new_teams})
    sports1.update({sport:sd})

def matches_end(short, long):

    short_token=[x.lower() for x in word_tokenize(short)]
    long_token=[x.lower() for x in word_tokenize(long)]

    if set(short_token)<=set(long_token):
        return True
    else:
        return False
def search_sports(sports, teams):

    for sport in sports:
        for league in sports[sport]:
            sport_league_teams = []
            matched_first=False
            matched_second=False
            for team in sports[sport][league]:
                if  matches_end(teams[0], team):
                    matched_first=True
                    if matched_second:
                        sport_league_teams.append(team)
                        return sport_league_teams
                    else:
                        sport_league_teams+=[sport, league, team]

                if  matches_end(teams[1], team):
                    matched_second=True
                    if matched_first:
                        sport_league_teams.append(team)
                        return  sport_league_teams
                    else:
                        sport_league_teams+=[sport, league, team]




print("With search string" +" '"+searchstring+"', found:")
t0 = time.time()

print(search_sports(sports, teams))
t1 = time.time()

total = t1-t0
#print("In time "+str(total))