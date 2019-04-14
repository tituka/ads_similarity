import re
import pickle
from nltk import word_tokenize
import time
import sys
import os
cwd = os.getcwd()
print(cwd)


with open("api/sports.pickle", 'rb') as f:
    sports = pickle.load(f)

def parse_and_match(searchstring):
    teams=re.split("\sversus\s|\sv\s|\sv\s|\sv.\s|\svs.\s", searchstring, flags=re.IGNORECASE)
    if len(teams)<2:
        print("Did not find corect format")
        return  False, []
    else:
        print(teams)
        teams=((teams[0], teams[1]), ([x.lower() for x in word_tokenize(teams[0])], [x.lower() for x in word_tokenize(
            teams[1])]))
        return search_sports(sports, teams)


def matches_end(short_token, long_token):
    if set(short_token)<=set(long_token):
        return True
    else:
        return False

def search_sports(sports, teams_pair):
    teams=teams_pair[1]
    for sport in sports:
        print("here")
        for league in sports[sport]:
            sport_league_teams = []
            matched_first=False
            matched_second=False
            for team in sports[sport][league]:
                if  matches_end(teams[0], team[1]):


                    matched_first=True
                    if matched_second:
                        sport_league_teams.append(team[0])
                        return True, sport_league_teams
                    else:
                        sport_league_teams+=[sport, league, team[0]]

                if  matches_end(teams[1], team[1]):
                    matched_second=True
                    if matched_first:
                        sport_league_teams.append(team[0])
                        return  True, sport_league_teams
                    else:
                        sport_league_teams+=[sport, league, team[0]]
    print("correct format, no teams")
    return False, sport_league_teams

"""ss="Juventus vs. Milsssan"


print("With search string" +" '"+ss+"', found:")
t0 = time.time()

print(parse_and_match(ss))
t1 = time.time()

total = t1-t0
print("In time "+str(total))"""
