import requests
import pickle

headers={
    "x-api-key":"1"
}
resp = requests.get("https://www.thesportsdb.com/api/v1/json/1/all_leagues.php")

if resp.status_code != 200:
    # This means something went wrong.
    with open("sports.pickle", 'rb') as f:
        sports = pickle.load(f)
else:
    leagues=[]
    sports=dict()
    for league in resp.json()["leagues"]:
        leagues_for_sport=dict()
        sport_name=league["strSport"]
        league_name=league["strLeague"]
        if sport_name not in sports:
            sports.update({sport_name:[league_name]})
        else:
            old_list=sports[sport_name]
            del(sports[sport_name])
            sports.update({sport_name:old_list+[league_name]})
    sports_leagues_teams = dict()
    for sport in sports:
        print(sports[sport])
        teams_per_sport = dict()

        for league in sports[sport]:

            resp = requests.get("https://www.thesportsdb.com/api/v1/json/1/search_all_teams.php?l="+league.replace(" ",
        "%20"))
            teams=[]
            if not resp.json()['teams']==None:
                for  x in resp.json()['teams']:
                    teams.append(x['strTeam'] )
            teams_per_sport.update({league:teams})
        sports_leagues_teams.update({sport:teams_per_sport})


    with open("sports.pickle", 'wb') as f:
        pickle.dump(sports_leagues_teams, f, pickle.HIGHEST_PROTOCOL)


print(sports_leagues_teams)

