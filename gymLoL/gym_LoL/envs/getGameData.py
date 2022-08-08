import requests
import warnings
# import scan
import json
import time
import math
import psutil

# import actions
# import cli

warnings.filterwarnings('ignore')

ENDPOINTS = {
    "allgamedata": "https://127.0.0.1:2999/liveclientdata/allgamedata",  # all game data
    "activeplayer": "https://127.0.0.1:2999/liveclientdata/activeplayer",  # active player
    "activeplayername": "https://127.0.0.1:2999/liveclientdata/activeplayername",  # summonerName
    "activeplayerabilities": "https://127.0.0.1:2999/liveclientdata/activeplayerabilities",  # spell data
    "playerlist": "https://127.0.0.1:2999/liveclientdata/playerlist",  # all player data
    "playerscores": "https://127.0.0.1:2999/liveclientdata/playerscores?summonerName=",
    # player scores with  summonerName
    "eventdata": "https://127.0.0.1:2999/liveclientdata/eventdata",  # event data
    "gamestats": "https://127.0.0.1:2999/liveclientdata/gamestats"

}


class player:
    def __init__(self, championName='ee'):
        # set information at the beginning
        self.summonerName = None
        self.champ_info = None
        self.currentLevel = 1
        self.itemNumber = 0

        # playerState
        self.health = None
        self.mana = None

        # playerscores
        self.kills = 0
        self.deaths = 0
        self.assists = 0
        self.creepScore = 0

        self.abilitiesQ = False
        self.abilitiesW = False
        self.abilitiesE = False
        self.abilitiesR = False

        # keep update
        self.stats = None
        self.playerAbilities = None
        self.playerScores = None


    def get_gold(self):
        return math.floor(self.stats["currentGold"])

    def get_level(self):
        return self.stats["level"]

    def get_time(self):
        return requests.get(ENDPOINTS["allgamedata"], verify=False).json()["gameData"]["gameTime"]

    def get_regen(self):
        return self.stats["championStats"]["healthRegenRate"]

    def update(self):
        self.summonerName = requests.get(ENDPOINTS["activeplayername"], verify=False).json()
        self.stats = requests.get(ENDPOINTS["activeplayer"], verify=False).json()
        self.playerScores = requests.get(ENDPOINTS["playerscores"] + self.summonerName, verify=False).json()
        self.playerAbilities = requests.get(ENDPOINTS["activeplayerabilities"], verify=False).json()
        self.health = self.stats["championStats"]["currentHealth"] / self.stats["championStats"]["maxHealth"] * 100
        self.mana = self.stats["championStats"]["resourceValue"] / self.stats["championStats"]["resourceMax"] * 100
        self.kills = self.playerScores["kills"]
        self.deaths = self.playerScores["deaths"]
        self.assists = self.playerScores["assists"]

        self.abilitiesQ = self.playerAbilities['Q']['abilityLevel'] > 0
        self.abilitiesW = self.playerAbilities['W']['abilityLevel'] > 0
        self.abilitiesE = self.playerAbilities['E']['abilityLevel'] > 0
        self.abilitiesR = self.playerAbilities['R']['abilityLevel'] > 0

    # def getData(self, name):
    #     response = requests.get(ENDPOINTS[f'{name}'], verify=False)
    #     return response.json()

    # check to make sure that player is in a live game.
    def is_live(self):
        print("Waiting for league to start...")
        gameOpen = "League of Legends.exe" in (i.name() for i in psutil.process_iter())

        while gameOpen == False:
            time.sleep(0.5)
            gameOpen = "League of Legends.exe" in (i.name() for i in psutil.process_iter())

        if gameOpen == True:
            return True
        else:
            return False


if __name__ == '__main__':
    player = player()
    start = time.time()
    player.update()
    print(player.mana)

    # data = requests.get(ENDPOINTS["allgamedata"], verify=False).json()
    # print(data)
    # {
    #     'activePlayer':
    #         {
    #             'abilities':
    #                 {
    #                     'E': {
    #                         'abilityLevel': 0,
    #                         'displayName': '90 Caliber Net',
    #                         'id': 'CaitlynE',
    #                         'rawDescription': 'GeneratedTip_Spell_CaitlynE_Description',
    #                         'rawDisplayName': 'GeneratedTip_Spell_CaitlynE_DisplayName'
    #                     },
    #                     'Passive': {
    #                         'displayName': 'Headshot',
    #                         'id': 'CaitlynPassive',
    #                         'rawDescription': 'GeneratedTip_Passive_CaitlynPassive_Description',
    #                         'rawDisplayName': 'GeneratedTip_Passive_CaitlynPassive_DisplayName'
    #                     },
    #                     'Q': {
    #                         'abilityLevel': 0,
    #                         'displayName': 'Piltover Peacemaker', 'id': 'CaitlynQ',
    #                         'rawDescription': 'GeneratedTip_Spell_CaitlynQ_Description',
    #                         'rawDisplayName': 'GeneratedTip_Spell_CaitlynQ_DisplayName'
    #                     },
    #                     'R': {
    #                         'abilityLevel': 0,
    #                         'displayName': 'Ace in the Hole',
    #                         'id': 'CaitlynR',
    #                         'rawDescription': 'GeneratedTip_Spell_CaitlynR_Description',
    #                         'rawDisplayName': 'GeneratedTip_Spell_CaitlynR_DisplayName'
    #                     },
    #                     'W': {
    #                         'abilityLevel': 0,
    #                         'displayName': 'Yordle Snap Trap',
    #                         'id': 'CaitlynW',
    #                         'rawDescription': 'GeneratedTip_Spell_CaitlynW_Description',
    #                         'rawDisplayName': 'GeneratedTip_Spell_CaitlynW_DisplayName'
    #                     }
    #                 },
    #             'championStats': {
    #                 'abilityHaste': 0.0,
    #                 'attackDamage': 67.4000015258789,
    #                 'attackRange': 650.0,
    #                 'attackSpeed': 0.7378000020980835,
    #                 'currentHealth': 580.0,
    #                 'healShieldPower': 0.0,
    #                 'healthRegenRate': 0.699999988079071,
    #                 'maxHealth': 580.0,
    #                 'moveSpeed': 325.0,
    #                 'resourceMax': 315.0,
    #                 'resourceRegenRate': 1.4800000190734863,
    #                 'resourceType': 'MANA',
    #                 'resourceValue': 315.0,
    #                 'spellVamp': 0.0,
    #                 'tenacity': 0.0},
    #             'currentGold': 518.3599853515625,
    #             'level': 1,
    #             'summonerName': 'Awesome World YJ',
    #             'allPlayers': [
    #                 {
    #                     'championName': 'Caitlyn',
    #                     'isBot': False,
    #                     'isDead': False,
    #                     'items': [],
    #                     'level': 1,
    #                     'position': '',
    #                     'rawChampionName': 'game_character_displayname_Caitlyn',
    #                     'rawSkinName': 'game_character_skin_displayname_Caitlyn_13',
    #                     'respawnTimer': 0.0,
    #                     'scores':
    #                         {
    #                             'assists': 0,
    #                             'creepScore': 0,
    #                             'deaths': 0,
    #                             'kills': 0,
    #                             'wardScore': 0.0
    #                         },
    #                     'summonerName': 'Awesome World YJ',
    #                     'summonerSpells':
    #                         {
    #                             'summonerSpellOne':
    #                                 {'displayName': 'Heal'
    #                                  },
    #                             'summonerSpellTwo':
    #                                 {
    #                                     'displayName': 'Flash'
    #                                 }
    #                         },
    #                     'team': 'ORDER'},
    #                 {
    #                     'championName': 'Ezreal',
    #                     'isBot': True,
    #                     'isDead': False,
    #                     'level': 1,
    #                     'position': '',
    #                     'rawChampionName': 'game_character_displayname_Ezreal',
    #                     'respawnTimer': 0.0,
    #                     'scores': {'assists': 0, 'creepScore': 0, 'deaths': 0, 'kills': 0, 'wardScore': 0.0},
    #                     'summonerName': 'Ezreal Bot',
    #                     'summonerSpells': {
    #                         'summonerSpellOne':
    #                             {
    #                                 'displayName': 'Ignite'
    #                             },
    #                         'summonerSpellTwo':
    #                             {
    #                                 'displayName': 'Exhaust'
    #                             }
    #                     },
    #                     'team': 'CHAOS'}],
    #             'events': {
    #                 'Events': [{'EventID': 0, 'EventName': 'GameStart', 'EventTime': 0.010380400344729424},
    #                            {'EventID': 1, 'EventName': 'MinionsSpawning', 'EventTime': 65.00933074951172}]},
    #             'gameData': {'gameMode': 'PRACTICETOOL', 'gameTime': 118.65847778320312, 'mapName': 'Map11',
    #                          'mapNumber': 11,
    #                          'mapTerrain': 'Default'}}
