from lcu_driver import Connector

connector = Connector()

async def create_custom_lobby(connection):
    custom = {
        'customGameLobby': {
            'configuration': {
                # PRACTICETOOL,CLASSIC
                'gameMode': 'PRACTICETOOL',
                'gameMutator': '',
                'gameServerRegion': '',
                'mapId': 11,
                'mutators': {'id': 1},
                'spectatorPolicy': 'AllAllowed',
                'teamSize': 1
            },
            'lobbyName': 'PRACTICETOOL',
            'lobbyPassword': ''
        },
        'isCustom': True
    }
    await connection.request('POST', '/lol-lobby/v2/lobby', data=custom)


# -----------------------------------------------------------------------------
# 添加单个机器人
# -----------------------------------------------------------------------------
async def add_bots_team1(connection):
    soraka = {
        'championId': 16,
        'botDifficulty': 'EASY',
        'teamId': '100'
    }
    await connection.request('POST', '/lol-lobby/v1/lobby/custom/bots', data=soraka)


# -----------------------------------------------------------------------------
# 批量添加机器人
# -----------------------------------------------------------------------------
async def add_bots_team2(connection):
    # 获取自定义模式电脑玩家列表
    activedata = await connection.request('GET', '/lol-lobby/v2/lobby/custom/available-bots')
    champions = {bot['name']: bot['id'] for bot in await activedata.json()}

    team2 = ['Ashe']

    for name in team2:
        bot = {'championId': champions[name], 'botDifficulty': 'MEDIUM', 'teamId': '200'}
        await connection.request('POST', '/lol-lobby/v1/lobby/custom/bots', data=bot)


# -----------------------------------------------------------------------------
# websocket
# -----------------------------------------------------------------------------
@connector.ready
async def connect(connection):
    # await get_summoner_data(connection)
    # await get_lockfile(connection)
    await create_custom_lobby(connection)
    # await add_bots_team1(connection)
    await add_bots_team2(connection)


def create():
    connector.start()


if __name__ == '__main__':
    create()
