import pydest

async def every_frame():
    destiny = pydest.Pydest('53c540dc8efa44b5a5181214518283b6')
    try:
        json = await destiny.api.search_destiny_entities('DestinyInventoryItemDefinition', '')
    except Exception as e:
        print(e)
        pass
    print(json)
    await destiny.close()

async def main():
    await every_frame()

import asyncio
import time

asyncio.run(main())
