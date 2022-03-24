import pydest
import requests
import os

# Not too hard to get your own
API_KEY = None

# Writes string to file
def str2data(name, s):
    f = open("./destguns_txt/" + name + ".txt", "w")
    f.write(s)
    f.close()

class InfoWrapper:
    def __init__(self, info):
        self.info = info
        self.incomplete = False
        try:
            self.name = self.info['displayProperties']['name']
            self.type = self.info['itemTypeDisplayName']
            self.desc = self.info['flavorText']
        except:
            self.incomplete = True

        # / in name breaks things cause of directories
        # theres not that many guns with this so just ignore it lol
        if self.name.find("/") != -1:
            self.incomplete = True 

    def complete(self):
        return not self.incomplete

    def needed(self):
        # Is the item this info corresponds to needed?
        # If we've already collected data on it, no need to do so again
        already_collected = os.path.exists("./destguns_txt/" + self.name + ".txt")
        return not already_collected

    # Writes itself into dataset (assumes it's complete)
    def writeSelf(self):
        str2data(self.name, self.type + "\n" + self.desc)

class SearchWrapper:
    def __init__(self, search_result):
        self.query = search_result['Response']['results']
        self.total_res = self.query['totalResults']
        self.has_more = self.query['hasMore']
        self.res = self.query['results']

        self.hashes = [res['hash'] for res in self.res]

async def every_frame():
    # What types of items to save?
    DESIRED_TYPES = ["Auto Rifle", "Hand Cannon", "Shotgun", "Submachine Gun", "Combat Bow",
            "Fusion Rifle", "Rocket Launcher", "Pulse Rifle", "Sidearm", "Scout Rifle", "Sniper Rifle",
            "Machine Gun", "Trace Rifle", "Grenade Launcher", "Sword", "Linear Fusion Rifle"]
    QUERIES_PER_MINUTE = 20
    sec_per_q = 60 / QUERIES_PER_MINUTE

    # Create all 3 letter permutations
    from itertools import product
    from string import ascii_lowercase
    keywords = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]
    total_size = len(keywords)
    total_progress = 898

    destiny = pydest.Pydest(API_KEY)
    try:
        while total_progress < total_size:
            page = 0
            kw = keywords[total_progress]
            if(kw == "bin"): # This word breaks things for some reason
                total_progess += 1
                continue
            print("[" + str(total_progress) + "/" + str(total_size) + "]: (" + kw + ")")
            while True:
                query = await destiny.api.search_destiny_entities('DestinyInventoryItemDefinition', kw, page)
                
                query = SearchWrapper(query)
                has_more = query.has_more
                if query.total_res == 0: break
                
                for hash_ in query.hashes:
                    info = await destiny.decode_hash(hash_, 'DestinyInventoryItemDefinition')
                    info = InfoWrapper(info)
                    if info.complete():
                        if info.type in DESIRED_TYPES and info.needed():
                            info.writeSelf();
                            await asyncio.sleep(sec_per_q)
                if has_more:
                    page += 1
                else:
                    break
            total_progress += 1

    except Exception as e:
        print(e)
        pass
    
    await destiny.close()

async def main():
    await every_frame()

import asyncio
import time

asyncio.run(main())
