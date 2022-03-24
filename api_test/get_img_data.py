import pydest
from PIL import Image
import requests
import os

# Not too hard to get your own
API_KEY = None

# Gets tuple of:
# entity name
# image icon image urls from an api entity response
def get_icons(call_dict):
    result_list = call_dict['Response']['results']['results']

    name_list = [item['displayProperties'] for item in result_list]
    name_list = [item['name'] for item in name_list]

    im_list = [item['screenshot'] for item in result_list]

    return [(name, im) for (name, im) in zip(name_list, im_list)]

# Writes url of image to path
async def url2data(im_url, path):
    with open(path, 'wb') as handle:
        response = requests.get('https://bungie.net' + im_url, stream = True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

class InfoWrapper:
    def __init__(self, info):
        self.info = info
        self.incomplete = False
        try:
            self.name = self.info['displayProperties']['name']
            self.id = self.info['collectibleHash'] # Better as an identifier
            self.ico_url = self.info['displayProperties']['icon']
            self.type = self.info['itemTypeDisplayName']
            self.ss_url = self.info['screenshot']
        except:
            self.incomplete = True

    # Entry should be ignored if its incomplete
    def complete(self):
        return not self.incomplete

    def needed(self):
        # Is the item this info corresponds to needed?
        # If we've already collected data on it, no need to do so again
        already_collected = os.path.exists("./destguns_img/" + str(self.id) + ".jpg")
        return not already_collected

    # scrape this entry (get screenshot and icon)
    async def scrape_entry(self):
        await url2data(self.ico_url, "./destguns_ico/" + str(self.id) + ".jpg")
        await url2data(self.ss_url, "./destguns_img/" + str(self.id) + ".jpg")

class SearchWrapper:
    def __init__(self, search_result):
        self.query = search_result['Response']['results']
        self.total_res = self.query['totalResults']
        self.has_more = self.query['hasMore']
        self.res = self.query['results']

        self.hashes = [res['hash'] for res in self.res]

async def every_frame():
    total_progress = 0

    # What types of items to save?
    DESIRED_TYPES = ["Weapon Ornament", "Auto Rifle", "Hand Cannon", "Shotgun", "Submachine Gun", "Combat Bow",
                "Fusion Rifle", "Rocket Launcher", "Pulse Rifle", "Sidearm", "Scout Rifle", "Sniper Rifle",
              "Machine Gun", "Trace Rifle", "Grenade Launcher", "Linear Fusion Rifle"]
    
    QUERIES_PER_MINUTE = 20
    sec_per_q = 60 / QUERIES_PER_MINUTE

    # Create all 3 letter permutations
    from itertools import product
    from string import ascii_lowercase
    keywords = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]
    total_size = len(keywords)

    destiny = pydest.Pydest(API_KEY)
    try:
        while total_progress < total_size:
            page = 0
            kw = keywords[total_progress]
            if(kw == "bin"): # This word breaks things for some reason
                total_progress += 1
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
                            await info.scrape_entry()
                            await asyncio.sleep(1)
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
