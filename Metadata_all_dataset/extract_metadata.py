import os
import json
import shutil
#To convert all metadata.json file into a csv file

import csv
import re
import pandas as pd
from pandas.io.json import json_normalize
import collections
import io

def modify_all_files():
    rows =[]
    csvfile = open('Metadata_all_dataset/convertcsv.csv', 'r')
    rdr = csv.reader(csvfile)
    for i in rdr:
        rows = i
        break

    for folder in os.listdir("smithsonian"):
        if os.path.isdir("smithsonian/" + folder):
            for subfolder in os.listdir("smithsonian/" + folder):
                if os.path.isdir("smithsonian/" + folder + "/" + subfolder):
                    for filename in os.listdir("smithsonian/" + folder + "/" + subfolder):
                        with open("smithsonian/" + folder + "/" + subfolder + "/" + "metadata.json") as json_file:
                            data = json.load(json_file)
                            json_file.close()
                            k = []
                            for keys in data:
                                k.append(str(keys))
                            for r in rows:
                                if r not in k:
                                    data[r] = None
                            ordered_d = collections.OrderedDict(sorted(data.items()))
                            json_file = open("smithsonian/" + folder + "/" + subfolder + "/" + "metadata.json", 'w+')
                            json_file.write(json.dumps(ordered_d))
                            json_file.close()
                            break

def extract_data():
    count = 0
    for folder in os.listdir("smithsonian"):
        if os.path.isdir("smithsonian/" + folder):
            for subfolder in os.listdir("smithsonian/" + folder):
                if os.path.isdir("smithsonian/" + folder + "/" + subfolder):
                    for filename in os.listdir("smithsonian/" + folder + "/" + subfolder):
                        with open("smithsonian/" + folder + "/" + subfolder + "/" + "metadata.json") as json_file:
                            data = json.load(json_file)
                            json_file.close()
                            metadata = pd.DataFrame(json_normalize(data))
                            if count == 0:
                                metadata.to_csv('all_metadata.csv',mode= 'a', encoding='utf-8',line_terminator='\n',index=False )
                                count = count + 1
                            else:
                                metadata.to_csv('all_metadata.csv',mode= 'a', encoding='utf-8',header=False,line_terminator='\n',index=False)
                            break
                            


'''
def modify_json_files():
    rows =[]
    csvfile = open('convertcsv.csv', 'r')
    rdr = csv.reader(csvfile)
    for i in rdr:
        rows = i
        break 
    json_file = open('metadata.json', 'r')
    data = json.load(json_file)
    json_file.close()
    k = []
    for keys in data:
        k.append(str(keys))
    for r in rows:
        if r not in k:
            data[r] = None
    ordered_d = collections.OrderedDict(sorted(data.items()))
    json_file = open('metadata.json','w+')
    json_file.write(json.dumps(ordered_d))
    json_file.close
'''
        

def main():
    #modify_json_files()
    modify_all_files()
    extract_data()

if __name__ == "__main__":
    main()