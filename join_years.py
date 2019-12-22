# coding: utf-8

# Given symbols files for several years, combine them into a single file per symbol
# 
# Each file has a list of all stocks on the NASDAQ and corresponds to a single day. So we must: 
# - identify the day, month, year from the filename

# In[ ]:


import numpy as np
import pandas as pd
import glob as glob
import re 
import scanf

folders = glob.glob("NASDAQ*")
years = []
for f in folders:
    # scanf.scanf returns a list (even a list of 1)
    y = scanf.scanf("NASDAQ_%d", f)[0]
    years.append(y)
print(years)
years.sort() # in place

# year_d[2008][sym] = list of data frames
#years_d = {}
sym_d = {}
for y in years:
    folder = "NASDAQ_" + str(y)
    syms = glob.glob(folder + "/" + "*[A-Z].txt")
    for sym_fn in syms:
        f, s = scanf.scanf("%s/%s.txt", sym_fn)
        sym_df = pd.read_csv(sym_fn, index_col=0)
        try:
        	sym_d[s].append(sym_df)
        except:
            sym_d[s] = []
            sym_d[s].append(sym_df)
        print(len(sym_d), len(sym_d[s]))

for k,v in sym_d.items():
	df = pd.concat(sym_d[k])
	fn = k + ".txt"
	df.to_csv(fn)
    

def getDate(fn):
    s,y,m,d = scanf.scanf("%s_%4d%2d%2d", fn)
    return y,m,d


# In[ ]:



"""
def processFiles(folder, files):
    dfs = []
    for fn in files:
        y,m,d = getDate(fn)
        df = pd.read_csv(fn)
        df.columns = ["sym", "date", "o", "h", "l", "c", "vol"]
        dfs.append(df)
    
    df = pd.concat(dfs)
    syms = df.sym.unique()
    
    df = df.sort_values(by=['sym'])
    
    for sym in syms:
        dfsym = df[df['sym'] == sym]
        dfsym = dfsym.sort_values(by='date')
        dfsym = dfsym.reset_index() #inplace=True)
        dfsym.drop("index", inplace=True, axis=1)   # there should be a better way
        # Perhaps I should remove the symbol name as well, since it the name of the file
        dfsym.drop("sym", inplace=True, axis=1)
        dfsym.to_csv(folder + "/" + sym + ".txt",index=True)
    
# Process all the years. Once that is done, concatenate the years for each symbol with a separate program. 
# Now that this works, convert the notebook to a Python program

dirs = glob.glob("NASDAQ*")
for d in dirs: 
    files = glob.glob(d + "/NASDAQ*.txt")
    processFiles(d, files)

"""
