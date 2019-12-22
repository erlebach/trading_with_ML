# coding: utf-8

# Starting with raw stock data, generate a Pandas dataframe to simplify post-processing. We will create one file for each stock, that will span 20 years. The data will be stored in a Pandas file and stored either in csv or binary format. We will perform some timings to evaluate the most effect approach. 
# 
# Each file has a list of all stocks on the NASDAQ and corresponds to a single day. So we must: 
# - identify the day, month, year from the filename

# In[ ]:


import numpy as np
import pandas as pd
import glob as glob
import re 
import scanf
import utils as u


# In[ ]:


# In[ ]:


def processFiles(folder, files):
    dfs = []
    for fn in files:
        y,m,d = u.getDate(fn)
        # no need for the index_col if there is no index column in the file
        # being read. 
        df = pd.read_csv(fn, index_col=0) #, index_col=0)
        df.columns = ["date", "o", "h", "l", "c", "vol"]
        # Symbols are unique in fn
        df.index.names = ['sym']
        dfs.append(df)
    
    df = pd.concat(dfs)
    
    # Get unique symbols in the index
    syms = df.index.values
    
    for sym in syms:
        # Create condition on the row label
        dfsym = df.loc[df.index == sym]
        # Perhaps I should remove the symbol name as well, since it the name of the file
        #dfsym.drop("sym", inplace=True, axis=1)
        dfsym.to_csv(folder + "/" + sym + ".txt",index=True)
        print("sym= ", sym, fn, folder)
    
# Process all the years. Once that is done, concatenate the years for each symbol with a separate program. 
# Now that this works, convert the notebook to a Python program

dirs = glob.glob("NASDAQ*")
for d in dirs: 
    print("dir= ", d)
    files = glob.glob(d + "/NASDAQ*.txt")
    processFiles(d, files)

