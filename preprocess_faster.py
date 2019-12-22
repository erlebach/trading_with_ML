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


def processFiles(folder, dfs):
    files = glob.glob(folder + "/NASDAQ*.txt")
    for fn in files: #[0:10]:
        y,m,d = u.getDate(fn)
        # no need for the index_col if there is no index column in the file
        # being read. 
        df = pd.read_csv(fn)  # add an index [0,1,2,...]
        df.columns = ["sym", "date", "o", "h", "l", "c", "vol"]
        # Symbols are unique in fn
        dfs.append(df)
    
# Process all the years. Once that is done, concatenate the years for each symbol with a separate program. 
# Now that this works, convert the notebook to a Python program

dirs = glob.glob("NASDAQ*")
dfs = []
for folder in dirs: #[0:3]: 
    print("Folder= ", folder)
    processFiles(folder, dfs)

df = pd.concat(dfs)
print("finished concat, df.shape: ", df.shape)
    
# Get unique symbols in the index
syms = df['sym'].unique()
print("nb symbols: ", len(syms))

# use multi-indexing
#print(df.columns)
df = df.set_index(['sym','date']) # multi-index
#print(df.head(3))
#print(df.loc['ARTW'].sort_index())
#print(syms)
    
for sym in syms: #[0:2]:
    # Create condition on the row label
    dfsym = df.loc[sym].sort_index()
    print("dfsym.shape= ", dfsym.shape)
    # Perhaps I should remove the symbol name as well, since it the name of the file
    #dfsym.drop("sym", inplace=True, axis=1)
    #dfsym.to_csv(folder + "/" + sym + ".txt") #,index=True)
    dfsym.to_csv(sym + ".txt") #,index=True)
    print("sym= ", sym, folder)
