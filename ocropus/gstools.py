#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import os
import sys


# In[9]:


def gslist(arg: str):
    try:
        with os.popen(f"gsutil ls '{arg}'") as stream:
            lines = [s.strip() for s in list(stream.readlines())]
    except os.CommandException as exn:
        print("here", exn)
        return []
    return lines


# In[7]:


gslist("gs://ocropus4-models")


# In[21]:


def gsexists(path):
    status = os.system(f"gsutil ls '{path}' > /dev/null 2>&1")
    return status == 0

def gstouch(path):
    os.system(f"gsutil cp /dev/null '{path}'")
    
def gsrm(path):
    assert os.system(f"gsutil rm '{path}'") == 0
    
def gsrmf(path):
    os.system(f"gsutil rm -f '{path}'")
    
def gscp(src, dst):
    assert os.system(f"gsutil cp '{src}' '{dst}'") == 0
    
def gsread(src):
    return os.popen(f"gsutil cat '{src}'", "rb")

def gswrite(dst):
    return os.popen(f"gsutil cp - '{dst}'", "wb")

def gsmb(dst):
    assert os.system(f"gsutil mb '{dst}'") == 0


# In[18]:


gsexists("gs://ocropus4-models/wordmodel.pth"), gsexists("gs://ocropus4-models/wordmodel.pth.foo")


# In[30]:


def gsprocess(source, dest, nshards):
    sourcedir, sourcename = os.path.split(source)
    assert "%" in sourcename
    destdir, destname = os.path.split(dest)
    assert "%" in destname
    sources = gslist(sourcedir)
    sourcenames = [os.path.split(s)[1] for s in sources]
    dests = gslist(destdir)
    destnames = [os.path.split(s)[1] for s in dests]
    missing = []
    for i in range(nshards):
        assert sourcename % i in sourcenames, (sourcename % i, sourcenames)
        if destname %i not in destnames:
            missing.append(i)
    print(len(missing), "shards to be processed, out of", nshards)
    for i in missing:
        lockname = (destname % i) + ".lock"
        if lockname in destnames:
            continue
        if gsexists(destdir + "/" + lockname):
            continue
        gstouch(destdir + "/" + lockname)
        yield source % i, dest % i
        gsrm(destdir + "/" + lockname)   


# In[31]:


#gsmb("gs://tmbdev-temp")
for src, dst in gsprocess("gs://nvdata-ocropus/ia1-%06d.tar", "gs://tmbdev-temp/ia1-%06d.tar", 34):
    print(src, dst)


# In[ ]:




