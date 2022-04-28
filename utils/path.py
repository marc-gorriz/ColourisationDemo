import os, glob, re
from pathlib import Path

def listdir(path, last=False, N=None, prefix=None, sort=True):
    if prefix == None:
        prefix = "*"
    else:
        prefix = "*.%s" % prefix
    l = glob.glob(os.path.join(path, prefix))
    if last:
        l = [i.split("/")[-1] for i in l]
    if N != None:
        shuffle(l)
        l = l[:N]
    if sort:
        if "." in l[0]:
            p = l[0].split(".")[-1]
            ll = [i.split(".")[0] for i in l]
            ll = sort_list(ll)
            l = [i + ".%s" % p for i in ll]
        else:
            l = sort_list(l)
    return l

def mkdir(path_name):
    path = Path(path_name)
    path.mkdir(parents=True, exist_ok=True)
    
def sort_list(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)