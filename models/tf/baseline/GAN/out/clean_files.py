import os
import glob

files = glob.glob('./metric/*')
for f in files:
    if f[-5:] is 'empty':
        continue
    os.remove(f)

files = glob.glob('./evolution_hist/*')
for f in files:
    if f[-5:] is 'empty':
        continue
    os.remove(f)

files = glob.glob('./losses/*')
for f in files:
    if f[-5:] is 'empty':
        continue
    os.remove(f)

files = glob.glob('./saved_weights/*')
for f in files:
    if f[-5:] is 'empty':
        continue
    os.remove(f)

files = glob.glob('./tensorboard_files/*')
for f in files:
    if f[-5:] is 'empty':
        continue
    os.remove(f)
