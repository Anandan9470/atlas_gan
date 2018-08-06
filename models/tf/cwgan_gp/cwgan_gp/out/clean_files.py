import os
import glob

files = glob.glob('./chi2/*')
for f in files:
    os.remove(f)

files = glob.glob('./evolution_hist_65/*')
for f in files:
    os.remove(f)

files = glob.glob('./evolution_hist_524/*')
for f in files:
    os.remove(f)

files = glob.glob('./losses/*')
for f in files:
    os.remove(f)

files = glob.glob('./saved_weights/*')
for f in files:
    os.remove(f)

files = glob.glob('./tensorboard_files/*')
for f in files:
    os.remove(f)
