import os

os.chdir('module')

for F in os.listdir():
  with open(F, "r+") as f:
    data = f.read()
    o = data.split('=')[-1]
    f.seek(0)
    f.write(o)
    f.truncate()