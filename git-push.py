import os

os.system("python3 -m pip freeze > requirements.txt")
os.system("git add .")
os.system("git commit -m ';'")
os.system("git push")
