import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "setup.py",
    "app.py",
    "requirements.txt",
    "research/trials.ipynb"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file name : {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)) == 0:
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating an Empty File : {filepath}")
        
    else:
        logging.info(f"{filename} already exists in the Directory.")