from datetime import date
import os

directory = "training_logs"
file_date = directory + "/exec" + str(date.today()) + ".txt"

os.makedirs(directory, exist_ok=True)

aa = open(file_date, 'a')
print(f'File: {aa}', file=aa)

aa.close()
