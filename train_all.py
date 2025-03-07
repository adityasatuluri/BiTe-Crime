import subprocess

region = int(input("Enter region: "))
category = int(input("Enter category: "))

# Define the ranges for ID1 and ID2
id1_range = range(region, 78)  # ID1
id2_range = range(1,5)   # ID2

# Loop over the range of ID1 and ID2
for ID1 in id1_range:
    for ID2 in id2_range:
        # Construct the command
        if ID1 == region and ID2 < category:
            continue
        cmd = f"python train.py --tct=chicago --tr={ID1} --tc={ID2}"
        
        print(f"{cmd}")

        # Execute the command in the CMD terminal
        subprocess.run(cmd, shell=True)
        
        print(f"Executed: {cmd}") 