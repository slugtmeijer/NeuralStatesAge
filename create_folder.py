import os

def create_folder(folder):
    # Check if the folder already exists
    if not os.path.exists(folder):
        # If it doesn't exist, create it
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")
    else:
        print(f"Folder '{folder}' already exists.")