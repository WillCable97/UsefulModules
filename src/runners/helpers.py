import os

def model_path(model_name: str) -> str:
    base_bath = os.path.abspath("./") #to be changed
    model_path = os.path.join(base_bath, "models", model_name)
    return model_path


def create_dir(dir_path: str) -> bool: #Returns true of directory created 
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path)
        return True
    return False