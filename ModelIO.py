import os

def model_path(model_name: str) -> str:
    base_bath = os.path.abspath("./")
    model_path = os.path.join(base_bath, "models", model_name)
    return model_path


def create_dir(dir_path: str) -> bool: #Returns true of directory created 
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path)
        return True
    return False


def populate_model_details(input_data_obj, input_model):
    final_train = input_data_obj.final_train
    final_val = input_data_obj.final_val