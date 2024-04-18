import os


"""
    Contains function for reading and writting text files
"""


#####################################_TEXT FILES (BASE)_#####################################
def read_text_file(file_path: str, encoding:str = None) -> str:
    """
        Reading in data from a text file. 
        Will return a continuous string containing the entire contents of the file
        - file_path: string object for file path to needed text file
        - encoding: how is the data in the text file encoded, to be passed into open function
    """
    file = open(file_path, "r", encoding=encoding)
    text = file.read()
    return text


def write_text_file(file_path: str, content:str):
    """
        Writting data to a text file. 
        - file_path: string object for file path to needed text file
        - content: The content to be dumped into file
    """
    file = open(file_path, "w+")
    file.write(content)





#####################################_FOR RETURNING COLLECTIONS OF RAW DATA_#####################################
def base_train_val_paths(parent_path, encoding:str = None, file_prefix=""):
    base_path = os.path.join(parent_path, f"{file_prefix}base.txt")
    train_path = os.path.join(parent_path, f"{file_prefix}train.txt")
    val_path = os.path.join(parent_path, f"{file_prefix}val.txt")
    print(f"######__CUSTOM RUN LOG: Successfully read text file with prefix: {file_prefix}")
    return base_path, train_path, val_path
