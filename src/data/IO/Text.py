import os


"""
    Contains function for reading and writting text files
"""


#####################################_TEXT FILES_#####################################
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