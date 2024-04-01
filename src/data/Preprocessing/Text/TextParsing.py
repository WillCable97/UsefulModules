"""
    Functions for converting raw text inut into packaged input data of form:
        - [abc, def, hij, ..., xyz]
"""

def pivot_text(input_string: str, split_on: str = "\n", keep_pivot_string = True) -> list:
    """
        Will take an input string and split it on and input string
        The output of the function is of shape [[], [], ...,[]]
        - input_string: full input string
        - split_on: Character that marks end of sequence
        - keep_pivot_string: is the split string kept in the output, if true the split on character is appended to the end of each sequence
    """
    split_data = input_string.split(split_on)
    if keep_pivot_string: split_data = [x.append(split_on) for x in split_data]
    return split_data


def sequence_string(input_string: str, split_on: str, sequence_len: int
                    , discard_overflow = True, custom_char: str = None):
    """
        Will take an input string and create sequences of fixed length
        The output of the function is of shape [str1, str2, ...,strn]
        - input_string: full input string
        - split_on: Character that marks end of sequence
        - sequence_len: length of the sequences used
        - discard_overflow: keep the text that does not fit into fixed lenth
        - custom_char: custom character to split on
    """
    if split_on == "char":
        split_list = [i for i in range(len(input_string))]
    elif split_on == "word":
        split_list = find_positions(input_string, " ")
    else:
        split_list = find_positions(input_string, custom_char)

    #Perform split based on indexes
    overflow =  not (((len(split_list)+1) % sequence_len) ==0)
    updated_split_points = split_list[sequence_len-1::sequence_len]#[1:]
    list_splits = split_by_index(input=input_string, indexes=updated_split_points)
    if overflow: list_splits = list_splits[:-1]

    return list_splits



#####################################_HELPER FUNCTIONS_#####################################
def find_positions(text, char):
    positions = []
    index = text.find(char)
    while index != -1:
        positions.append(index)
        index = text.find(char, index + 1)
    return positions


def split_by_index(input: str, indexes:list) -> list:
    """
        Will split a string by index instead of character
    """
    if indexes == []: return [input]
    input_len = len(input)
    start_index, end_index = [indexes[0], indexes[-1]]

    if start_index!=0: indexes = [0] + indexes
    if input_len > end_index: indexes.append(input_len) 

    seq_count = len(indexes)-1 
    return [input[indexes[x]: indexes[x+1]] for x in range(seq_count)]










