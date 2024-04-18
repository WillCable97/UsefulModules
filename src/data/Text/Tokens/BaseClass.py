"""
    Base class for tokenising class objects. Responisble to converting text input into sequences of integer values


    1- generate_vocab: function responsible for taking initial raw input and generating the token mapping
        (a) Will also define the vocab size here
    2- tokenise: perform tokenising on an input string, this will use the vocab mapping from the generate_vocab function
    3- detokenise: reverse tokenising: take list of tokens and return string values


    ------init function------
    1- Handle specific token implementing (this will initialize pre existing tokeniserd if existing) 
        or created variabled for custom token objects
    2- Will contain and define some high level, class variables, these are:
        (a) pad_token-- assumed 0
        (b) start string: the string of character that represents the beginning of a sequence
        (b) end string: the string or character that represents the end of a sequence
"""
from abc import ABC, abstractmethod
import tensorflow as tf

class TokenBaseClass(ABC):
    """
        Base class definition
    """

    @abstractmethod
    def generate_vocab(self, input: list) -> None:
        """
            Generate token vocabulary from input text
        """

    @abstractmethod
    def tokenise(self, input: list, sequence_length: int = None) -> tf.Tensor:
        """
            Perform token conversion
        """

    @abstractmethod
    def detokenise(self, input: tf.Tensor, display_padding = False) -> list:
        """
            Perform conversion back to text
        """