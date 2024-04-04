"""from src.data.Preprocessing.Text.Tokenise.TensorflowTokenisers import TensorflowWordTokeniser
from src.data.Preprocessing.Text.Tokenise.KerasNlpTokenisers import TensorflowWordPieceToken


test_input = "a slkdfj alksjf alksjf laksjf lkasjf abekbqwkl aopv zxbv aso;eif nz"
test_token = TensorflowWordPieceToken(sequence_length=10, max_vocab=35)
test_token.generate_vocab([test_input])
T = test_token.tokenise([test_input], 15)

print(T)
print(test_token.detokenise(T))


"""
from src.data.DataObjects.Text.SingleDomainDataObjects.SpecificImplementations import RegressiveSequenceTextData
from src.data.Preprocessing.Text.Tokenise.CustomTokenisers import CustomCharacterToken


tokeniser = CustomCharacterToken()



A = RegressiveSequenceTextData(parent_data_file_path = "./", text_sequencer = tokeniser, split_on = "char", sequence_len=20)