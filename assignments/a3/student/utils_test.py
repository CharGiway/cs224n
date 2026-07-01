from utils import pad_sents

def test_pad_sents():
    print(pad_sents([['cc','cc'],['aa','aa','aa'],['b']],'<EOS>'))
    
test_pad_sents()