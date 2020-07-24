import transformers


MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
EPOCHS = 1
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH =  "../input/model/modelbertv1.pt"
TRAINING_FILE = "../input/data/train.csv"
TESTING_FILE = "../input/data/test.csv" 
SUBMIT_FILE = "../input/data/submit.csv" 
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
