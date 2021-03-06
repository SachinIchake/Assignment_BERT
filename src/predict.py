import config
import torch
import time
from model import BERTBaseUncased
import functools
import torch.nn as nn
import joblib
import pandas as pd 


MODEL = None
DEVICE = "cpu"
PREDICTION_DICT = dict()
memory = joblib.Memory("../input/", verbose=0)


# def predict_from_cache(text):
#     if text in PREDICTION_DICT:
#         return PREDICTION_DICT[text]
#     else:
#         result = text_prediction(text)
#         PREDICTION_DICT[text] = result
#         return result


@memory.cache
def text_prediction(text):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    text = str(text)
    text = " ".join(text.split())

    inputs = tokenizer.encode_plus(
        text, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    # ids = ids.to(DEVICE, dtype=torch.long)
    # token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    # mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


def predict():
    df_test = pd.read_csv(config.TESTING_FILE).fillna("none").reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
  

    # text = df_test["text"]
    # start_time = time.time()
    # df_test_new = df_test[]
    predict_dict = {}
    for ind in df_test.index: 
        id = df_test['id'][ind]
        text = df_test['text'][ind]
        
        positive_prediction = text_prediction(text)
        negative_prediction = 1 - positive_prediction
        if positive_prediction > negative_prediction:
            print('1')
            result = 1 
        else:
            print('0')
            result = 0 

        # print(id,' = ',positive_prediction)    
        predict_dict.update([('id',id), ('result',result), ('outputlabel',positive_prediction) , ('negative', negative_prediction)]) 
    
   
         
    submission = pd.DataFrame.from_dict(predict_dict)
    
    submission.to_csv('submission.csv', index=False)
    
   
   
    # negative_prediction = 1 - positive_prediction
    # response = {}
    # response["response"] = {
    #     "positive": str(positive_prediction),
    #     "negative": str(negative_prediction),
    #     "text": str(text),
    #     "time_taken": str(time.time() - start_time),
    # }
    # return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH,map_location=torch.device('cpu')))
    MODEL.eval()
    predict()
