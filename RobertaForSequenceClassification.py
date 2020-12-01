# Imports
from .helper import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import time
import datetime
import random
import torch
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():    # If there's a GPU available...
    device = torch.device("cuda") # Tell PyTorch to use the GPU.
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else: # If not...
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class RoBERTa():

    def __init__(self, sentences, NUM_CLASS, seed_val=42, random_state=2018, evaluate_score=flat_accuracy, domain='roberta-base'):

      self.seed_val = seed_val
      self.random_state = random_state
      self.evaluate_score = evaluate_score
      self.NUM_CLASS = NUM_CLASS
      self.domain = domain

      print('Loading RoBERT tokenizer...') # Load the BERT tokenizer.
      tokenizer = RobertaTokenizer.from_pretrained(self.domain, do_lower_case=True)

      print('Set max_length as: ', min(512, np.max(np.array([len(tokenizer.encode(i, add_special_tokens=True)) for i in sentences]))) )

    def fit(self, sentences, labels, model_save_path, device, do_lower_case=True, debug=False, max_length=512, add_special_tokens=True, test_size=0.1, batch_size=8, output_attentions=True, output_hidden_states=True, epochs=2):
      '''
      - sentences : input string (as numpy array)
      - labels : numerical label (as numpy array)
        labels should be 0,1,2 like that
        Example on how you can obtain labels = train_data['labels'].values
      - test_size: is validation size (train and validation split basically)
      '''

      print('Loading RoBERTa tokenizer...') # Load the BERT tokenizer.
      tokenizer = RobertaTokenizer.from_pretrained(self.domain, do_lower_case=do_lower_case)

      if debug:
        # Print the original sentence.
        print(' Original: ', sentences[0])

        # Print the sentence split into tokens.
        print('Tokenized: ', tokenizer.tokenize(sentences[0]))

        # Print the sentence mapped to token ids.
        print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

      
      input_ids, attention_masks = get_inputid_attentionmasks(tokenizer, sentences, debug=False, max_length=max_length, add_special_tokens=True)

      # Use 90% for training and 10% for validation.
      train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=self.random_state, test_size=test_size)
      # Do the same for the masks
      train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,random_state=self.random_state, test_size=test_size)

      train_data, train_sampler, train_dataloader = create_data(train_inputs, train_masks, train_labels, batch_size)
      validation_data, validation_sampler, validation_dataloader = create_data(validation_inputs, validation_masks, validation_labels,batch_size)


      # Load RobertaForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
      model = RobertaForSequenceClassification.from_pretrained(self.domain, 
        num_labels = self.NUM_CLASS, 
        output_attentions=output_attentions, 
        output_hidden_states=output_hidden_states)

      model.cuda() # Tell pytorch to run this model on the GPU.

      # Total number of training steps is number of batches * number of epochs.
      total_steps = len(train_dataloader) * epochs
      # Note: AdamW is a class from the huggingface library (as opposed to pytorch) I believe the 'W' stands for 'Weight Decay fix"
      optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8 )
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

      print("")
      print('Training Batches: ',len(train_dataloader))
      print('Validation Batches: ',len(validation_dataloader))
      print('Batch Size: ',batch_size)
      print('Epochs: ',epochs)

      random.seed(self.seed_val)
      np.random.seed(self.seed_val)
      torch.manual_seed(self.seed_val)
      torch.cuda.manual_seed_all(self.seed_val)

      # Store the average loss after each epoch so we can plot them.
      loss_values = []
      logits_list, label_ids_list = [], []

      model = train_val_loop(model, epochs, train_dataloader, optimizer, scheduler, validation_dataloader, model_save_path, custom_name = '_')

      return model, tokenizer, device, max_length


      # Function for getting prediction and prediction probability
    def predict(self, model, example_test_sentence, tokenizer, device, max_length):
        '''
        Arg: 
            model: pre-trained model
            example_test_sentence: input sentence
        
        Returns:
            prediciton: 0/1 prediction corresponding to the sentence
            prediction_probabaility: corresponding to the prediction
        '''
        encoded_sent = [tokenizer.encode(example_test_sentence, max_length = max_length, add_special_tokens = True)]
        input_ids = pad_sequences(encoded_sent, maxlen=max_length, dtype="long", value=0, truncating="post", padding="post")
        att_mask = [[int(token_id > 0) for token_id in input_ids[0]]]
        input_ids = torch.tensor(input_ids).to(device)
        att_mask = torch.tensor(att_mask).to(device)

        outputs = model(input_ids, token_type_ids=None, attention_mask=att_mask)
        
        logits = outputs[0]
        prediction = logits.detach().cpu().numpy().argmax()
        prediction_probability = torch.softmax(logits, axis=1).detach().cpu().numpy().max()

        return prediction, prediction_probability

    
    def fill_output(self, model, example_test_sentence, tokenizer, device, max_length):
        prediction, prediction_probability = self.predict(model, example_test_sentence, tokenizer, device, max_length)
        attention, tokens = self.get_attention_tokens(model, example_test_sentence, tokenizer, device, max_length)
        _, weights = self.get_average_attention_weights(attention, tokens)  # These are normalized weights, if you want absolute, take the first argument
        
        return prediction, prediction_probability, weights.values

	# -----------------------------------------------------------------
	# Functions for getting attention tokens
    def get_attention_tokens(self, model, example_test_sentence, tokenizer, device, max_length):
        '''
        Arg:
            model: trained model
            example_test_sentence: example sentence, this runs sentence by sentence
            max_length: max_length criteria used while training the model
            
        Returns:
            attention: a tuple consisting of 12 or 24 attention layers; each layers has 12 or 16 attention heads
            tokens: correspoding to the input sentence
        '''
        
        inputs = tokenizer.encode_plus(example_test_sentence, max_length=max_length, add_special_tokens=True,return_tensors='pt')
        inputs.to(device)
        
        token_type_ids = inputs['token_type_ids']
        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']
        
        input_id_list = input_ids[0].tolist() # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)
        
        # BERT has multiple layer of attention (12 in BERT Base and 24 in BERT Large)
        # Every layer incorporates multiple attention heads (12 or 16)
        # Since model weights are not shared between layers; Hence every attention layer will give different attention to same sentence
        # BERT LArge can have upto 24*16 =384 attention mechanism

        attention = model(input_ids, token_type_ids = token_type_ids)[-1]
        # attention is a tuple
        return attention, tokens



    def get_attention_weights(self, attention,LAYER, HEAD, TOKEN_INDEX, average_heads=True, Normalize=True, Normalize_type='min-max',EXCLUDE_SPECIAL_TOKENS=True):    
        ''' If average_heads=True, it doesn't matter which HEAD you are giving
        '''    
        # 1. Taking Average of everything
        no_heads = attention[0].size()[1]
        if average_heads:
            attention_outputs = torch.sum(attention[LAYER][0][:, TOKEN_INDEX, :], dim=0)/no_heads
        if average_heads==False:
            attention_outputs = attention[LAYER][0][HEAD, TOKEN_INDEX, :].size()
        
        if Normalize:
            if average_heads:
                if EXCLUDE_SPECIAL_TOKENS:
                    attention_normalize_outputs = attention_outputs[1:-1]
                else:
                    attention_normalize_outputs = attention_outputs
                # -------------------------------------------------------
                if Normalize_type=='min-max':
                    max_weight, min_weight = attention_normalize_outputs.max(), attention_normalize_outputs.min()
                    attention_normalize_outputs = (attention_normalize_outputs - min_weight) / (max_weight - min_weight)

                elif Normalize_type=='normal':
                    mu, std = attention_normalize_outputs.mean(), attention_normalize_outputs.std()
                    attention_normalize_outputs = (attention_normalize_outputs - mu) / std
        
        # Note length of attention_outputs is 30 (including special tokens) while length of attention_normalize_outputs is 28 (no special tokens)
        return attention_outputs, attention_normalize_outputs, LAYER


    def categorizing_tokens(self, attention_outputs, attention_normalize_outputs, tokens, LAYER):
        
        tokens_updated = tokens[1:-1] # Removing Begining and End tokens
        
        attention_outputs_updated = attention_outputs[1:-1]
        
        df_attention = pd.DataFrame(data = np.random.rand(len(tokens_updated)))
        df_attention_normalized = pd.DataFrame(data = np.random.rand(len(tokens_updated)))

        attention_outputs_updated =  attention_outputs_updated.cpu().tolist()
        attention_normalize_outputs = attention_normalize_outputs.cpu().tolist()

        df_attention['tokens'] = tokens_updated
        df_attention['attention_' + str(LAYER)] = attention_outputs_updated
        
        df_attention_normalized['tokens'] = tokens_updated
        df_attention_normalized['attention_normalized_' + str(LAYER)] = attention_normalize_outputs
        
        del df_attention[0]
        del df_attention_normalized[0]
        
        return df_attention, df_attention_normalized

    def get_average_attention_weights(self, attention, tokens):
        
        '''
        Arg:
            attention: the attention weights calculated from the model
            tokens: corresponding to the input senetence
        
        Returns:
            attention_weights_df: the average attention weights over all the layers; For each layer, average of all the heads
            is taken into consideration, taken into account by average_heads parameter; token_index =0 is used for CLS tag
            
            attention_weights_normalized_df: normalized version of above
        '''

        N_LAYERS = len(attention)
        N_HEADS = attention[0].size()[1]
        N_TOKENS = attention[0].size()[2]

        df_attention_list = []
        df_attention_normalized_list = []

        for i in range(N_LAYERS):
            attention_outputs, attention_normalize_outputs, LAYER = self.get_attention_weights(attention, 
                                                                              LAYER=i, 
                                                                              HEAD=11, 
                                                                              TOKEN_INDEX=0)
            if i==0:
                df, dd = self.categorizing_tokens(attention_outputs, attention_normalize_outputs, tokens, LAYER)
                df_attention_list.append(df)
                df_attention_normalized_list.append(dd)
            else:
                df, dd = self.categorizing_tokens(attention_outputs, attention_normalize_outputs, tokens, LAYER)
                del df['tokens']
                del dd['tokens']
                df_attention_list.append(df)
                df_attention_normalized_list.append(dd)

        attention_weights_df = pd.concat(df_attention_list, axis=1)
        attention_weights_normalized_df = pd.concat(df_attention_normalized_list, axis=1)
        
        attention_weights_df['attention_AVERAGE'] = attention_weights_df.mean(axis=1)
        attention_weights_normalized_df['attention_normalized_AVERAGE'] = attention_weights_normalized_df.mean(axis=1)
        
        attention_weights_df = attention_weights_df[['tokens', 'attention_AVERAGE']]
        attention_weights_normalized_df = attention_weights_normalized_df[['tokens', 'attention_normalized_AVERAGE']]
        
        return attention_weights_df, attention_weights_normalized_df




