# Imports
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

def flat_accuracy(preds, labels):  # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def f1_score_2(preds, labels):  # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # return f1_score(labels_flat, pred_flat)
    return f1_score(labels_flat, pred_flat, average='weighted')


class RoBERTa():

    def __init__(self, sentences, NUM_CLASS, seed_val=42, random_state=2018, evaluate_score=flat_accuracy):

      self.seed_val = seed_val
      self.random_state = random_state
      self.evaluate_score = evaluate_score
      self.NUM_CLASS = NUM_CLASS

      print('Loading RoBERT tokenizer...') # Load the BERT tokenizer.
      tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

      print('Set max_length as: ', min(512, np.max(np.array([len(tokenizer.encode(i, add_special_tokens=True)) for i in sentences]))) )


    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))  # Round to the nearest second.
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def get_inputid_attentionmasks(self, tokenizer, sentences, max_length, debug=False, add_special_tokens=True):
	    '''
	    add_special_tokens = Add '[CLS]' and '[SEP]'
	    max_length = maximum length of sentence
	    '''
	    # Tokenize all of the sentences and map the tokens to thier word IDs.
	    input_ids = []
	    for sent in sentences:
	        encoded_sent = tokenizer.encode(sent, max_length=max_length, add_special_tokens=add_special_tokens)
	        input_ids.append(encoded_sent)
	        # `encode` will: Tokenize the sentence --> Prepend the `[CLS]` token to the start 
	        # --> Append the `[SEP]` token to the end --> Map tokens to their IDs.
	        # Truncate all sentences.
	        # This function also supports truncation and conversion to pytorch tensors, but we need to do padding, so we can't use these features :(
	        # return_tensors = 'pt', # Return pytorch tensors.
	                  
	    if (debug):
	        print('Original: ', sentences[0])
	        print('Token IDs:', input_ids[0])
	        print('Max sentence length: ', max([len(sen) for sen in input_ids]))
	        print('\nPadding/truncating all sentences to %d values...' % max_length)
	        print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

	    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", 
	                              value=0, truncating="post", padding="post")  # "post" indicates that we want to pad and truncate at the end of the sequence, as opposed to the beginning.
	    
	    attention_masks = [] # Create attention masks
	    for sent in input_ids: # For each sentence, Create the attention mask.
	        #  If a token ID is 0, then it's padding, set the mask to 0.
	        #  If a token ID is > 0, then it's a real token, set the mask to 1.
	        att_mask = [int(token_id > 0) for token_id in sent]
	        attention_masks.append(att_mask) # Store the attention mask for this sentence.
	    return input_ids, attention_masks


    def create_data(self, inputs_ids, attention_masks, labels, batch_size):
	    inputs_ids = torch.tensor(inputs_ids)
	    attention_masks = torch.tensor(attention_masks)
	    labels = torch.tensor(labels)
	    
	    data = TensorDataset(inputs_ids, attention_masks, labels)
	    sampler = RandomSampler(data)
	    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	    return data, sampler, dataloader

    def train_val_loop(self, model, epochs, train_dataloader, optimizer, scheduler, validation_dataloader, model_save_path, custom_name = '_'):

      # For each epoch...
      train_loss=[]
      val_loss=[]
      train_score=[]
      val_score=[]

      for epoch_i in range(0, epochs):
          #               Training
          # Perform one full pass over the training set.
          print("")
          print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
          print('Training...')
          t0 = time.time() # Measure how long the training epoch takes.
          total_loss, total_score = 0, 0 # Reset the total loss for this epoch.

          model.train()

          for step, batch in enumerate(train_dataloader): # For each batch of training data...
              if step % 200 == 0 and not step == 0:  # Progress update every 40 batches.
                  elapsed = self.format_time(time.time() - t0) # Calculate elapsed time in minutes.
                  
                  # Report progress.
                  print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
              
              b_input_ids = batch[0].to(device)
              b_input_mask = batch[1].to(device)
              b_labels = batch[2].to(device)
              model.zero_grad()
              
              outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
              loss = outputs[0]
              total_loss += loss.item()
              # print(outputs[1])

              total_score += self.evaluate_score(outputs[1].detach().cpu().numpy(), b_labels.to('cpu').numpy())

              loss.backward() # Perform a backward pass to calculate the gradients.
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.

              optimizer.step()
              scheduler.step() # Update the learning rate.
          
          train_loss.append(total_loss / len(train_dataloader)) # Store the loss value for plotting the learning curve.

          print("")
          print("  Train loss: {0:.2f}".format(total_loss / len(train_dataloader)))
          print("  Train score: {0:.2f}".format(total_score / len(train_dataloader)))
          print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))
              
          #               Validation

          print("")
          print("Running Validation...")

          t0 = time.time()
          model.eval()

          eval_loss, eval_accuracy = 0, 0
          nb_eval_steps, nb_eval_examples = 0, 0

          for batch in validation_dataloader:
              batch = tuple(t.to(device) for t in batch) # Add batch to GPU
              b_input_ids, b_input_mask, b_labels = batch # Unpack the inputs from our dataloader

              with torch.no_grad():

                  outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

              # print(len(outputs))
              # print(outputs[0])
              # print(outputs[1])
              logits = outputs[1] # Get the "logits" output by the model. The "logits" are the output values prior to applying an activation function like the softmax.
              eval_loss += outputs[0].item()
              logits = logits.detach().cpu().numpy() # Move logits and labels to CPU
              label_ids = b_labels.to('cpu').numpy()
              # print(logits)
              # print(label_ids)

              # tmp_eval_accuracy = flat_accuracy(logits, label_ids) # Calculate the accuracy for this batch of test sentences.
              # tmp_eval_accuracy = f1_score_2(logits, label_ids)
              tmp_eval_accuracy = self.evaluate_score(logits, label_ids)
              eval_accuracy += tmp_eval_accuracy
              nb_eval_steps += 1 

          
          print("  Val Loss: {0:.2f}".format(eval_loss/nb_eval_steps))
          print("  Val Score: {0:.2f}".format(eval_accuracy/nb_eval_steps))
          print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

          filename = custom_name + 'RoBERTa_epoch={0}_trloss={1:.2f}_trscore={2:.2f}_valloss={3:.2f}_valscore={4:.2f}_.pkl'.format(str(epoch_i), total_loss/len(train_dataloader),total_score/len(train_dataloader), eval_loss/nb_eval_steps, eval_accuracy/nb_eval_steps)
          model_path = os.path.join(model_save_path, filename)
          torch.save(model, model_path)
          print("")
          print("Model Saved for epoch {}!".format(epoch_i+1))

      print("")
      print("Training complete!")


      return model


    def fit(self, sentences, labels, model_save_path, device, do_lower_case=True, debug=False, max_length=512, add_special_tokens=True, test_size=0.1, batch_size=8, output_attentions=True, output_hidden_states=True, epochs=2):
      '''
      - sentences : input string (as numpy array)
      - labels : numerical label (as numpy array)
        labels should be 0,1,2 like that
        Example on how you can obtain labels = train_data['labels'].values
      - test_size: is validation size (train and validation split basically)
      '''

      print('Loading RoBERTa tokenizer...') # Load the BERT tokenizer.
      tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=do_lower_case)

      if debug:
        # Print the original sentence.
        print(' Original: ', sentences[0])

        # Print the sentence split into tokens.
        print('Tokenized: ', tokenizer.tokenize(sentences[0]))

        # Print the sentence mapped to token ids.
        print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

      
      input_ids, attention_masks = self.get_inputid_attentionmasks(tokenizer, sentences, debug=False, max_length=max_length, add_special_tokens=True)

      # Use 90% for training and 10% for validation.
      train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=self.random_state, test_size=test_size)
      # Do the same for the masks
      train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,random_state=self.random_state, test_size=test_size)

      train_data, train_sampler, train_dataloader = self.create_data(train_inputs, train_masks, train_labels, batch_size)
      validation_data, validation_sampler, validation_dataloader = self.create_data(validation_inputs, validation_masks, validation_labels,batch_size)


      # Load RobertaForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
      model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
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

      model = self.train_val_loop(model, epochs, train_dataloader, optimizer, scheduler, validation_dataloader, model_save_path, custom_name = '_')

      return model, tokenizer, device, max_length


    def fill_output(self, model, example_test_sentence, tokenizer, device, max_length):
        prediction, prediction_probability = self.predict(model, example_test_sentence, tokenizer, device, max_length)
        attention, tokens = self.get_attention_tokens(model, example_test_sentence, tokenizer, device, max_length)
        _, weights = self.get_average_attention_weights(attention, tokens)  # These are normalized weights, if you want absolute, take the first argument
        
        return prediction, prediction_probability, weights.values



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




