from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import time
import datetime
import random
import torch
import os
import pandas as pd

def flat_accuracy(preds, labels):  # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def f1_score_2(preds, labels):  # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # return f1_score(labels_flat, pred_flat)
    return f1_score(labels_flat, pred_flat, average='weighted')

def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))  # Round to the nearest second.
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

def get_inputid_attentionmasks(tokenizer, sentences, max_length, debug=False, add_special_tokens=True):
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


def create_data(inputs_ids, attention_masks, labels, batch_size):
	    inputs_ids = torch.tensor(inputs_ids)
	    attention_masks = torch.tensor(attention_masks)
	    labels = torch.tensor(labels)
	    
	    data = TensorDataset(inputs_ids, attention_masks, labels)
	    sampler = RandomSampler(data)
	    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
	    return data, sampler, dataloader


def train_val_loop(model, epochs, train_dataloader, optimizer, scheduler, validation_dataloader, model_save_path, evaluate_score, custom_name = '_'):

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
                  elapsed = format_time(time.time() - t0) # Calculate elapsed time in minutes.
                  
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

              total_score += evaluate_score(outputs[1].detach().cpu().numpy(), b_labels.to('cpu').numpy())

              loss.backward() # Perform a backward pass to calculate the gradients.
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.

              optimizer.step()
              scheduler.step() # Update the learning rate.
          
          train_loss.append(total_loss / len(train_dataloader)) # Store the loss value for plotting the learning curve.

          print("")
          print("  Train loss: {0:.2f}".format(total_loss / len(train_dataloader)))
          print("  Train score: {0:.2f}".format(total_score / len(train_dataloader)))
          print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
              
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
              tmp_eval_accuracy = evaluate_score(logits, label_ids)
              eval_accuracy += tmp_eval_accuracy
              nb_eval_steps += 1 

          
          print("  Val Loss: {0:.2f}".format(eval_loss/nb_eval_steps))
          print("  Val Score: {0:.2f}".format(eval_accuracy/nb_eval_steps))
          print("  Validation took: {:}".format(format_time(time.time() - t0)))

          filename = custom_name + 'BERT_epoch={0}_trloss={1:.2f}_trscore={2:.2f}_valloss={3:.2f}_valscore={4:.2f}_.pkl'.format(str(epoch_i), total_loss/len(train_dataloader),total_score/len(train_dataloader), eval_loss/nb_eval_steps, eval_accuracy/nb_eval_steps)
          model_path = os.path.join(model_save_path, filename)
          torch.save(model, model_path)
          print("")
          print("Model Saved for epoch {}!".format(epoch_i+1))

      print("")
      print("Training complete!")


      return model




