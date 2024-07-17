from transformers import BartForConditionalGeneration, BartTokenizer 

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", 
													forced_bos_token_id=0) 
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large") 


sent = "Machine learning is a field of <mask> that uses <mask> to give computers, the ability to <mask> without being <mask> explicitly."


tokenized_sent = tokenizer(sent, return_tensors='pt') 

generated_encoded = bart_model.generate(tokenized_sent['input_ids'], max_length=100) 

print(tokenizer.batch_decode(generated_encoded, skip_special_tokens=True)[0]) 

