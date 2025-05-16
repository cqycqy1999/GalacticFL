from transformers import AutoTokenizer, AutoModelForCausalLM
# BloomModel
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
inputs = tokenizer("Hello, my next paper TCAD must can accept!", return_tensors="pt")
outputs = model(**inputs)
# original_output = model.predict(inputs)
# print(outputs)
# with open('output1.txt', 'w') as f:
#     f.write(str(outputs))