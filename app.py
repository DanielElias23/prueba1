import streamlit as st
from transformers import AutoModel, AutoTokenizer   #BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

st.title("Sistema de preguntas con BETO")

question = st.text_area("Introduce una pregunta")
#context = st.text_area("Introduce un contexto")

if st.button("Responder"):
            input_ids = tokenizer("Hello, my dog is cuteHello world!", return_tensors="pt")
            outputs = model(input_ids, labels=input_ids)
            #loss, prediction_scores = outputs[:2]
            st.write(f"{outputs}")
            #st.write(f"{loss} y {prediction_scores}")

       
"""inputs = tokenizer("Hola")
        #outputs = model(**inputs)
        #st.write(outputs) 
        with torch.no_grad():
             outputs = model(**inputs)
             
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        answer = tokenizer.convert_tokens_to.string(answer_tokens)
        
        st.write(f"Respuesta: {answer}")
"""
