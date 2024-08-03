streamlit as st
from transformer import DistilBertTokenizer, DistinBertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-spanish-wwm-cased")
model = BertForQuestionAnswering.from_pretrained("bert-base-spanish-wwm-cased")

st.title("Sistema de preguntas con BETO")

question = st.text_area("Introduce una pregunta")
context = st.text_area("Introduce un contexto")

        with torch.no_grad():
             outputs = model(**inputs)
             
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        answer_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        answer = tokenizer.convert_tokens_to.string(answer_tokens)
        
        st.write(f"Respuesta: {answer}"))
        
