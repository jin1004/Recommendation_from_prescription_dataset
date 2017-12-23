FROM python:3.6

COPY Recommendation.py model1.out le_sex.out enc_sex.out vectorizer_symptoms.out vectorizer_diagnoses.out le_medicines.out input_scaler.out requirements.txt /demo/

WORKDIR /demo

RUN pip3 install -r /demo/requirements.txt

CMD [ "python", "Recommendation.py" ]