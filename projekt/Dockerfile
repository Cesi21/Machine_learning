FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
COPY lstm_model.h5 lstm_model.h5
COPY scaler_pm10.pkl scaler_pm10.pkl
COPY projekt_api.py projekt_api.py
RUN pip install -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=projekt_api.py
CMD ["flask", "run", "--host=0.0.0.0"]
