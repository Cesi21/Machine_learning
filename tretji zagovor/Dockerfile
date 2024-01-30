# Uporabimo osnovno sliko Python
FROM python:3.9-slim

# Nastavimo delovni direktorij v Docker kontejnerju
WORKDIR /app

# Kopiramo zahtevane datoteke v kontejner
COPY requirements.txt requirements.txt
COPY random_forest_model.pkl random_forest_model.pkl
COPY best_model_GRU.h5 best_model_GRU.h5
COPY best_cnn_model.h5 best_cnn_model.h5
COPY scaler4.pkl scaler4.pkl
COPY target_scaler.pkl target_scaler.pkl
COPY scaler6.pkl scaler6.pkl
COPY naloga_7.py naloga_7.py

# Namestimo potrebne Python knjižnice
RUN pip install -r requirements.txt

# Izpostavimo vrata, ki jih uporablja Flask
EXPOSE 5000

# Nastavimo spremenljivko okolja FLASK_APP
ENV FLASK_APP=naloga_7.py

# Zagon Flask aplikacije
CMD ["flask", "run", "--host=0.0.0.0"]
