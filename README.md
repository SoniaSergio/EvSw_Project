# ECG Arrhythmia Classifier

![Deploy](https://img.shields.io/badge/Deploy-ecg.heremy.link-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)

> **Classificazione Automatica delle Aritmie ECG**
> *Sistema diagnostico comparativo CNN 1D vs Random Forest su segnali elettrocardiografici, evoluzione dell'interfaccia Gradio sviluppata per l'esame di Sistemi Multimediali / Evoluzione del Software — Università degli Studi di Bari "Aldo Moro", A.A. 2025/2026.*

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![MongoDB](https://img.shields.io/badge/MongoDB-7.0-brightgreen) ![Docker](https://img.shields.io/badge/Docker-Compose-blue) ![Security](https://img.shields.io/badge/Security-AES--256%20ALDE-red) ![HTTPS](https://img.shields.io/badge/HTTPS-Let's_Encrypt-yellow)

---

## Indice

- [1. Visione del Progetto e Origine](#1-visione-del-progetto-e-origine)
- [2. Evoluzione Architetturale: da Gradio a Microservizi](#2-evoluzione-architetturale-da-gradio-a-microservizi)
- [3. Architettura del Sistema e Sicurezza (ALDE)](#3-architettura-del-sistema-e-sicurezza-alde)
- [4. Stack Tecnologico](#4-stack-tecnologico)
- [5. Struttura del Progetto](#5-struttura-del-progetto)
- [6. I Modelli di Classificazione](#6-i-modelli-di-classificazione)
- [7. API Reference](#7-api-reference)
- [8. Frontend: Interfaccia Web](#8-frontend-interfaccia-web)
- [9. Guida all'Installazione](#9-guida-allinstallazione)
- [10. Variabili d'Ambiente](#10-variabili-dambiente)
- [11. Verifica Sicurezza Database (Cifratura ALDE)](#11-verifica-sicurezza-database-cifratura-alde)
- [12. Risoluzione Problemi (Troubleshooting)](#12-risoluzione-problemi-troubleshooting)
- [13. Riferimenti Scientifici](#13-riferimenti-scientifici)

---

## 1. Visione del Progetto e Origine

**ECG Arrhythmia Classifier** è un sistema web di supporto diagnostico per la classificazione automatica delle aritmie cardiache a partire da segnali ECG del dataset standard **MIT-BIH Arrhythmia Database** (87.554 campioni, 5 classi, 360 Hz).

Dal punto di vista dell'Ingegneria del Software, questo sistema è concepito non solo come un progetto isolato, ma come un vero e proprio prodotto software: un sistema generico nato per cogliere un'opportunità di business (il supporto clinico rapido) in grado di fornire funzionalità utili a una vasta gamma di utenti medici, garantendo scalabilità e manutenibilità a lungo termine.

Il progetto mette a confronto due paradigmi opposti di machine learning:

- **Random Forest (RF):** approccio classico basato su feature engineering manuale (9 descrittori morfologici). Offre trasparenza decisionale tramite feature importance, ma soffre di limitata separabilità sulle classi minoritarie patologiche.
- **CNN 1D:** architettura deep learning end-to-end su 3 blocchi convoluzionali (~204K parametri). Apprende autonomamente le gerarchie di pattern direttamente dal segnale grezzo, abbattendo drasticamente i falsi negativi sulle aritmie critiche.

La metrica primaria adottata è la **macro Recall**, clinicamente più rilevante dell'accuracy globale in presenza di forte sbilanciamento.

| Modello | Accuracy | Macro Recall | Macro F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 0.83 | 0.69 | 0.57 | 0.903 |
| CNN 1D | 0.96 | 0.92 | 0.82 | 0.989 |

> La CNN 1D ottiene risultati superiori in tutte le metriche.

La CNN 1D riduce dell'**83% i falsi negativi sulla classe Ventricolare** rispetto alla RF (Recall: 0.60 → 0.91), la classe a maggiore rischio clinico per il paziente.

---

## 2. Evoluzione Architetturale: da Gradio a Microservizi

Il progetto nasce come prototipo monolitico in cui i modelli erano incapsulati in un'interfaccia interattiva **Gradio** eseguita localmente su Google Colab.

Questa versione rappresenta la sua **evoluzione in un sistema distribuito**, riprogettato secondo la definizione formale **IEEE di Architettura Software**: *"L'organizzazione fondamentale di un sistema sw che si concretizza nei suoi componenti, nelle loro relazioni reciproche e con l'ambiente e nei principi che ne guidano la progettazione e l'evoluzione"*.

Si è passati da un approccio monolitico a un **approccio orientato ai servizi (Service-Oriented Architecture)**, particolarmente adatto per il software basato su cloud, in cui il sistema è stato scomposto in servizi a grana fine, isolati e resilienti ai guasti:

| Aspetto | Versione Gradio iniziale | Versione Attuale (Microservizi) |
|:---|:---|:---|
| **Deployment** | Google Colab, locale | Docker Compose, server remoto con HTTPS |
| **Interfaccia** | Gradio auto-generata | Frontend custom HTML/CSS/JS (Nginx) |
| **Persistenza** | Nessuna | MongoDB: storico completo e cifrato |
| **Architettura** | Monolitica, single-process | 4 servizi indipendenti dockerizzati |
| **Inferenza** | Sincrona, single-thread | FastAPI async, latenza media < 200 ms |
| **Scalabilita** | Non scalabile | Servizi indipendentemente scalabili al variare del carico |
| **Sicurezza** | Nessuna | HTTPS/TLS 1.3 + **Cifratura AES-256 (ALDE)** nel DB |
| **Input** | Manuale o da file locale | Manuale, CSV upload, segnale casuale da dataset MIT-BIH |

---

## 3. Architettura del Sistema e Sicurezza (ALDE)

Il sistema è composto da **5 servizi Docker** comunicanti su una rete bridge dedicata (`ecg-net`), isolata dalla rete host. I container isolano l'applicazione nello spazio utente sfruttando i meccanismi del kernel Linux (`namespaces` e `cgroups`).


```mermaid
graph TD
    User((Utente)) --> Browser[Browser Client]
    
    subgraph "Docker Network ecg-net"
        Browser --> Nginx[Nginx Reverse Proxy]
        Nginx --> Pred[Prediction Service - FastAPI]
        Nginx --> Hist[History Service - FastAPI]
        
        Pred --> Security[ALDE Encryption Layer]
        Security --> Mongo[(MongoDB)]
        Hist --> Security
    end

    style Security fill:#f9f,stroke:#333,stroke-width:2px
    style Mongo fill:#e1f5fe,stroke:#01579b
  
  ```

Per garantire la confidenzialità dei dati medici, l'architettura implementa il pattern **Application-Layer Data Encryption (ALDE)**. Nessun dato clinico sensibile risiede in chiaro nel database: il Service Layer cifra i dati **prima** della scrittura e li decifra **dopo** la lettura, esclusivamente in memoria RAM, utilizzando l'algoritmo **AES-256 in modalità CBC/HMAC (Fernet)**. Questo garantisce il principio di *Encryption at Rest*: anche in caso di compromissione diretta del database, i dati risultano illeggibili senza la chiave.


### Flusso di una predizione

1. Il browser invia `POST /api/predict/` con il segnale (array di 187 float).
2. Nginx fa reverse proxy verso `prediction-service:8000/predict/`.
3. Il prediction service esegue l'inferenza con CNN 1D e successivamente con RF, restituendo per ciascun modello la diagnosi, la confidenza e la distribuzione di probabilità sulle 5 classi.
4. Il segnale grezzo e i risultati diagnostici vengono cifrati con Fernet prima di essere scritti su MongoDB.
5. La risposta API — con i dati in chiaro — viene restituita immediatamente al frontend per la visualizzazione.
6. Nelle letture successive tramite lo storico, l'history-service recupera i documenti cifrati e li decifra in RAM prima di inviarli alla UI; il segnale grezzo è escluso dalla lista paginata e incluso solo nella lettura per ID.

## 4. Stack Tecnologico

| Layer | Tecnologia | Versione | Ruolo |
|:---|:---|:---|:---|
| **Frontend** | HTML5 / CSS3 / Vanilla JS | — | Interfaccia utente |
| **Frontend Server** | Nginx Alpine | 1.27 | Serve statico + reverse proxy |
| **Prediction API** | FastAPI + Uvicorn | 0.111 / 0.30 | Inferenza CNN e RF |
| **History API** | FastAPI + Uvicorn | 0.115 / 0.32 | CRUD storico predizioni + campioni casuali |
| **Deep Learning** | TensorFlow / Keras | 2.16.1 | Modello CNN 1D |
| **Machine Learning** | Scikit-learn + Joblib | 1.5.0 | Modello Random Forest |
| **Feature Engineering** | Pandas + NumPy | 2.2.2 / 1.26.4 | Estrazione descrittori morfologici |
| **Sicurezza / Crittografia** | Cryptography (Fernet) | 42.0.5 | Cifratura simmetrica ALDE AES-256 |
| **Database** | MongoDB | 7.0 | Persistenza predizioni + campioni MIT-BIH |
| **ODM Asincrono** | Motor + PyMongo | 3.6.0 / 4.9.2 | Driver async MongoDB per FastAPI |
| **Validazione** | Pydantic | 2.x | Validazione input/output API |
| **Containerization** | Docker + Docker Compose | — | Orchestrazione servizi |
| **TLS/HTTPS** | Let's Encrypt + Certbot | — | Certificati SSL con rinnovo automatico |

---

## 5. Struttura del Progetto

```text
ECG-ARRHYTHMIA-/
├── frontend/
│   ├── src/
│   │   ├── index.html          # SPA principale
│   │   ├── app.js              # Logica UI: input, predizione, storico
│   │   └── style.css           # Design system (CSS variables, componenti)
│   ├── Dockerfile              # Build Nginx Alpine + copia statici
│   └── nginx.conf              # Reverse proxy, HTTPS, redirect HTTP→HTTPS
│
├── prediction-service/
│   ├── models/                 # NON incluso nel repo su GitHub(vedere sezione 9.2)
│   │   ├── ecg_cnn_model.h5    # Pesi CNN 1D addestrata (TensorFlow/Keras)
│   │   └── ecg_rf_model.pkl    # Modello Random Forest serializzato (joblib)
│   ├── routers/
│   │   └── predict.py          # POST /predict/ — orchestrazione inferenza
│   ├── services/
│   │   ├── cnn_service.py
│   │   ├── rf_service.py
│   │   └── db_service.py       # Cifratura Fernet e save_prediction()
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── history-service/
│   ├── routers/
│   │   └── history.py          # GET /history/, /history/random, /history/{id}
│   ├── services/
│   │   └── db_service.py       # Decifratura on-the-fly, get_predictions(), get_random_ecg_sample()
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── seed/
│   ├── Dockerfile              # Immagine Python con pandas, pymongo, cryptography
│   ├── seed_db.py              # Script di popolamento collection ecg_samples
│   └── mitbih_test.csv         # NON incluso nel repo (vedere sezione 9.3)
│
├── mongo/
│   └── init/
│       └── init.js             # Crea collection predictions + indici MongoDB
│
├── docker-compose.yml          # Orchestrazione: mongo, prediction, history, frontend, certbot, seed
├── .env                        # Variabili d'ambiente REALI (non committato)
├── .env.example                # Template variabili (committato)
└── .gitignore
```

---

## 6. I Modelli di Classificazione

### 6.1 CNN 1D

Architettura end-to-end con **~204K parametri addestrabili**. Input: segnale ECG reshapato a `(187, 1)`.

```
Input (187, 1)
    ↓
[Blocco 1] Conv1D(32, kernel=3, ReLU) → BatchNorm → MaxPooling1D → Dropout(0.2)
    ↓
[Blocco 2] Conv1D(64, kernel=3, ReLU) → BatchNorm → MaxPooling1D → Dropout(0.2)
    ↓
[Blocco 3] Conv1D(128, kernel=3, ReLU) → BatchNorm → MaxPooling1D → Dropout(0.2)
    ↓
Flatten → Dense(64, ReLU) → Dense(5, Softmax)
    ↓
Output: distribuzione di probabilita sulle 5 classi
```

**Configurazione di addestramento:** ottimizzatore Adam (learning rate=0.0001), categorical cross-entropy, Early Stopping (patience=5, monitora `val_loss`), convergenza ottimale all'epoca 6.

**Class weights** (inversamente proporzionali alla frequenza nel training set):

| Classe | Peso |
|:---|:---:|
| N (Normale) | 0.24 |
| S (Sopraventricolare) | 7.88 |
| V (Ventricolare) | 3.03 |
| F (Fusion) | 27.32 |
| Q (Non classificabile) | 2.72 |

### 6.2 Random Forest

150 alberi, profondità massima 15, iperparametro nativo `class_weight='balanced'`. Opera su **9 feature morfologiche** estratte per ogni battito di 187 campioni: media, deviazione standard, skewness, kurtosis, valore massimo, valore minimo, range, energia, zero-crossing rate.

Media, skewness e kurtosis contribuiscono a oltre il 65% della capacita discriminativa (Gini importance), ma soffrono di elevata sovrapposizione distributiva tra le classi N, S e V — limite strutturale che il modello CNN supera operando localmente sul segnale.

### 6.3 Stato di affidabilita

Entrambi i modelli espongono uno **stato di affidabilita** basato su soglia fissa:

- **Confidenza >= 0.60** → `"Diagnosi ad alta confidenza"`
- **Confidenza < 0.60** → `"Bassa confidenza (Revisione clinica raccomandata)"`

---

## 7. API Reference

### Prediction Service (porta 8000)

#### `POST /predict/`

Esegue l'inferenza con CNN 1D e RF sul segnale ECG fornito e salva il risultato cifrato su MongoDB.

**Request body:**
```json
{
  "signal": [0.123, -0.045, 0.567, "..."],
  "ground_truth": "N (Normale)"
}
```

> `signal` deve contenere esattamente **187 valori float**. `ground_truth` è opzionale: se presente (es. da CSV MIT-BIH o campione casuale), viene salvato cifrato insieme alla predizione per consentire la validazione clinica a posteriori.

**Response `200 OK`:**
```json
{
  "cnn": {
    "diagnosi": "N (Normale)",
    "confidenza": 0.9823,
    "distribuzione": {
      "N (Normale)": 0.9823,
      "S (Sopraventricolare)": 0.0041,
      "V (Ventricolare)": 0.0089,
      "F (Fusion)": 0.0032,
      "Q (Non classificabile)": 0.0015
    },
    "stato_affidabilita": "Diagnosi ad alta confidenza"
  },
  "rf": {
    "diagnosi": "N (Normale)",
    "confidenza": 0.7200,
    "distribuzione": { "...": "..." },
    "stato_affidabilita": "Diagnosi ad alta confidenza"
  }
}
```

#### `GET /health`

```json
{ "status": "ok", "service": "prediction-service" }
```

---

### History Service (porta 8001)

#### `GET /history/`

Ritorna le predizioni salvate, ordinate dalla più recente. Il segnale grezzo è escluso dalla risposta per alleggerire il payload.

**Query parameters:**

| Parametro | Tipo | Default | Range | Descrizione |
|:---|:---|:---:|:---:|:---|
| `limit` | int | 50 | 1–200 | Numero massimo di record |
| `skip` | int | 0 | ≥ 0 | Offset per la paginazione |

**Response `200 OK`:**
```json
{
  "total": 2,
  "skip": 0,
  "limit": 50,
  "data": [
    {
      "id": "6650a1b2c3d4e5f6a7b8c9d0",
      "timestamp": "2026-06-08T14:32:10.123Z",
      "cnn": { "diagnosi": "V (Ventricolare)", "confidenza": 0.91, "...": "..." },
      "rf":  { "diagnosi": "N (Normale)",      "confidenza": 0.87, "...": "..." },
      "ground_truth": "V (Ventricolare)"
    }
  ]
}
```

#### `GET /history/{prediction_id}`

Ritorna una singola predizione per ID, **incluso il segnale grezzo decifrato**. Utile per il ricaricamento di un segnale dallo storico direttamente nell'interfaccia.

**Path parameter:** `prediction_id` — ObjectId MongoDB (stringa esadecimale a 24 caratteri).

**Response `200 OK`:** stesso schema di `/history/` con l'aggiunta del campo `signal` (array di 187 float).

**Response `404 Not Found`:** `{ "detail": "Predizione non trovata" }`

#### `GET /history/random`

Restituisce un segnale ECG casuale dalla collection `ecg_samples` (dataset MIT-BIH), con il relativo ground truth. Usato dal frontend per la modalità "Esempio casuale".

**Response `200 OK`:**
```json
{
  "id": "6650a1b2c3d4e5f6a7b8c9d1",
  "signal": [0.123, -0.045, "..."],
  "ground_truth": "S (Sopraventricolare)"
}
```

**Response `404 Not Found`:** `{ "detail": "Nessun campione disponibile" }` — indica che il seed non è stato eseguito.

#### `GET /health`

```json
{ "status": "ok", "service": "history-service" }
```

---
## 8. Frontend: Interfaccia Web

L'interfaccia è una Single Page Application servita da Nginx, accessibile via browser senza installazioni aggiuntive. È strutturata in due tab principali: **Predizione** e **Storico**.

### 8.1 Modalità di input

Il sistema supporta tre modalità di caricamento del segnale ECG:

**Input manuale** — inserimento diretto di 187 valori float separati da virgola, con contatore campioni in tempo reale e anteprima grafica del tracciato ECG aggiornata live.

![Caricamento manuale](docs/caricamento-manuale.png)

**Upload CSV** — trascinamento o selezione di un file `.csv` compatibile con il formato MIT-BIH. Se il file contiene 188 valori (187 campioni + etichetta), la classe reale viene estratta automaticamente e mostrata nel badge **Ground Truth**, senza possibilità di modifica manuale.

![Caricamento CSV](docs/caricamento-csv.png)

**Esempio casuale** — estrazione di un segnale casuale dalla collection `ecg_samples` del database (popolata dal seed MIT-BIH), con ground truth già associata.

![Estrazione random](docs/estrazione-random.png)

### 8.2 Risultati diagnostici

Dopo la classificazione, l'interfaccia mostra in parallelo l'output di CNN 1D e Random Forest: diagnosi, confidenza con barra grafica, distribuzione di probabilità sulle 5 classi e badge di affidabilità. Un banner segnala l'accordo o il disaccordo tra i due modelli; se il ground truth è disponibile, un secondo banner confronta le predizioni con la classe reale.

![Esempio predizione](docs/esempio-predizione.png)

> Il modulo **Stato Affidabilità** segnala proattivamente la necessità di revisione clinica quando la confidenza scende sotto la soglia del 60%, prevenendo l'*automation bias* in caso di predizioni incerte.

### 8.3 Storico predizioni

Il tab **Storico** mostra le ultime 50 predizioni salvate nel database, ordinate dalla più recente, con timestamp, diagnosi CNN e RF, livelli di confidenza e ground truth (se disponibile). I dati vengono decifrati on-the-fly dall'history-service prima di essere inviati al frontend: il segnale grezzo è escluso dalla lista per alleggerire la risposta.

![Storico predizioni](docs/storico-predizioni.png)

---


## 9. Guida all'Installazione

**Prerequisiti:** Docker Engine 24+, Docker Compose v2, Git.

### 9.1 Clone e configurazione ambiente

```bash
git clone https://github.com/SoniaSergio/EvSw_Project.git
cd EvSw_Project
cp .env.example .env
# Modifica .env inserendo la tua ENCRYPTION_KEY e le variabili necessarie
```

### 9.2 Download e Posizionamento dei Modelli (MLOps)

Per mantenere il repository leggero e rispettare le best practice di versionamento, i file dei modelli addestrati non sono inclusi nel codice sorgente e vanno caricati manualmente.

**Passaggio 1:** Scarica i modelli pre-addestrati da questo link:
https://drive.google.com/drive/folders/1A8xFMW73WwlKUykkts3UPYPOiUmoZimu?usp=drive_link

I file da scaricare sono:
1. `ecg_cnn_model.h5` — pesi della rete neurale convoluzionale
2. `ecg_rf_model.pkl` — modello Random Forest serializzato

**Passaggio 2:** Posiziona i file nella cartella `prediction-service/models/`. In locale puoi trascinarli manualmente. Su un server remoto (es. Oracle Cloud) usa un client SFTP come MobaXterm: trascina l'intera cartella `models/` nel pannello file a sinistra, navigando fino a `~/EvSw_Project/prediction-service/`.

### 9.3 Seed del Dataset MIT-BIH

La funzionalità "Esempio casuale" richiede che la collection `ecg_samples` di MongoDB sia popolata con i campioni del test set MIT-BIH (~21.000 battiti etichettati). Il file CSV non è incluso nel repository per ragioni di dimensione.

**Passaggio 1:** Scarica `mitbih_test.csv` da Kaggle dal dataset originale al link seguente: https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=mitbih_test.csv 

**Passaggio 2:** Carica il file nella cartella `seed/`:
```
~/EvSw_Project/seed/mitbih_test.csv
```

**Passaggio 3:** Esegui il seed (va fatto una sola volta — lo script rileva automaticamente se la collection è gia popolata e non reinserisce):

```bash
cd ~/EvSw_Project
docker compose build seed
docker compose up seed
```

Attendere il messaggio: `Inseriti X campioni in ecg_samples.`

**Verifica:**
```bash
docker exec -it ecg-mongo mongosh ecgdb --eval "db.ecg_samples.countDocuments({})"
```
Deve restituire circa 21.000.

### 9.4 Avvio in ambiente locale (HTTP)

Il file `nginx.conf` incluso nel repository è configurato per la produzione con HTTPS e dominio reale. Per eseguire il progetto in locale, sostituisci temporaneamente il contenuto di `frontend/nginx.conf` con la seguente versione HTTP:

```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/predict/ {
        proxy_pass http://prediction-service:8000/predict/;
    }

    location /api/history/ {
        proxy_pass http://history-service:8001/history/;
    }
}
```

Assicurarsi che nel `.env` sia impostato `CERTBOT_PATH=./certbot/empty`, poi:

```bash
docker compose up -d
```

Il frontend sarà accessibile su `http://localhost`.

### 9.5 Avvio in produzione (HTTPS con Let's Encrypt)

Il setup HTTPS richiede un dominio reale con record DNS che punta all'IP del server e le porte 80/443 aperte.

**Passaggio 1:** Installa Git e aggiungi l'utente al gruppo Docker (necessario solo al primo accesso sul server):

```bash
sudo dnf install git -y
sudo usermod -aG docker $USER
newgrp docker
```

**Passaggio 2:** Clona il repository e configura l'ambiente:

```bash
git clone https://github.com/SoniaSergio/EvSw_Project.git
cd EvSw_Project
nano .env
# Inserisci ENCRYPTION_KEY, MONGO_URI e CERTBOT_PATH=/etc/letsencrypt
```

**Passaggio 3:** Carica i modelli nella cartella `prediction-service/models/`.

**Passaggio 4:** Crea le cartelle per Certbot sul server host:

```bash
sudo mkdir -p /var/www/certbot
sudo mkdir -p /etc/letsencrypt
```

**Passaggio 5:** Apri le porte sul firewall:

```bash
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --reload
```

**Passaggio 6:** Avvia frontend e mongo per rendere disponibile il challenge HTTP:

```bash
docker compose up -d frontend mongo
```

**Passaggio 7:** Esegui Certbot per ottenere il certificato SSL:

```bash
docker run --rm \
  -v /etc/letsencrypt:/etc/letsencrypt \
  -v /var/www/certbot:/var/www/certbot \
  certbot/certbot certonly --webroot \
  --webroot-path=/var/www/certbot \
  --email TUA_EMAIL@esempio.com \
  --agree-tos \
  --no-eff-email \
  -d tuo-dominio.com
```

**Passaggio 8:** Se il certificato è stato emesso con successo, avvia l'intero stack:

```bash
docker compose up -d
```

**Passaggio 9:** Esegui il seed del dataset MIT-BIH (vedi sezione 9.3).

Il sistema sarà raggiungibile su `https://tuo-dominio.com`.

### 9.6 Verifica dello stato dei servizi

```bash
# Stato container
docker compose ps

# Log in tempo reale
docker compose logs -f prediction-service
docker compose logs -f history-service
```

---

## 10. Variabili d'Ambiente

Copiare `.env.example` in `.env` e configurare le variabili prima dell'avvio. Il file `.env` è incluso nel `.gitignore` e non deve essere mai committato.

| Variabile | Default / Esempio | Descrizione |
|:---|:---|:---|
| `MONGO_URI` | `mongodb://mongo:27017/ecgdb` | URI di connessione MongoDB. In Docker Compose usare il nome del servizio (`mongo`) come host. |
| `ENCRYPTION_KEY` | `g3k8F...=` | Chiave simmetrica a 32-byte codificata in URL-safe Base64 per l'algoritmo Fernet (ALDE). Deve essere identica tra locale e server. Generabile con: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `CERTBOT_PATH` | `./certbot/empty` | Percorso dei volumi SSL. Impostare `./certbot/empty` per sviluppo locale (HTTP) o `/etc/letsencrypt` per la produzione (HTTPS). |

---

## 11. Verifica Sicurezza Database (Cifratura ALDE)

Per verificare che i dati siano correttamente cifrati a riposo (Encryption at Rest) su MongoDB, è possibile ispezionare direttamente il database per confermare che le informazioni cliniche non siano accessibili in chiaro:

Accedere alla shell di MongoDB nel container:

```bash
docker exec -it ecg-mongo mongosh
```

Selezionare il database e interrogare la collezione:

```javascript
use ecgdb
db.predictions.find().sort({_id: -1}).limit(1).pretty()
```

Nota: I campi `signal`, `cnn`, `rf` e `ground_truth` appariranno come stringhe cifrate (es: `gAAAAA...`), confermando che il pattern ALDE impedisce l'accesso ai dati clinici sensibili anche in caso di compromissione del database.

Per verificare la cifratura anche sulla collection `ecg_samples`:

```javascript
db.ecg_samples.findOne()
```

Il campo `signal_encrypted` dovrà apparire come stringa cifrata, mentre `ground_truth` e `source` sono in chiaro (metadati non sensibili).

---

## 12. Risoluzione Problemi (Troubleshooting)

| Sintomo | Causa Probabile | Soluzione |
|:---|:---|:---|
| **Crash "Manca la ENCRYPTION_KEY"** | Il file `.env` non esiste o la variabile non è mappata in Docker Compose | Assicurarsi di aver copiato `.env.example` in `.env` e aver compilato la chiave Fernet. |
| **`prediction-service` in crash loop** | File modello `.h5` o `.pkl` mancante o inaccessibile | Verificare che la cartella `./prediction-service/models/` contenga entrambi i file addestrati. |
| **"Attese 187 valori, trovati N"** | Il file CSV contiene header o formati non validi | Verificare che la prima riga contenga esattamente 187 valori numerici separati da virgola (o 188 per formato MIT-BIH). |
| **Storico sempre vuoto** | Il frontend non raggiunge l'`history-service` | Controllare che il container `ecg-history` sia in esecuzione (`docker compose ps`) e che `nginx.conf` esegua il proxy corretto. |
| **Errore HTTPS "certificato non valido"** | Certbot non ha ancora emesso il certificato | Eseguire prima il challenge HTTP (Passaggio 6 della guida) e verificare che i record DNS puntino all'IP del server. |
| **Predizione lenta (> 5s) al primo avvio** | Lazy loading di TensorFlow in memoria | Comportamento normale solo alla prima inferenza dopo l'avvio. Le chiamate successive risponderanno in < 200 ms. |
| **MongoDB: "Connection refused"** | Race condition all'avvio | Attendere 10-15 secondi e riavviare: `docker compose restart prediction-service history-service`. |
| **`cryptography` / `libgomp1` not found** | Dipendenze mancanti nell'immagine compilata | Eseguire `docker compose build --no-cache`. |
| **`permission denied` su docker** | Utente non nel gruppo docker | Eseguire `sudo usermod -aG docker $USER` seguito da `newgrp docker`. |
| **Certbot: `unauthorized` / 404** | Le cartelle `/var/www/certbot` o `/etc/letsencrypt` non esistono sull'host | Eseguire `sudo mkdir -p /var/www/certbot && sudo mkdir -p /etc/letsencrypt` e riavviare il frontend prima di rieseguire Certbot. |
| **"Nessun campione disponibile" su Esempio casuale** | La collection `ecg_samples` è vuota | Caricare `mitbih_test.csv` in `seed/` via SFTP e lanciare `docker compose build seed && docker compose up seed`. |
| **Seed: `SyntaxError Non-UTF-8`** | Il file `seed_db.py` contiene caratteri accentati o non ASCII | Riscrivere il file dal server con `cat > seed_db.py << 'EOF'` evitando caratteri non ASCII nel testo. |

---

## 13. Riferimenti Scientifici

- **[MIT-BIH Arrhythmia Database]** G. B. Moody, R. G. Mark — *"The impact of the MIT-BIH arrhythmia database"*, IEEE Engineering in Medicine and Biology Magazine, vol. 20, no. 3, pp. 45-50, 2001.

- **[Kailan et al., 2025]** — *"Efficient ECG classification based on machine learning and feature selection algorithm for IoT-5G enabled health monitoring systems"*, International Journal of Intelligent Engineering and Systems, vol. 18, no. 1, pp. 1187-1199.

- **[Mohebbanaaz et al., 2025]** — *"A novel inference system for detecting cardiac arrhythmia using deep learning framework"*, Neural Computing and Applications, vol. 37, no. 16, pp. 9759-9775.

- **[Zhang et al., 2025]** — *"MSFT: A multi-scale feature-based transformer model for arrhythmia classification"*, Biomedical Signal Processing and Control, vol. 100.

---

*Evoluzione del Software 2025/2026 · Universita degli Studi di Bari Aldo Moro · Sonia Sergio, 796129*