# ECG Arrhythmia Classifier
Classificazione Automatica delle Aritmie ECG
Sistema diagnostico comparativo CNN 1D vs Random Forest su segnali elettrocardiografici, evoluzione dell'interfaccia Gradio sviluppata per l'esame di Sistemi Multimediali / Evoluzione del Software — Università degli Studi di Bari "Aldo Moro", A.A. 2025/2026.

--- 
## Indice

- [ECG Arrhythmia Classifier](#ecg-arrhythmia-classifier)
  - [Indice](#indice)
  - [1. Visione del Progetto e Origine](#1-visione-del-progetto-e-origine)
  - [2. Evoluzione Architetturale: da Gradio a Microservizi](#2-evoluzione-architetturale-da-gradio-a-microservizi)
  - [3. Architettura del Sistema e Sicurezza (ALDE)](#3-architettura-del-sistema-e-sicurezza-alde)
    - [Flusso di una predizione](#flusso-di-una-predizione)
  - [4. Stack Tecnologico](#4-stack-tecnologico)
  - [5. Struttura del Progetto](#5-struttura-del-progetto)
  - [6. I Modelli di Classificazione](#6-i-modelli-di-classificazione)
    - [6.1 CNN 1D](#61-cnn-1d)
    - [6.2 Random Forest](#62-random-forest)
    - [6.3 Stato di affidabilità](#63-stato-di-affidabilità)
  - [7. API Reference](#7-api-reference)
    - [Prediction Service (porta 8000)](#prediction-service-porta-8000)
      - [`POST /predict/`](#post-predict)
      - [`GET /health`](#get-health)
    - [History Service (porta 8001)](#history-service-porta-8001)
      - [`GET /history/`](#get-history)
      - [`GET /history/{prediction_id}`](#get-historyprediction_id)
      - [`GET /health`](#get-health-1)
  - [8. Frontend: Interfaccia Web](#8-frontend-interfaccia-web)
    - [Tab Predizione](#tab-predizione)
    - [Tab Storico](#tab-storico)
  - [9. Guida all'Installazione](#9-guida-allinstallazione)
    - [9.1 Clone e configurazione ambiente](#91-clone-e-configurazione-ambiente)
    - [9.2 Download e Posizionamento dei Modelli (MLOps)](#92-download-e-posizionamento-dei-modelli-mlops)
    - [9.3 Avvio in ambiente locale (HTTP)](#93-avvio-in-ambiente-locale-http)
    - [9.4 Avvio in produzione (HTTPS con Let's Encrypt)](#94-avvio-in-produzione-https-con-lets-encrypt)
    - [9.5 Verifica dello stato dei servizi](#95-verifica-dello-stato-dei-servizi)
  - [10. Variabili d'Ambiente](#10-variabili-dambiente)
  - [11. Verifica Sicurezza Database (Cifratura ALDE)](#11-verifica-sicurezza-database-cifratura-alde)
  - [12. Risoluzione Problemi (Troubleshooting)](#12-risoluzione-problemi-troubleshooting)
  - [13. Riferimenti Scientifici](#13-riferimenti-scientifici)

---
    
## 1. Visione del Progetto e Origine

**ECG Arrhythmia Classifier** è un sistema web di supporto diagnostico per la classificazione automatica delle aritmie cardiache a partire da segnali ECG del dataset standard **MIT-BIH Arrhythmia Database** (87.554 campioni, 5 classi, 360 Hz).

Dal punto di vista dell'Ingegneria del Software, questo sistema è concepito non solo come un progetto isolato, ma come un vero e proprio **prodotto software**: un sistema generico nato per cogliere un'opportunità di business (il supporto clinico rapido) in grado di fornire funzionalità utili a una vasta gamma di utenti medici, garantendo scalabilità e manutenibilità a lungo termine.

Il progetto mette a confronto due paradigmi opposti di machine learning:

- **Random Forest (RF):** approccio classico basato su feature engineering manuale (9 descrittori morfologici). Offre trasparenza decisionale tramite feature importance, ma soffre di limitata separabilità sulle classi minoritarie patologiche.
- **CNN 1D:** architettura deep learning end-to-end su 3 blocchi convoluzionali (~204K parametri). Apprende autonomamente le gerarchie di pattern direttamente dal segnale grezzo, abbattendo drasticamente i falsi negativi sulle aritmie critiche.

La metrica primaria adottata è la **macro Recall**, clinicamente più rilevante dell'accuracy globale in presenza di forte sbilanciamento.

| Modello | Accuracy | Macro Recall | Macro F1 | AUC |
|:---|:---:|:---:|:---:|:---:|
| Random Forest | 0.83 | 0.69 | 0.57 | 0.903 |
| **CNN 1D** | **0.96** | **0.92** | **0.82** | **0.989** |

La CNN 1D riduce dell'**83% i falsi negativi sulla classe Ventricolare** rispetto alla RF (Recall: 0.60 → 0.91), la classe a maggiore rischio clinico per il paziente.

---

## 2. Evoluzione Architetturale: da Gradio a Microservizi

Il progetto nasce come prototipo monolitico sviluppato originariamente per l'esame, in cui i modelli erano incapsulati in un'interfaccia interattiva **Gradio** eseguita localmente su Google Colab.

Questa versione rappresenta la sua **evoluzione in un sistema distribuito**, riprogettato secondo la definizione formale **IEEE di Architettura Software**: *"L'organizzazione fondamentale di un sistema sw che si concretizza nei suoi componenti, nelle loro relazioni reciproche e con l'ambiente e nei principi che ne guidano la progettazione e l'evoluzione"*.

Si è passati da un approccio monolitico a un **approccio orientato ai servizi (Service-Oriented Architecture)**, particolarmente adatto per il software basato su cloud, in cui il sistema è stato scomposto in servizi a grana fine, isolati e resilienti ai guasti:

| Aspetto | Versione Gradio (Esame) | Versione Attuale (Microservizi) |
|:---|:---|:---|
| **Deployment** | Google Colab, locale | Docker Compose, server remoto con HTTPS |
| **Interfaccia** | Gradio auto-generata | Frontend custom HTML/CSS/JS (Nginx) |
| **Persistenza** | Nessuna | MongoDB: storico completo e cifrato |
| **Architettura** | Monolitica, single-process | 4 servizi indipendenti dockerizzati |
| **Inferenza** | Sincrona, single-thread | FastAPI async, latenza media < 200 ms |
| **Scalabilità** | Non scalabile | Servizi indipendentemente scalabili al variare del carico |
| **Sicurezza** | Nessuna | HTTPS/TLS 1.3 + **Cifratura AES-256 (ALDE)** nel DB |
| **Input** | Manuale o da file locale | Manuale, CSV upload, segnale casuale da storico |

---

## 3. Architettura del Sistema e Sicurezza (ALDE)

Il sistema è composto da **4 servizi Docker** comunicanti su una rete bridge dedicata (`ecg-net`). I container isolano l'applicazione nello spazio utente sfruttando i meccanismi del kernel Linux (`namespaces` e `cgroups`).

Per garantire l'assoluta confidenzialità dei dati medici, l'architettura implementa il pattern **Application-Layer Data Encryption (ALDE)**. Nessun dato clinico in chiaro risiede nel database: il Service Layer cifra e decifra i dati asincronamente utilizzando l'algoritmo **AES-256 in modalità CBC/HMAC (Fernet)** prima di interagire con MongoDB (garantendo il principio di *Encryption at Rest* e un disaccoppiamento logico perfetto).

```text
                        ┌─────────────────────────────────┐
                        │          Client Browser         │
                        └─────────────────────────────────┘
                                        │ HTTPS :443
                        ┌─────────────────────────────────┐
                        │     Frontend (Nginx)            │
                        │  - Serve HTML/CSS/JS statici    │
                        │  - Reverse proxy /api/* │
                        │  - Redirect HTTP → HTTPS        │
                        └────────────┬────────────┬───────┘
                                     │            │
              /api/predict/          │            │   /api/history/
                        ┌────────────▼──┐    ┌────▼───────────────┐
                        │  Prediction   │    │  History Service   │
                        │  Service      │    │  (FastAPI :8001)   │
      (Cifra i dati  ←──┤  (FastAPI     │    │  - Decifra i dati  ├──→ (Estrae JSON
      prima del DB)     │   :8000)      │    │    al volo per UI  │     in chiaro)
                        └──────┬────────┘    └───────────┬────────┘
                               │                         │
                               └──────────────┬──────────┘
                                              │
                              ┌───────────────▼────────────┐
                              │        MongoDB :27017      │
                              │   Database: ecgdb          │
                              │   Collection: predictions  │
                              │   (Contiene SOLO dati AES) │
                              └────────────────────────────┘
```
### Flusso di una predizione

1. Il browser invia `POST /api/predict/` con il segnale (array di 187 float).
2. Nginx fa reverse proxy verso `prediction-service:8000/predict/`.
3. Il prediction service esegue inferenza **parallela** su CNN 1D e RF.
4. I risultati diagnostici e il segnale vengono cifrati in stringhe illeggibili dal microservizio Python tramite la secret key.
5. I dati protetti vengono salvati su MongoDB.
6. Il frontend renderizza i risultati ricevuti in chiaro dalla risposta API.
7. Nelle letture successive, l'history-service estrarrà il dato cifrato e lo decifrerà in RAM prima di mandarlo alla UI (In-Memory Processing).

---

## 4. Stack Tecnologico

| Layer | Tecnologia | Versione | Ruolo |
|:---|:---|:---|:---|
| **Frontend** | HTML5 / CSS3 / Vanilla JS | — | Interfaccia utente |
| **Frontend Server** | Nginx Alpine | 1.27 | Serve statico + reverse proxy |
| **Prediction API** | FastAPI + Uvicorn | 0.111 / 0.30 | Inferenza CNN e RF |
| **History API** | FastAPI + Uvicorn | 0.115 / 0.32 | CRUD storico predizioni |
| **Deep Learning** | TensorFlow / Keras | 2.16.1 | Modello CNN 1D |
| **Machine Learning** | Scikit-learn + Joblib | 1.5.0 | Modello Random Forest |
| **Feature Engineering** | Pandas + NumPy | 2.2.2 / 1.26.4 | Estrazione descrittori morfologici |
| **Sicurezza / Crittografia**| Cryptography (Fernet) | 42.0.5 | Cifratura simmetrica ALDE AES-256 |
| **Database** | MongoDB | 7.0 | Persistenza predizioni + segnali |
| **ODM Asincrono** | Motor + PyMongo | 3.x / 4.x | Driver async MongoDB per FastAPI |
| **Validazione** | Pydantic | 2.x | Validazione input/output API |
| **Containerization** | Docker + Docker Compose | — | Orchestrazione servizi |
| **TLS/HTTPS** | Let's Encrypt + Certbot | — | Certificati SSL con rinnovo automatico |

---

## 5. Struttura del Progetto

```text
ECG-ARRHYTHMIA-/
├── frontend/
│   ├── src/
│   │   ├── index.html          # SPA principale
│   │   ├── app.js              # Logica UI: input, predizione, storico
│   │   └── style.css           # Design system (CSS variables, componenti)
│   ├── Dockerfile              # Build Nginx Alpine + copia statici
│   └── nginx.conf              # Reverse proxy, HTTPS, redirect HTTP→HTTPS
│
├── prediction-service/
│   ├── models/
│   │   ├── ecg_cnn_model.h5    # Pesi CNN 1D addestrata (TensorFlow/Keras)
│   │   └── ecg_rf_model.pkl    # Modello Random Forest serializzato (joblib)
│   ├── routers/
│   │   └── predict.py          # POST /predict/ — orchestrazione inferenza
│   ├── services/
│   │   ├── cnn_service.py      
│   │   ├── rf_service.py       
│   │   └── db_service.py       # Cifratura Fernet e save_prediction()
│   ├── main.py                 
│   ├── requirements.txt        # Include cryptography==42.0.5
│   └── Dockerfile              
│
├── history-service/
│   ├── routers/
│   │   └── history.py          
│   ├── services/
│   │   └── db_service.py       # Decifratura on-the-fly e get_predictions()
│   ├── main.py                 
│   ├── requirements.txt        
│   └── Dockerfile              
│
├── mongo/
│   └── init/
│       └── init.js             # Crea collection predictions + indici MongoDB
│
├── docker-compose.yml          # Orchestrazione: mongo, prediction, history, frontend, certbot
├── .env                        # Variabili d'ambiente REALI (non committato)
├── .env.example                # Template variabili (committato)
└── .gitignore

```
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
Output: distribuzione di probabilità sulle 5 classi
```

**Configurazione di addestramento:** ottimizzatore Adam (lr=0.0001), categorical cross-entropy, Early Stopping (patience=5, monitora `val_loss`), convergenza ottimale all'epoca 6.

**Class weights** (inversamente proporzionali alla frequenza nel training set):

| Classe | Peso |
|:---|:---:|
| N (Normale) | 0.24 |
| S (Sopraventricolare) | 7.88 |
| V (Ventricolare) | 3.03 |
| F (Fusion) | 27.32 |
| Q (Non classificabile) | 2.72 |

### 6.2 Random Forest

150 alberi, profondità massima 15, `class_weight='balanced'`. Opera su **9 feature morfologiche** estratte per ogni battito di 187 campioni: media, deviazione standard, skewness, kurtosis, valore massimo, valore minimo, range, energia, zero-crossing rate.

Media, skewness e kurtosis contribuiscono a oltre il 65% della capacità discriminativa (Gini importance), ma soffrono di elevata sovrapposizione distributiva tra le classi N, S e V — limite strutturale che il CNN supera operando localmente sul segnale.

### 6.3 Stato di affidabilità

Entrambi i modelli espongono uno **stato di affidabilità** basato su soglia fissa:

- **Confidenza ≥ 0.60** → `"Diagnosi ad alta confidenza"`
- **Confidenza < 0.60** → `"Bassa confidenza (Revisione clinica raccomandata)"`

Questo meccanismo è particolarmente rilevante per i falsi positivi a bassa confidenza della RF, che l'interfaccia segnala proattivamente all'operatore prima che venga accettato un allarme potenzialmente errato.

---

## 7. API Reference

### Prediction Service (porta 8000)

#### `POST /predict/`

Esegue l'inferenza parallela con CNN 1D e RF sul segnale ECG fornito. Salva il risultato su MongoDB.

**Request body:**
```json
{
  "signal": [0.123, -0.045, 0.567, "..."]
}
```
> Il campo `signal` deve contenere esattamente **187 valori float**. La validazione è gestita da Pydantic; in caso contrario viene restituito `422 Unprocessable Entity`.

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

Ritorna le ultime predizioni salvate, ordinate dalla più recente. Il segnale grezzo è escluso dalla lista per alleggerire la risposta.

**Query parameters:**

| Parametro | Tipo | Default | Range | Descrizione |
|:---|:---|:---:|:---:|:---|
| `limit` | int | 50 | 1–200 | Numero massimo di record da restituire |
| `skip` | int | 0 | ≥ 0 | Offset per la paginazione |

**Response `200 OK`:**
```json
{
  "total": 2,
  "skip": 0,
  "limit": 50,
  "data": [
    {
      "id": "6849a1f3c2e4d500123abcde",
      "timestamp": "2026-06-09T14:32:00.000Z",
      "cnn": { "diagnosi": "V (Ventricolare)", "confidenza": 0.9123, "..." },
      "rf":  { "diagnosi": "N (Normale)",      "confidenza": 0.8800, "..." }
    }
  ]
}
```

#### `GET /history/{prediction_id}`

Ritorna il dettaglio completo di una singola predizione, **segnale grezzo incluso** (187 float).

#### `GET /health`

```json
{ "status": "ok", "service": "history-service" }
```

---

## 8. Frontend: Interfaccia Web

Single-page application statica servita da Nginx. Nessun framework JS — vanilla HTML/CSS/JS per massima portabilità e latenza minima.

### Tab Predizione

Tre modalità di input del segnale ECG:

- **Manuale** — textarea per incollare 187 valori separati da virgola/spazio/punto e virgola. Contatore campioni in tempo reale con anteprima del tracciato ECG su canvas.
- **CSV** — drag & drop o selezione file. Supporta la prima riga del formato MIT-BIH standard (188 valori: i 187 campioni + etichetta di classe, che viene automaticamente rimossa).
- **Esempio casuale** — recupera un segnale reale dal test set MIT-BIH tramite history service per validazione interattiva.

Per ogni predizione vengono mostrati:
- Diagnosi principale di ciascun modello
- Confidenza con barra progress animata
- Stato di affidabilità (badge verde/rosso)
- Distribuzione di probabilità sulle 5 classi (mini-chart a barre)
- Banner di accordo/disaccordo diagnostico tra CNN e RF

### Tab Storico

Lista paginata delle predizioni precedenti con timestamp, diagnosi CNN e RF, confidenze. Pulsante di aggiornamento manuale.

---

## 9. Guida all'Installazione

**Prerequisiti:** Docker Engine 24+, Docker Compose v2, Git.

### 9.1 Clone e configurazione ambiente

```bash
git clone <url-repository>
cd ECG-ARRHYTHMIA
cp .env.example .env
mkdir -p certbot/empty
# Modifica .env con i parametri personalizzati se necessario
```
### 9.2 Download e Posizionamento dei Modelli (MLOps)

Per mantenere il repository leggero e rispettare le best practice di versionamento (evitando il tracciamento di artefatti binari pesanti), i file dei modelli addestrati non sono inclusi direttamente nel codice sorgente.

**Passaggio 1:** Scarica i modelli pre-addestrati da questo link:
https://drive.google.com/drive/folders/1A8xFMW73WwlKUykkts3UPYPOiUmoZimu?usp=drive_link

I file da scaricare sono due:
1. `ecg_cnn_model.h5` (Pesi della rete neurale convoluzionale)
2. `ecg_rf_model.pkl` (Modello Random Forest serializzato)

**Passaggio 2:** Copia i due file appena scaricati all'interno della directory del Prediction Service eseguendo questi comandi (o trascinandoli manualmente):

```bash
cp cartella_download/ecg_cnn_model.h5  ./prediction-service/models/
cp cartella_download/ecg_rf_model.pkl  ./prediction-service/models/
```

### 9.3 Avvio in ambiente locale (HTTP)

Impostare CERTBOT_PATH=/etc/letsencrypt nel .env.

Eseguire: docker compose up -d frontend mongo.

Eseguire il challenge Certbot e avviare: docker compose up --build -d.

Il frontend sarà accessibile su `http://localhost`.

### 9.4 Avvio in produzione (HTTPS con Let's Encrypt)

Il setup HTTPS richiede un dominio reale e un server con porte 80/443 aperte.

1. Assicurarsi che nel file `.env` il percorso sia impostato a `CERTBOT_PATH=/etc/letsencrypt`.
2. Eseguire il primo avvio per il challenge di Certbot:
   ```bash
   docker compose up -d frontend mongo
   ```
### 9.5 Verifica dello stato dei servizi

```bash
# Stato container
docker compose ps

# Log in tempo reale
docker compose logs -f prediction-service
```

---

## 10. Variabili d'Ambiente

Copiare `.env.example` in `.env` e configurare le variabili prima dell'avvio. Il file `.env` è incluso nel `.gitignore` e non deve essere mai committato, in quanto contiene dati sensibili.

| Variabile | Default / Esempio | Descrizione |
|:---|:---|:---|
| `MONGO_URI` | `mongodb://mongo:27017/ecgdb` | URI di connessione MongoDB. In Docker Compose usare il nome del servizio (`mongo`) come host. |
| `ENCRYPTION_KEY` | `g3k8F...=` | Chiave simmetrica a 32-byte codificata in URL-safe Base64 per l'algoritmo Fernet (ALDE). Deve essere identica tra locale e server. |
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
Nota: I campi signal, cnn e rf appariranno come stringhe cifrate (es: gAAAAA...), confermando che il pattern ALDE impedisce l'accesso ai dati clinici sensibili anche in caso di compromissione del database.

--- 
## 12. Risoluzione Problemi (Troubleshooting)

| Sintomo | Causa Probabile | Soluzione |
|:---|:---|:---|
| **Crash "Manca la ENCRYPTION_KEY"** | Il file `.env` non esiste o la variabile non è mappata in Docker Compose | Assicurarsi di aver clonato `.env.example` in `.env` e aver compilato la chiave Fernet a 32 byte. |
| **`prediction-service` in crash loop** | File modello `.h5` o `.pkl` mancante o inaccessibile | Verificare che la cartella `./prediction-service/models/` contenga entrambi i file addestrati. |
| **"Attese 187 valori, trovati N"** | Il file CSV contiene header o formati non validi | Verificare che la prima riga contenga esattamente 187 valori numerici separati da virgola (o 188 per formato MIT-BIH). |
| **Storico sempre vuoto** | Il frontend non raggiunge l'`history-service` | Controllare che il container `ecg-history` sia in esecuzione (`docker compose ps`) e che `nginx.conf` esegua il proxy corretto. |
| **Errore HTTPS "certificato non valido"** | Certbot non ha ancora emesso il certificato | Eseguire prima il challenge HTTP (Step 2 della guida) e verificare che i record DNS puntino all'IP del server. |
| **Predizione lenta (> 5s) al primo avvio** | Lazy loading di TensorFlow in memoria | Comportamento normale solo alla prima inferenza dopo l'avvio. Le chiamate successive risponderanno in < 200 ms. |
| **MongoDB: "Connection refused"** | Race condition all'avvio: i servizi FastAPI tentano la connessione prima che Mongo sia pronto | Attendere 10-15 secondi e riavviare i servizi dipendenti: `docker compose restart prediction-service history-service`. |
| **`cryptography` / `libgomp1` not found** | Dipendenze mancanti nell'immagine compilata | Eseguire `docker compose build --no-cache` per forzare il download del modulo Fernet e le librerie di TensorFlow. |

---


## 13. Riferimenti Scientifici

- **[MIT-BIH Arrhythmia Database]** G. B. Moody, R. G. Mark — *"The impact of the MIT-BIH arrhythmia database"*, IEEE Engineering in Medicine and Biology Magazine, vol. 20, no. 3, pp. 45–50, 2001.

- **[Kailan et al., 2025]** — *"Efficient ECG classification based on machine learning and feature selection algorithm for IoT-5G enabled health monitoring systems"*, International Journal of Intelligent Engineering and Systems, vol. 18, no. 1, pp. 1187–1199. *(Approccio PSO + SVM, baseline ML classico)*

- **[Mohebbanaaz et al., 2025]** — *"A novel inference system for detecting cardiac arrhythmia using deep learning framework"*, Neural Computing and Applications, vol. 37, no. 16, pp. 9759–9775. *(DeepBiLSTMnet — confronto architetture ricorrenti)*

- **[Zhang et al., 2025]** — *"MSFT: A multi-scale feature-based transformer model for arrhythmia classification"*, Biomedical Signal Processing and Control, vol. 100. *(Architettura ibrida CNN + Transformer — confronto complessità)*

---

*Evoluzione del Software 2025/2026 · Università degli Studi di Bari Aldo Moro · Sonia Sergio, 796129*