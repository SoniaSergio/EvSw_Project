db = db.getSiblingDB('ecgdb');

db.createCollection('predictions');
db.predictions.createIndex({ "timestamp": -1 });
db.predictions.createIndex({ "cnn.diagnosi": 1 });