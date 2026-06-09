FROM nginx:1.27-alpine

# Rimuove la configurazione di default
RUN rm /etc/nginx/conf.d/default.conf

# Copia la nostra configurazione
COPY nginx.conf /etc/nginx/conf.d/app.conf

# Copia i file statici del frontend
COPY src/ /usr/share/nginx/html/

EXPOSE 80