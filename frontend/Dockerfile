FROM nginx:1.27-alpine
COPY src/ /usr/share/nginx/html/
COPY nginx.http.conf.template  /etc/nginx/templates/
COPY nginx.https.conf.template /etc/nginx/templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && rm /etc/nginx/conf.d/default.conf
ENTRYPOINT ["/entrypoint.sh"]