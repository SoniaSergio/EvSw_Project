#!/bin/sh
set -e

# Scegli il template in base alla variabile NGINX_MODE (http | https)
NGINX_MODE=${NGINX_MODE:-http}
TEMPLATE="/etc/nginx/templates/nginx.${NGINX_MODE}.conf.template"

echo "Nginx mode: ${NGINX_MODE}, server: ${SERVER_NAME:-localhost}"

envsubst '${SERVER_NAME}' < "$TEMPLATE" > /etc/nginx/conf.d/default.conf

exec nginx -g 'daemon off;'