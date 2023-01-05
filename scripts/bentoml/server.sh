# Development mode
bentoml serve service:svc --reload --port=7010

# Non-dev model with ssl certs
bentoml serve service:svc --reload --port=7010 \
--ssl-certfile ~/.ssl/certs/nginx-selfsigned.crt --ssl-keyfile ~/.ssl/private/nginx-selfsigned.key