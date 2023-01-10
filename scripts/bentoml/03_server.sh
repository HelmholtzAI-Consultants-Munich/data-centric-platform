# Development mode
/home/ubuntu/active-learning-platform/src/active_learning_platform/server
bentoml serve service:svc --reload --port=7010

# Non-dev model with ssl certs
/home/ubuntu/active-learning-platform/src/active_learning_platform/server
bentoml serve service:svc --port=7010 \
--ssl-certfile ~/.ssl/certs/nginx-selfsigned.crt --ssl-keyfile ~/.ssl/private/nginx-selfsigned.key