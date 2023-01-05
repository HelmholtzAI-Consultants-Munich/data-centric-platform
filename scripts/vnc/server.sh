# Start the VNC as localhost
vncserver -localhost
vncpasswd # This will ask you to set a password for the client

# Start VNC Client web app
# git clone https://github.com/novnc/noVNC.git
sudo /home/ubuntu/noVNC/utils/novnc_proxy --vnc localhost:5901 --listen localhost:7009 \
--cert /etc/ssl/certs/nginx-selfsigned.crt --key /etc/ssl/private/nginx-selfsigned.key
