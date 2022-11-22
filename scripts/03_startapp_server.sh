echo "Activating conda env"
conda activate base

echo "Starting xterm app in Xpra"
xpra start --start=xterm

echo "This will show the display number and the client can connect to that xpra session on their local
machine using that display number"