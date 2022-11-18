echo "Activating conda env"
conda activate base

echo "Starting xterm app in Xpra"
xpra start --start=xterm
