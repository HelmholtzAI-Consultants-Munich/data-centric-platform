# SSH tunnelling solution
ssh -L 7009:localhost:7009 ubuntu@jusuf-vm2
echo "You can now access the novnc webapp via https://localhost:7009/vnc.html"