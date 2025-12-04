python bxc.py test1.bx
gcc -no-pie -o test1.exe test1.s bxruntime.c

python bxc.py test2.bx
gcc -no-pie -o test2.exe test2.s bxruntime.c

python bxc.py test3.bx
gcc -no-pie -o test3.exe test3.s bxruntime.c

python bxc.py test4.bx
gcc -no-pie -o test4.exe test4.s bxruntime.c

./test1.exe
./test2.exe
./test3.exe 
./test4.exe


