nvcc brent-kung.cu -o brent-kung_test
nvcc kogge-stone.cu -o kogge-stone_test
nvcc sklansky.cu -o sklansky_test

./brent-kung_test
./kogge-stone_test
./sklansky_test

ncu --set full -f --import-source on -o brent-kung ./brent-kung_test
ncu --set full -f --import-source on -o kogge-stone ./kogge-stone_test
ncu --set full -f --import-source on -o sklansky ./sklansky_test