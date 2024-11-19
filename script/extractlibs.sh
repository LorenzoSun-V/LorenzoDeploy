#! /bin/sh

cp -v  $(ldd $1 | awk '{if (match($3,"/")){ printf("%s "),$3 } }') $2
