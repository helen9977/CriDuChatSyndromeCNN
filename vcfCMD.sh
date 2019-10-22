#!/bin/bash
awk '{if($1==5 && $5=="<DEL>")print $2,$8}'> vcf.txt

awk -F";" '{print$1,$4}' > vcf0.txt

sed -i 's/[A-Z]/ /g' vcf0.txt 
sed -i 's/=/ /g' vcf0.txt 
