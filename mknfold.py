import random

random.seed(12)
libsvm_data=open("libsvm_data.txt")
train=open("train.txt","w")
valid=open("valid.txt","w")

for l in libsvm_data:
    if random.randint(1,10)<=4:
        valid.write(l)
    else:
        train.write(l)

libsvm_data.close()
train.close()
valid.close()