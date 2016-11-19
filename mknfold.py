import random

random.seed(12)
libsvm_data=open("libsvm_data.txt")
train=open("train.txt","w")
test=open("test.txt","w")

for l in libsvm_data:
    if random.randint(1,10)<=4:
        test.write(l)
    else:
        train.write(l)

libsvm_data.close()
train.close()
test.close()