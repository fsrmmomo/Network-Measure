
a = [2,1]
b = [1,2]

c = [20,20]
while c[0]>0 or c[1]>0:
    w1 = a[0]/c[0] + a[1]/c[1]
    w2 = b[0]/c[0] + b[1]/c[1]
    # print(str(w1) + " " + str(w2))
    if w1>=w2:
        print("b")
        c[0] -= b[0]
        c[1] -= b[1]
    else:
        print("a")
        c[0] -= a[0]
        c[1] -= a[1]
    print(c)