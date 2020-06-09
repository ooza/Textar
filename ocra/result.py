file =open ("./ocr/output3.txt", 'r')
newFile=open ("./ocr/resulttxt.txt", 'w')
s=''
for line in file:
    if line[0]==('"'):
        newFile.write(s)
        newFile.write("\n")
        s=''
    elif line[0]==" ":
        s=s+line[1]
    else:
        s=s+line[0]
newFile.write(s)
newFile.close()
file.close()


        

    
    
