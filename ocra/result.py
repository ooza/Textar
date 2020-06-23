file =open ("./ocra/output_htk_format.txt", 'r')
newFile=open ("./ocra/final_resul.txt", 'w')
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


        

    
    
