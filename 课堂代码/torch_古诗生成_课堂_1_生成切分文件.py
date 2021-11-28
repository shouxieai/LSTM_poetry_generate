import numpy as np

def spilt_poetry(file = "poetry_7.txt"):
    all_data = open(file,"r",encoding = "utf-8").read()
    all_data_split = " ".join(all_data)
    with open("split.txt","w",encoding = 'utf-8') as f:
        f.write(all_data_split)
    return all_data.split("\n")




if __name__ == "__main__" :
    all_data = spilt_poetry()
    pass


