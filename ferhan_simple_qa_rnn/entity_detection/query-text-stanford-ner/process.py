import sys

# python process.py test test-peng.txt test.txt
dataset = sys.argv[1]
outfile = open(sys.argv[3], 'w')
with open(sys.argv[2], 'r') as f:
    for i, line in enumerate(f):
        items = line.split(" #### ")
        question = items[0].strip()
        query = items[1].strip()
        if query == "":
            query = question
        outfile.write("{}-{} %%%% {}\n".format(dataset, i+1, query))

outfile.close()

