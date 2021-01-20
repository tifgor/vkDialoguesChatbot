import os
import io
import codecs
import format
from pathlib import Path

trainc = codecs.open('train.txt', 'w', encoding='utf-8')

vkName = "имя фамилия" #lowercase: "имя фамилия"
myNameVkOpt = '/./' + vkName
dialoguesPath = 'C:/dialogues'

alines = []
blines = []
blockB = False
writeToA = True
def writeLines(lines, oppNameVkOpt):
    global writeToA
    global blockB
    for i in range(len(lines)):
        if lines[i].startswith(oppNameVkOpt):
            writeToA=True
            blockB=False
        elif lines[i].startswith(myNameVkOpt):
            writeToA=False
        else:
            if writeToA:
                if i+1<len(lines):
                    if lines[i+1].startswith(myNameVkOpt):
                        alines.append(format.format(lines[i]))
            if not writeToA and not blockB:
                blines.append(format.format(lines[i]))
                blockB = True

def composeFile():
    i = 0
    print("lines in a: ", len(alines))
    print("lines in b: ", len(blines))
    for line1 in alines:
        if i<len(alines) and i<len(blines):
            if not (alines[i]=='' or blines[i]==''):
                if not 'http' in (alines[i] or blines[i]):
                    trainc.write(alines[i].strip('\n'))
                    trainc.write('\t')
                    trainc.write(blines[i].strip('\n'))
                    trainc.write('\n')
            i = i + 1
    print("formatted lines: ", i)
    trainc.close()

def main():
    rootdir = Path(dialoguesPath)
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]
    for f in file_list:
        with io.open (f, encoding = 'utf-8') as f:
            lines = []
            foldername = os.path.basename(os.path.dirname(f.name)).lower()
            oppName = foldername.split('(')[0]
            oppNameVkOpt = "/./" + oppName

            lines.append(oppNameVkOpt)
            lines.append('')
            lines.append(myNameVkOpt)
            lines.append('')
            for line in f:
                if not line.startswith('\t'):
                    line = line.rstrip('\n')
                    lines.append(line.strip().lower())
            writeLines(lines, oppNameVkOpt)
    composeFile()

main()
