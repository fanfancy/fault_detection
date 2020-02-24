import linecache
import os
import re

#start and end time

# #heilongjiang
# start  = 673 #840
# end    = 993 #1160
# change = 883 #1000

normal = 1
repeat_times = 36

#IEEE14
if normal:
# Normal
    start  = 1500
    end    = 1900
    change = 1700
else:
    start  = 1800
    end    = 2200
    change = 2000



curdir = os.path.abspath(os.curdir)

# #original data
# data = '/home/zs/data/ieee_whole'
# # data = '/home/zs/data/6TestEtsdac'

# large shorcutresistance data
data = '../data/0416'


loadtypes = [f for f in os.listdir(data) if os.path.isdir(os.path.join(data,f))]

writefile = 'train_data'
suffix = '_400point_large_resistance.txt'

# linenum_all = ['49','269','313','316','75','72','69','66']
linenum_all = ['49']
# linenum_all = ['100002']
faults = {'AG':0,'CG':0,'BG':0,'ABG':1,'ACG':1,'BCG':1,'AC':2,'BC':2,'AB':2,'ABCG':3}
phase = {'A': 0,'B':1,'C':2,'G':3}



for linenum in linenum_all:

    if normal:
        pass
    else:
        if(os.path.exists('./data/'+ linenum +writefile+suffix)):
            os.remove('./data/'+linenum+writefile+suffix)

    # choose fault types of differnet power


    for loadtype in loadtypes:

        absloadtype = os.path.join(data, loadtype)
        dirsnames = os.listdir(absloadtype)
        i = 0
        # print (loadtype)
        '''
        get the fault current and the fault imformation for every fault types,include 360 kinds
        '''
        for dir in dirsnames:
            absdir = os.path.join(absloadtype, dir)
            '''
            choose the line to be processed
            '''
            if os.path.isdir(absdir):
                pattern = re.compile('0?\.?\d+')
                linenums = pattern.findall(dir)
#                 print (dir)
                if (linenums[0] != linenum):
                    continue
                if (float(linenums[-1])>100):
                    continue

                '''
                get the fault current location
                '''
                Simple = open(os.path.join(absdir, 'SimpleVariables.txt'), encoding='latin-1')

                for line in Simple.readlines()[1:]:
                    line = line.strip()
                    line = re.sub(',', ' ', line)
                    line = line.split()
                    if (len(re.findall(r'PiLine-' + linenums[0] + '-J', line[3])) > 0):
                        # print (line[7]," ",line[8])
                        outputfile = line[7]
                        outputcol = line[8]
                        break

                # get the fault name from filename
                faulttypes = re.compile(r'[A-Z]+')
                types = faulttypes.findall(dir)[0]

                # determine the fault type from the name
                whichfault = [0, 0, 0, 0, 0]
                if normal:
                    whichfault[4] = 1
                else:
                    whichfault[faults[types]] = 1

                # determine the fault phases from the name
                whichphase = [0, 0, 0, 0]
                for loc in range(len(types)):
                    whichphase[phase[types[loc]]] = 1

                '''
                write to the file
                '''
                # print (os.path.join(absdir, outputfile))
                for line in linecache.getlines(os.path.join(absdir, outputfile))[start:end]:
                    line = re.sub(',', ' ', line)
                    line = line.split()
                    # print (dir)
                    # print (line)
                    # print (len(line))
                    with open('./data/' + linenum + writefile + suffix, 'a') as f:
                        f.write(
                            line[int(outputcol) ] + ' ' + \
                            line[int(outputcol)+1] + ' ' + \
                            line[int(outputcol) + 2] + ' ' + \
                            str(whichfault[0]) + ' ' + \
                            str(whichfault[1]) + ' ' + \
                            str(whichfault[2]) + ' ' + \
                            str(whichfault[3]) + ' ' + \
                            str(whichfault[4]) + ' ' + \
                            str(whichphase[0]) + ' ' + \
                            str(whichphase[1]) + ' ' + \
                            str(whichphase[2]) + ' ' + \
                            str(whichphase[3]) + ' ' + \
                            str(float(linenums[-2]) / 100.0) + ' '+ \
                            dir + '_' + str(loadtype) + '_' + str(normal) + '\n')
                        # break
                    # break
            #     print (absdir)
                i = i +1
            if normal and i==repeat_times:
                break
