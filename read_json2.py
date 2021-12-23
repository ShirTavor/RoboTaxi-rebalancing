import json
import codecs
import urllib.request
import pandas as pd
import numpy

data = json.load(codecs.open('Manhattan.json', 'r', 'utf-8-sig'))["elements"]

print(data[0])



L = []
bus_stop_locations=[]

i = 1

for a in data:
    if 'tags' in a:
        if 'highway' in a['tags'] and a['tags']['highway'] == 'bus_stop':
            L.append((i, a['lat'], a['lon']))
            bus_stop_locations.append((a['lat'], a['lon']))
            i+=1

            
dist_mat = numpy.zeros((len(L), len(L)))
time_mat = numpy.zeros((len(L), len(L)))             

def OSRM_A2B(origin, destination, urlname= 'http://127.0.0.1:5000/'):

    all_url = urlname+'route/v1/walking/'+str(origin[1])+","+str(origin[0])+";"+str(destination[1])+","+str(destination[0])+'?steps=false'
    try:
        f = urllib.request.urlopen(all_url)
    except urllib.error.HTTPError as e:
        if e.code == 400:  # Bad request (origin or detination may be out of the map
            return -1,-1
        else:
            print(e.code)
            print(e.read())
            exit(-1)

    d = json.load(f)
    f.close()
    return d['routes'][0]['duration'], d['routes'][0]['distance']


coupls_with_5_min = []
coupls_with_3_min = []
coupls_with_2_min = []

for i in range(len(L)):
    for j in range(i+1,len(L)):
        dur, dis = OSRM_A2B((L[i][1],L[i][2]),(L[j][1],L[j][2]))
        dist_mat[i][j] = dis
        dist_mat[j][i] = dis
        time_mat[i][j] = dur
        time_mat[j][i] = dur
        
        #if dur <= 5*60:
         #   coupls_with_5_min.append((L[i][0],L[j][0]))
        if dur <= 3*60:
            coupls_with_3_min.append((L[i][0],L[j][0]))
        #if dur <= 2*60:
            #coupls_with_2_min.append((L[i][0],L[j][0]))
            
#print(coupls_with_5_min)
#pd.DataFrame(dist_mat).to_csv("dist_mat.csv")
#pd.DataFrame(time_mat).to_csv("time_mat.csv")

'''
import pandas as pd

df = pd.DataFrame(coupls_with_5_min)
writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
df.to_excel(writer,sheet_name='welcome',index=False)
writer.save()

print (L)
print(coupls_with_5_min)

f = open("5min.txt", "w")
f.write(str(coupls_with_5_min))
f.close()

f = open("3min.txt", "w")
f.write(str(coupls_with_3_min))
f.close()

f = open("bus_stops.txt", "w")
f.write(str(L))
f.close()

f = open("2min.txt", "w")
f.write(str(coupls_with_2_min))
f.close()
'''

f = open("3min.txt", "w")
f.write(str(coupls_with_3_min))
f.close()