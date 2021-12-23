import urllib.request
import sys
import csv
import json
import pickle

class Station():
    def __init__(self, location, id):
        self.location = location
        self.id= id
        self.locations = []
        Stations.append(self)

    def __lt__(self,station2):   # CHECK WHY WE NEED THIS
        return self.id < station2.id
    
Stations = pickle.load( open( "train.sorted_Stations.p", "rb" ) )

# check if origin and destination are in the map
def OSRM_A2B_check(lat1,lon1, lat2,lon2, urlname= 'http://127.0.0.1:5000/'):

    all_url = urlname+'route/v1/walking/'+str(lon1)+","+str(lat1)+";"+str(lon2)+","+str(lat2)+'?steps=false'


    try:
        f = urllib.request.urlopen(all_url)
        d = json.load(f)
    except urllib.error.HTTPError as e:
        if e.code == 400:  # Bad request (origin or detination may be out of the map
            return (False,'1','1')
        else:
            print(e.code)
            print(e.read())
            exit(-1)

    return (True, d['routes'][0]['duration'], d['routes'][0]['distance'])


#filename = '5k_rec'
#filename = sys.argv[1]  # original file name without .csv ext
#fo = open(filename+"o.csv","w")

#num_lines = sum(1 for line in open('test.csv'))

point_in_poly = (40.760476, -73.981660)
B = (40.63568115, -74.08963776)

border_points = [(40.8124,-73.9774,"top_left"),(40.7528,-74.0144,"middle_left"),(40.69245,-74.0285,"bottem_left"),(40.71227,-73.9736,"bottem_right"),(40.79,-73.9272,"top_right")]
#border_points = ((40.8124,-73.9774,"top_left"),(40.79,-73.9272,"top_right"),(40.7528,-74.0144,"middle_left"),(40.69245,-74.0285,"bottem_left"),(40.71227,-73.9736,"bottem_right"))
#lines = {"top_left":["top_right","middle_left"], "top_right":["bottem_right"], "bottem_left":["middle_left","bottem_right"] }

def calc_PolyIneq(lst,point_in_poly):
    PolyIneq = []

    n = len(border_points)
    for i in range(n):
        (y1,x1,ez) = border_points[i]
        (y2,x2,ez) = border_points[(i+1) % n]
        m = (y2- y1)/(x2-x1)
        a = y1 - m*x1
        if m*point_in_poly[1] + a > point_in_poly[0]:
            direc = 1
        else:
            direc = -1
        PolyIneq.append((m,a,direc))

    return PolyIneq


PolyIneq = calc_PolyIneq(border_points,point_in_poly)


def check_polygon(lat, lon):
    for (m,a,direc) in PolyIneq:
        if direc*lat > direc*(m*lon+a):
            return False
        #check if there are not accesable from/to central NY
        if OSRM_A2B_check(lat,lon, 40.760107, -73.983994)[0] < 0 :
            print(lat,lon)
            return False
        if OSRM_A2B_check( 40.760107, -73.983994, lat,lon)[0] < 0 :
            print(lat,lon)
            return False
    return True

print(check_polygon(38.89884949, -77.03943634))


for s in Stations:
    a=[]
    for l in s.locations:
        if check_polygon(l[0],l[1]):
            a.append(l)
    s.locations = a
    
for s in Stations:
    print(len(s.locations) ,"   ", s.id)


#pickle.dump(Stations,open( "train.sorted_Stations.p", "wb" ))
     
      
'''
with open(filename+'.csv', newline='') as f:
    reader = csv.reader(f)
    row0 = next(reader)
    for row1 in reader:
        pickup_long = float(row1[2])
        pickup_lat = float(row1[3])
        dropoff_long = float(row1[4])
        dropoff_lat = float(row1[5])

        check = OSRM_A2B_check(pickup_lat,pickup_long, dropoff_lat,dropoff_long)
        if check[0]:
            if check [1]!= 0:
                if check_polygon(pickup_lat, pickup_long) and check_polygon(dropoff_lat, dropoff_long):
                    print( ",".join(row1))
'''