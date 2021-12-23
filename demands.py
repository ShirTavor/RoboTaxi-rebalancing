# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:35:02 2020

@author: shirk


with 2 min (160 stations) bitween stations and 5 min max to station - 28,637 trips
with 3 min (70 stations) bitween stations and 5 min max to station - 28,644 trips

with 2 min (160 stations) bitween stations and 3 min max to station - 28,505 trips
with 3 min (70 stations) bitween stations and 3 min max to station - 28,294 trips

with 3 min (70 stations) bitween stations and 3 min max to station - 760,949 trips

"""


import heapq
import urllib.request
import json
import time
import sys
import datetime
import numpy
import pandas as pd
import pickle
import queue
import numpy as np
import math



class Station():
    def __init__(self, location, id):
        self.location = location
        self.id= id
        self.locations = []
        Stations.append(self)
        
    def __lt__(self,station2):
        return self.id < station2.id


#print when we start the simulation (in real time)
print("Start time")
start_time = datetime.datetime.now()
print(start_time)

# Command line parameters
dataset_name = sys.argv[1] #A csv file containing the trips (without the .csv extention)
file_name = dataset_name+".csv"
#Sampling_probablity = float(sys.argv[4])  # The probablity of sampling each request

verbal = False #True if we want to see prints
warmup_time = 6 * 3600
cooldown_time = 0
Stations = [] #list of station
#max_dist_to_station = 500 # If there is a station far from the specified distance we will not refer to the event
max_time_to_station = 3*60
NumberOfDays = 0
NumberOfTrips = 0
cur_year = 0
cur_mon = 0
cur_day = 0

# list of station location. took from optimizatiom nodel on OPL CPLEX STUDIO IBM
# with 3 minuts coupls(70 stations)
station_list = [(40.7571157,-73.9533466),(40.8630768,-73.9301236),(40.86529,-73.931025),(40.7576643,-73.9548433),(40.7423423,-73.9917105),(40.7140311,-73.9923707),(40.7771468,-73.9791346),(40.7862274,-73.9724899),(40.7695248,-73.9819016),(40.7931559,-73.9731304),(40.793315,-73.9673412),(40.7673909,-73.9647658),(40.7483206,-73.969005),(40.7959247,-73.9201249),(40.7049524,-74.0147827),(40.826758,-73.9554676),(40.7079588,-74.0036899),(40.812895,-73.9627159),(40.7128763,-74.0075138),(40.7176275,-74.0035031),(40.7240075,-73.9980765),(40.7307457,-73.992926),(40.7495962,-73.9874539),(40.7643549,-73.9982589),(40.7595161,-73.9813267),(40.7573966,-73.9762614),(40.7436314,-73.9995721),(40.7572755,-73.9669117),(40.7337897,-73.9840195),(40.7546299,-73.9841439),(40.7386007,-73.9730999),(40.7136152,-73.9940579),(40.810335,-73.937116),(40.7217479,-73.9841125),(40.7322906,-73.9965554),(40.738642,-74.0034355),(40.7993038,-73.942926),(40.806715,-73.9556855),(40.7195148,-74.0069797),(40.8182917,-73.9469502),(40.8294309,-73.9484768),(40.7248343,-73.9749744),(40.789991,-73.9396388),(40.7853292,-73.9430481),(40.777386,-73.9488736),(40.7753803,-73.9650045),(40.8362891,-73.9401248),(40.7416203,-73.9749864),(40.7107105,-74.0162606),(40.7569785,-73.9635937),(40.8741875,-73.9092987),(40.7333136,-74.0087128),(40.7408667,-73.9762259),(40.8487988,-73.9370012),(40.8485413,-73.9373446),(40.7314216,-73.9742476),(40.784773,-73.9281533),(40.7015255,-74.0138756),(40.8032913,-73.9325676),(40.7665213,-73.9627389),(40.7164264,-73.9803457),(40.7820712,-73.9601139),(40.7700245,-73.9844772),(40.7542201,-73.9980054),(40.7543756,-73.9991382),(40.8306522,-73.9363675),(40.8036214,-73.969519),(40.7916625,-73.9531281),(40.8065978,-73.9613789),(40.7885669,-73.9490732)]

#with 2 minuts coupls (160 stations)
#station_list = [(40.7571157,-73.9533466),(40.7454339,-73.9775952),(40.8573751,-73.9324549),(40.86529,-73.931025),(40.7576643,-73.9548433),(40.7835092,-73.9645896),(40.7140311,-73.9923707),(40.7751962,-73.9823807),(40.773827,-73.9815527),(40.7792227,-73.9775993),(40.7695248,-73.9819016),(40.7843171,-73.977163),(40.8010783,-73.9681125),(40.7939724,-73.9631631),(40.8015734,-73.9613037),(40.7673909,-73.9647658),(40.7286421,-74.002565),(40.7575213,-73.9902811),(40.7959247,-73.9201249),(40.7499291,-73.9688781),(40.7388494,-74.0063977),(40.826758,-73.9554676),(40.7466936,-73.9938486),(40.812024,-73.9607387),(40.7100901,-74.0112633),(40.7099415,-74.0113816),(40.7185463,-74.0027133),(40.7205751,-74.0010102),(40.7216454,-74.0001096),(40.7251336,-73.9971484),(40.7314134,-73.9923393),(40.8079375,-73.9635988),(40.7645545,-73.9916141),(40.7656291,-73.9973643),(40.7534331,-73.9669226),(40.7595161,-73.9813267),(40.762787,-73.9822007),(40.7652046,-73.9804255),(40.7636976,-73.9712017),(40.7494061,-73.9909692),(40.7905087,-73.9452994),(40.7157702,-74.0150924),(40.7353404,-74.0059326),(40.7551893,-73.9911301),(40.7745799,-73.959276),(40.7789234,-73.9560922),(40.749811,-73.9773744),(40.7511506,-73.9763815),(40.7086881,-74.0110824),(40.7256902,-74.0038242),(40.7244711,-74.0042985),(40.7378835,-73.9963057),(40.7584288,-73.9814041),(40.7464781,-73.990067),(40.7546299,-73.9841439),(40.7545084,-73.9842073),(40.7118883,-74.0069035),(40.7386007,-73.9730999),(40.7136152,-73.9940579),(40.8084262,-73.9385408),(40.8104757,-73.9433017),(40.7196452,-73.9784972),(40.7218715,-73.9803463),(40.7221236,-73.9935164),(40.7253928,-73.986739),(40.7243938,-73.9874634),(40.7325494,-74.0035717),(40.7362481,-73.997517),(40.8003573,-73.9468327),(40.806715,-73.9556855),(40.7517215,-73.9822981),(40.7518902,-73.98215),(40.7335852,-74.003263),(40.7195148,-74.0069797),(40.7390034,-73.9768941),(40.7410324,-73.9814417),(40.7420003,-73.9972565),(40.7369949,-73.9885376),(40.7413096,-73.9898667),(40.8039794,-73.9354847),(40.8092072,-73.9442011),(40.8068408,-73.9501215),(40.8255617,-73.9435272),(40.7922386,-73.9715491),(40.7928805,-73.9728758),(40.8274285,-73.9353946),(40.8423175,-73.9421168),(40.7248343,-73.9749744),(40.7746385,-73.9508342),(40.7873339,-73.941563),(40.788921,-73.9403039),(40.7650171,-73.9575242),(40.7711518,-73.953415),(40.7992936,-73.9328295),(40.7961067,-73.9385614),(40.7589986,-73.9656437),(40.7815422,-73.9492403),(40.7634037,-73.9624458),(40.7659025,-73.9606375),(40.8021334,-73.934323),(40.7914774,-73.9792924),(40.8150246,-73.936136),(40.834699,-73.9478694),(40.7892719,-73.9462213),(40.8140619,-73.9495226),(40.7285967,-73.9879396),(40.7701006,-73.9542102),(40.7371351,-73.978253),(40.7118757,-73.9993566),(40.8471075,-73.9354938),(40.7084505,-74.0171071),(40.7569785,-73.9635937),(40.7444556,-74.0029665),(40.8741875,-73.9092987),(40.8644627,-73.9186357),(40.7050694,-74.0077758),(40.8198221,-73.955624),(40.7475615,-74.0006876),(40.7349336,-73.9796162),(40.7955301,-73.9329672),(40.8532941,-73.9272165),(40.8485305,-73.9376879),(40.8487988,-73.9370012),(40.8252275,-73.9541245),(40.7234922,-73.9881611),(40.7119959,-74.0064481),(40.7196378,-73.9877744),(40.7144908,-73.9813626),(40.8590019,-73.9342546),(40.7297087,-73.9915037),(40.7964884,-73.9482321),(40.784773,-73.9281533),(40.7492441,-73.9693383),(40.7015255,-74.0138756),(40.7668893,-73.960055),(40.7969463,-73.9429879),(40.8161102,-73.9611177),(40.7815891,-73.9813462),(40.7934016,-73.949529),(40.7893492,-73.9524855),(40.7836786,-73.9589387),(40.7706933,-73.9684232),(40.7700245,-73.9844772),(40.7571434,-74.0007973),(40.7542201,-73.9980054),(40.7543756,-73.9991382),(40.8046165,-73.9413735),(40.7617206,-73.9686851),(40.7730921,-73.9825329),(40.7200047,-74.0055688),(40.8306522,-73.9363675),(40.8018257,-73.9457696),(40.8039041,-73.9690228),(40.8078885,-73.9604231),(40.7831116,-73.953079),(40.7843456,-73.9450187),(40.7883888,-73.9531952),(40.7320956,-73.9846215),(40.7294626,-73.9778674),(40.7409453,-73.9884445)]

# a 3D matrix of the demand for each hour of the day
demands = numpy.zeros((24,len(station_list),len(station_list)))
duration_real = numpy.zeros((24,len(station_list),len(station_list)))
duration_osrm = numpy.zeros((24,len(station_list),len(station_list)))

#initialize station objects
i=0
for s in station_list:
    Station(s,i)
    i+=1


"""
Return the travel, time and duraton between two points assuming OSRM is running localy
If serveral server (e.g. driving and walking are driving), override the default urlname, e.g.,  urlname= 'http://127.0.0.1:5001/'
If the server run on other server replace the urlname with the server address and port
"""
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

def FindStation(EventLocation):
    close_stations = []
    for i in Stations:
        dist_to_station = abs(i.location[0]-EventLocation[0])*111195 + \
    abs(i.location[1]-EventLocation[1])*84237
        heapq.heappush(close_stations, (dist_to_station,i))
    min_time = 9999999
    count_stations = 0
    while count_stations < 5:
        (d,s) = heapq.heappop(close_stations)
        dist_to_sation_OSRM = OSRM_A2B(EventLocation,s.location)[0]

        if dist_to_sation_OSRM < min_time:
            min_time = dist_to_sation_OSRM
            close_station = s
        count_stations += 1
    return close_station , min_time

def FindNeighbor(Station):
    close_stations = []
    for i in Stations:
        dist_to_station = abs(i.location[0]-Station.location[0])*111195 + \
    abs(i.location[1]-Station.location[1])*84237
        heapq.heappush(close_stations, (dist_to_station,i))
    top10min = []
    count_stations = 0
    time_to_sation_OSRM = 0
    while time_to_sation_OSRM < 10*60 and close_stations:
        (d,s) = heapq.heappop(close_stations)
        time_to_sation_OSRM , dist_to_sation_OSRM = OSRM_A2B(Station.location,s.location)
        heapq.heappush(top10min, (dist_to_sation_OSRM,s))
        count_stations += 1
          
    a = queue.Queue(maxsize=70)   
    while len(top10min) != 0 :
        (d,s) = heapq.heappop(top10min)
        a.put(s.id)
    Station.top10min = list(a.queue)

with open(file_name, newline='') as f:
    row1 = f.readline()
    while row1:

            row1 = row1.strip().split(',')

            pickup_time_struc = time.strptime(row1[0], "%d/%m/%Y %H:%M")
            if pickup_time_struc.tm_wday <= 5:
            # check if we are on a week day, and not on a weekend
                pickup_time = time.mktime(pickup_time_struc)
                pickup_long = float(row1[3])
                pickup_lat = float(row1[4])
                dropoff_long = float(row1[5])
                dropoff_lat = float(row1[6])
                duration = float(row1[7])

                origin_station , time1 = FindStation((pickup_lat,pickup_long))
                origin_station.locations.append((pickup_lat,pickup_long))
                dest_station , time2 = FindStation((dropoff_lat, dropoff_long))
                dest_station.locations.append((pickup_lat,pickup_long))

                if time1 <= max_time_to_station and time2 <= max_time_to_station:
                    trip_hour = pickup_time_struc.tm_hour
                    demands[trip_hour,origin_station.id,dest_station.id] += 1
                    osrm_duration = OSRM_A2B((pickup_lat,pickup_long),(dropoff_lat,dropoff_long))[0]
                    if duration <= 10*osrm_duration :
                        duration_real[trip_hour,origin_station.id,dest_station.id] += duration
                        duration_osrm[trip_hour,origin_station.id,dest_station.id] += osrm_duration 
                    NumberOfTrips += 1

                    if pickup_time_struc.tm_year != cur_year or pickup_time_struc.tm_mon != cur_mon or pickup_time_struc.tm_mday != cur_day:
                        cur_year = pickup_time_struc.tm_year
                        cur_mon = pickup_time_struc.tm_mon
                        cur_day = pickup_time_struc.tm_mday
                        NumberOfDays += 1
            row1 = f.readline()


          



for station in Stations:
    FindNeighbor(station)
            
demands/=NumberOfDays

dist_mat = numpy.zeros((len(station_list), len(station_list)))
time_mat = numpy.zeros((len(station_list), len(station_list)))     

def create_station_mats():
    for i in range(len(station_list)):
        for j in range(i+1,len(station_list)):
            dur, dis = OSRM_A2B((Stations[i].location),(Stations[j].location))
            dist_mat[i][j] = dis
            dist_mat[j][i] = dis
            time_mat[i][j] = dur
            time_mat[j][i] = dur
            
    pd.DataFrame(dist_mat).to_csv("dist_mat_70_stations.csv")
    pd.DataFrame(time_mat).to_csv("time_mat_70_stations.csv")


create_station_mats()

time_factor = duration_real/duration_osrm

for h in range(24):
    for o in range(70):
        for d in range(70):
            if time_factor[h,o,d] < 1:
                time_factor[h,o,d] = 1
            if time_factor[h,o,d] == numpy.inf:
                time_factor[h,o,d] = 1
            if math.isnan(time_factor[h,o,d]):
                if not math.isnan(time_factor[(h+1)%24,o,d]) and not math.isnan(time_factor[(h-1)%24,o,d]):
                    time_factor[h,o,d] = (time_factor[(h+1)%24,o,d]+time_factor[(h-1)%24,o,d])/2
    time_factor[h] = numpy.where(numpy.isnan(time_factor[h]),numpy.average(time_factor[h][~numpy.isnan(time_factor[h])]),time_factor[h])

                
                
            



pickle.dump( time_factor, open( dataset_name+"_time_factor.p", "wb" ) )
pickle.dump( demands, open( dataset_name+"_demand.p", "wb" ) )
pickle.dump( Stations, open( dataset_name+"_Stations.p", "wb" ) )

