"""
# Simulate autonomous car sharin system
#  Version 2 with sampling request from a file

Command line parameters  <Request file name>  <# of cars> <max waiting time for a car> <Sampling_probablity>

The input file is assumed to be a csv file with a list of requests in the following formar
Start time, End Time, # of pax, lon origin, lat origin,  lon destination, lat destination, duration in seconds

The currnet version ignores the number of pax field

By: Shir Tavor and Tal Raviv
Last update: 29/7/20

"""


"""
TODO
   see, https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate

"""

import random
import heapq
import urllib.request
import json
import time
import sys
import datetime
import pickle
import math
import numpy as np
import pandas as pd
import subprocess
import os
import copy


random.seed(0)
np.random.seed(0)

class Event():
    def __init__(self, time, eventType,car =-1, location=-1, returnEventLocation=-1,station = -1,return_station=-1):
        self.time = time # event time
        self.eventType = eventType # type of the event (1 for renting and 2 for return)
        self.car = car # id of the car  (for return event)
        self.location = location # the location of the event
        self.returnEventLocation = returnEventLocation #location of return (for rent event)
        self.station = station #station of request events
        self.return_station = return_station #station of return events
        heapq.heappush(P, self) #enter the event to the events list

    def __lt__(self,event2):
        return self.time < event2.time

    def __repr__(self):
        return f"timr: {self.time} , type: {self.eventType} , car:, {self.car}, location: {self.location}"


class Car():
    def __init__(self, location, availability_time, station = -1):
        self.location = location
        self.availability_time= availability_time
        self.station = station
        self.on_rebalance = False
        
    def __repr__(self):
        return f"location: {self.location} , availibilty: {self.availability_time} , station:, {self.station.id}"
        

class Station():
    def __init__(self, location, id):
        self.location = location
        self.id= id
        self.locations = []
        Stations.append(self)

    def __lt__(self,station2):   # CHECK WHY WE NEED THIS
        return self.id < station2.id

#print when we start the simulation (in real time)
print("Start time")
start_time = datetime.datetime.now()
start_time_num = time.time()
print(start_time)


# Command line parameters
program_name = sys.argv[0].split('/')[-1][:-3]
dataset_name = sys.argv[1] #A csv file containing the trips (without the .csv extention)
file_name = dataset_name+".csv"
NumOfCars =int(sys.argv[2]) #number of cars in the system
Max_time_to_passenger = float(sys.argv[3])/86400  # Maximum waiting time in seconds of a passenger for a car
T= float(sys.argv[4]) #the simulation length (in days)
beta = float(sys.argv[5]) #weight for customers that werent served
planning_horizon = int(sys.argv[6]) #number of periods in the planning horizon
period = float(sys.argv[7])/86400 # period for rebalncing. max value is 3600 (1 hour) #len of period in seconds
cplex_running_time = int(sys.argv[8]) # time in seconds for the time limit for cplex
demand_factor = float(sys.argv[9])  # multiply the demand in the input by this number
Rebalance_on = (cplex_running_time > 0)  # if cplex time is 0 rebalancing operation is disabled
CarSpeed = 2*86400 # Car speed m/days estimate in the city for the filtering stage (actual times are obtained from ORSM)
if len(sys.argv) >10:
    rebalance_method = sys.argv[10].upper() # options - TR= tavor, raviv; P -  pavone et.al 2012, NE - neigbhoorheed model by Tavor and Raviv
else:
    rebalance_method = 'TR'


#optimization parameters
alpha = 1 #weight for re-balancing travel distance
rho = 0.99 #discount factor for customers we don't serve
gamma = 1 #weight for waiting time of the customers 

verbal = False #True if we want to see prints
Max_cars_to_check = 5 # Number of closest cars for which we check the time with OSRM
Cars = [] # list of the Car objects
P = [] #event list (heap organized by time)
period_per_hour = int(1/(period*24))
period_per_day = int(1/period) 
demand_discretization = 24 #the demand is in hours

# Counters for collecting statistics - reported and reset at the end of each day
NumOfTrips = 0
NumOfRejection = 0
total_waiting_time = 0  # in seconds
total_waiting_time_dead_headig = 0  # in seconds
total_waiting_time_car_occupied = 0   # in seconds
total_waiting_time_car_on_rebalance = 0  # in seconds
total_in_vehicle_time = 0  # in seconds
total_rebalancing_trips = 0
total_rebalancing_time = 0  # in seconds
curr_time = 0.0 # simulation clock (days)
lower_bound = 0 # the sum of the lower bounds from the mip 
upper_bound = 0 #the sum of the upper bounds from the mip
rejection_per_period = [0]*period_per_day
rejection_last_period = 0
car_occupied_by_period = [0]*period_per_day
waiting_time_by_min = [0]*int(Max_time_to_passenger*1440+1)
day = 0
rep_lat = 40.75 # manhetten latitude 
rep_lat_rad = math.pi*rep_lat/180 
d_lon = math.acos(math.sin(rep_lat_rad)**2 + math.cos(rep_lat_rad)**2*math.cos(math.pi/180))*6371000 # the distance between two longitud in the simulation area (at lat=40.75)
d_lat = 111195
number_of_neighbors = 6 



# list of station location. took from optimizatiom nodel on OPL CPLEX STUDIO IBM
#station_list = [(40.7571157,-73.9533466),(40.8630768,-73.9301236),(40.86529,-73.931025),(40.7576643,-73.9548433),(40.7423423,-73.9917105),(40.7140311,-73.9923707),(40.7771468,-73.9791346),(40.7862274,-73.9724899),(40.7695248,-73.9819016),(40.7931559,-73.9731304),(40.793315,-73.9673412),(40.7673909,-73.9647658),(40.7483206,-73.969005),(40.7959247,-73.9201249),(40.7049524,-74.0147827),(40.826758,-73.9554676),(40.7079588,-74.0036899),(40.812895,-73.9627159),(40.7128763,-74.0075138),(40.7176275,-74.0035031),(40.7240075,-73.9980765),(40.7307457,-73.992926),(40.7495962,-73.9874539),(40.7643549,-73.9982589),(40.7595161,-73.9813267),(40.7573966,-73.9762614),(40.7436314,-73.9995721),(40.7572755,-73.9669117),(40.7337897,-73.9840195),(40.7546299,-73.9841439),(40.7386007,-73.9730999),(40.7136152,-73.9940579),(40.810335,-73.937116),(40.7217479,-73.9841125),(40.7322906,-73.9965554),(40.738642,-74.0034355),(40.7993038,-73.942926),(40.806715,-73.9556855),(40.7195148,-74.0069797),(40.8182917,-73.9469502),(40.8294309,-73.9484768),(40.7248343,-73.9749744),(40.789991,-73.9396388),(40.7853292,-73.9430481),(40.777386,-73.9488736),(40.7753803,-73.9650045),(40.8362891,-73.9401248),(40.7416203,-73.9749864),(40.7107105,-74.0162606),(40.7569785,-73.9635937),(40.8741875,-73.9092987),(40.7333136,-74.0087128),(40.7408667,-73.9762259),(40.8487988,-73.9370012),(40.8485413,-73.9373446),(40.7314216,-73.9742476),(40.784773,-73.9281533),(40.7015255,-74.0138756),(40.8032913,-73.9325676),(40.7665213,-73.9627389),(40.7164264,-73.9803457),(40.7820712,-73.9601139),(40.7700245,-73.9844772),(40.7542201,-73.9980054),(40.7543756,-73.9991382),(40.8306522,-73.9363675),(40.8036214,-73.969519),(40.7916625,-73.9531281),(40.8065978,-73.9613789),(40.7885669,-73.9490732)]

Stations = pickle.load( open( dataset_name+"_Stations.p", "rb" ) )
demands = pickle.load( open( dataset_name+"_demand.p", "rb" ) )
time_factor = pickle.load( open( "train.sorted_time_factor.p", "rb" ) )#Inflation coefficient for the OSRM times

num_of_stations = len(Stations)


df = pd.read_csv("time_mat_70_stations.csv", delimiter=',')
df = [list(row[1:]) for row in df.values]
tau = np.zeros((period_per_hour*24,num_of_stations,num_of_stations))
tauT = np.zeros((period_per_hour*24,num_of_stations,num_of_stations), dtype=int)
for t in range(24*period_per_hour): 
    for i in range(num_of_stations):
        for j in range(num_of_stations):
            h = t//period_per_hour
            tau[t,i,j]= df[i][j]*time_factor[h,i,j] 
            
for t in range(24*period_per_hour): 
    for i in range(num_of_stations):
        for j in range(num_of_stations):
            if i != j:
                k = 1
                while math.ceil(tau[(t-k)%period_per_day,i,j]/(3600/period_per_hour)) > k:
                    k += 1
                tauT[t,i,j]= k
if rebalance_method == 'NE':
    NeighborsֹOPL = "Neighbors= [\n"
    Tau3 = [ [] for _ in range(period_per_day)]
    Triplets = []
    for i in range(num_of_stations):
        h = []
        for k in Stations[i].top10min:
            heapq.heappush(h,(df[k][i],k))
        neighbors = []
        NeighborsֹOPL += "{"
        while h and len(neighbors) < number_of_neighbors:
            new_neighbor = heapq.heappop(h)[1]
            neighbors.append(new_neighbor)
            NeighborsֹOPL += str(new_neighbor) + " "
        NeighborsֹOPL += "}\n"
        for j in range(num_of_stations):
            for k in neighbors:
                Triplets.append((i,j,k))
                for t in range(period_per_day):
                    new_tau3 = 0
                    while math.ceil((tau[(t-new_tau3)%period_per_day,k,i]+tau[(t-new_tau3)%period_per_day,i,j])/(3600/period_per_hour)) > new_tau3:
                        new_tau3 += 1
                    Tau3[t].append(new_tau3)
    NeighborsֹOPL += "];\n"

#add headline to the file
row = str(["Model Type" ,"file_name" ",rebalance_method","T", "cplex_running_time","demand_factor", "NumOfCars", "Max_time_to_passenger", "Max_cars_to_check", "period" ,"planning_horizon" , "alpha", "beta", "rho", "num_of_stations" , "start_time_num", "start_time", "day" ,"NumOfTrips","NumOfRejection" , "total_waiting_time" ,"total_waiting_time_dead_headig", "total_waiting_time_car_occupied", "total_waiting_time_car_on_rebalance", "total_in_vehicle_time", "total_rebalancing_trips", "total_rebalancing_time", "mean_gap","rejection_per_period","car_occupied_by_period","waiting_time_by_min"]).replace("'","")
f =  open(f'ResultsByDays-{program_name}.csv', 'a')
f.write(row[1:-1]+"\n")
f.close()




# Added by Tal 12.1.19
""" return the manhatan distance between two locations represented by two (lat,lon) tupels. d_lon shound be adjusted for this function to work correctly in area that are not at lat=40.75 """
""" in days """
def estimate_time_to_passenger(car_i,passenger_loc,curr_time):    
    return (abs(Cars[car_i].location[0]-passenger_loc[0])*d_lat + \
    abs(Cars[car_i].location[1]-passenger_loc[1])*d_lon) / CarSpeed  \
    + max(0,Cars[car_i].availability_time-curr_time)



"""
Return the travel, time and duraton between two points in days assuming OSRM is running localy
If serveral server (e.g. driving and walking are driving), override the default urlname, e.g.,  urlname= 'http://127.0.0.1:5001/'
If the server run on other server replace the urlname with the server address and port
"""
def OSRM_A2B(origin, destination, urlname= 'http://127.0.0.1:5000/'):

    all_url = urlname+'route/v1/driving/'+str(origin[1])+","+str(origin[0])+";"+str(destination[1])+","+str(destination[0])+'?steps=false'
    try:
        f = urllib.request.urlopen(all_url)
    except urllib.error.HTTPError as e:
        if e.code == 400:  # Bad request (origin or detination may be out of the map
            print(f'no route found from {origin} to {destination}')
            #raise SystemExit(-1)
            #there is a bug that we cant calculate rout between two very close locations
            if abs(origin[0]-destination[0])*d_lat + abs(origin[1]-destination[1])*d_lon < 150 :
                return 0,0
            #raise SystemExit(-1)
            return -1,-1
        else:
            print(e.code)
            print(e.read())
            raise SystemExit(-1)

    d = json.load(f)
    f.close()
    return d['routes'][0]['duration']/86400, d['routes'][0]['distance']



# TODO: make the distance matrix, a parameter  or calculate here
def create_initial_dat_file():
        
    file1 = open("basic_data.dat" ,"w") #write mode
    file1.write("n = " + str(num_of_stations) +"; \n")
    file1.write("cplex_running_time = %d;\n" % cplex_running_time)
    if rebalance_method in ['TR', 'NE']:
        file1.write("alpha = " + str(alpha) +"; \n")
        file1.write("beta = " + str(beta) +"; \n")
        file1.write("rho = " + str(rho) +"; \n")
        file1.write("planning_horizon = " + str(planning_horizon) +"; \n")
        file1.write("period_per_hour = %d;\n" % period_per_hour )
        if rebalance_method == 'NE':
            file1.write("gamma = " + str(gamma) +"; \n")
            file1.write(NeighborsֹOPL + "\n")
    file1.close()



# This function creates input for the OPL model, used for the original TR model and for the neighbors model, launches it and returns the optimal rebalance plan for the next period
def SolveRebalanceModel_TR():
    file1 = open("sim.dat","w")#append mode

    for station in Stations:
        station.availble_cars = [0]*planning_horizon

    for car in Cars:
        station = car.station
        slot = max(math.ceil((car.availability_time - curr_time)/period) , 0)
        if slot < planning_horizon:
            station.availble_cars[int(slot)] += 1


    file1.write( "S =[ \n")
    for i in range(num_of_stations) :
        file1.write( str(Stations[i].availble_cars) + "\n")
    file1.write(  "]; \n")
        
    #make sure numpy matrix printed correcty 
    np.set_printoptions(threshold=sys.maxsize)
    file1.write( "D =[")
    for t in range(planning_horizon):
        hour = math.floor(((curr_time + t*period)%1)*24)%24
        file1.write( str(demand_factor*demands[hour,:,:]/period_per_hour)+" \n")
    file1.write("]; \n")

    
    starting_period = math.floor((curr_time%1)*period_per_day)
    file1.write( "tau =[")
    for t in range(planning_horizon):
        file1.write( str(tau[(starting_period+t)%period_per_day,:,:])+" \n")
    file1.write("]; \n")

    
    file1.write( "tauT =[")
    for t in range(planning_horizon):
        file1.write( str(tauT[(starting_period+t)%period_per_day,:,:])+" \n")
    file1.write("]; \n")

    
    if rebalance_method == 'NE':
        file1.write( "Triplets ={")
        for (i,j,k) in Triplets:
            file1.write(f'<{i} {j} {k}>\n')
        file1.write( "}; \n")
        
        file1.write( f"Tau3 = {Tau3[starting_period:starting_period+planning_horizon]};\n")            
        '''    
        file1.write( "Fourlets ={")
        for t in range(planning_horizon):
            for (i,j,k) in Triplets:
                file1.write(f'<{i} {j} {k} {t+1}>\n')
        file1.write( "}; \n")
        
        file1.write( "Tau3 =[")    
        for t in range(planning_horizon):
            for a in Tau3[starting_period + t]:
                file1.write(f'{a} ')
            file1.write("\n")
        file1.write("]; \n")
        '''
    
    file1.close() 
    
    if os.path.exists("export.txt"):
        os.remove("export.txt")  # We delete this file to make sure that an error will occur if Cplex fails

    if rebalance_method == 'TR':
        subprocess.run(["oplrun", "rebalance_TR.mod", "basic_data.dat", "sim.dat"], check=True)
    else:
        subprocess.run(["oplrun", "rebalance_NE.mod", "basic_data.dat", "sim.dat"], check=True)
    f = open("export.txt")
    s = f.read()
    x = eval(s)
    f.close()
    f = open("export2.txt")
    s = f.read()
    bound = eval(s)
    f.close()
    return (x, bound)
    
    
    


# This function creates input for the OPL model, launches it and returns the optimal rebalance plan for the next period
def SolveRebalanceModel_P():
    file1 = open("sim.dat","w")#append mode

    for station in Stations:
        station.availble_cars = 0
    
    total_num_of_idle_cars = 0
    for car in Cars:
        station = car.station
        if car.availability_time <= curr_time:
            station.availble_cars += 1
            total_num_of_idle_cars+= 1
            
    v_desired = [int(total_num_of_idle_cars/num_of_stations)]*num_of_stations
    
    file1.write( f"v_desired ={v_desired};\n")

    np.set_printoptions(threshold=sys.maxsize)
    file1.write( "S =[ \n")
    for i in range(num_of_stations) :
        file1.write( str(Stations[i].availble_cars) + " ")
    file1.write(  "]; \n")
            
    starting_period = math.floor((curr_time%1)*period_per_day)
    file1.write( "tau =")
    file1.write( str(tau[(starting_period),:,:])+" \n")
    file1.write("; \n")
    file1.close()
    
    if os.path.exists("export.txt"):
        os.remove("export.txt")  # We delete this file to make sure that an error will occur if Cplex fails

    subprocess.run(["oplrun", "rebalance_pavone.mod", "basic_data.dat", "sim.dat"], check=True)
    f = open("export.txt")
    s = f.read()
    x = eval(s)
    f.close()
    f = open("export2.txt")
    s = f.read()
    bound = eval(s)
    f.close()
    return (x, bound)

def FindStation(EventLocation):
    close_stations = []
    for i in Stations:
        dist_to_station = abs(i.location[0]-EventLocation[0])*111195 + \
    abs(i.location[1]-EventLocation[1])*d_lon 
        heapq.heappush(close_stations, (dist_to_station,i))
    min_dist = 9999999
    count_stations = 0
    while count_stations < 5:
        (d,s) = heapq.heappop(close_stations)
        if d <= 50 : # the car is exactly at the station
            return s

        dist_to_sation_OSRM = OSRM_A2B(EventLocation,s.location)[1]

        if dist_to_sation_OSRM < min_dist:
            min_dist = dist_to_sation_OSRM
            close_station = s
        count_stations += 1
    return close_station


#A = (40.71508026, -74.00964355)
##B = (40.63568116, -74.08963774)
#
#B = (40.63568115, -74.08963776)
#
#print(OSRM_A2B(A,B))
#print(OSRM_A2B(B,A))
#print(FindStation(B).id)


#raise SystemExit(0)

# Returns the index of the closest car availible, waiting time in days(deadheading, occuipeid, on rebalance), trip duration in days
def FindCar(curr_time, event):
    close_cars = []
    for i in range(len(Cars)):
        time_to_passenger = estimate_time_to_passenger(i, event.location, curr_time)
        if time_to_passenger < Max_time_to_passenger*2:
            heapq.heappush(close_cars, (time_to_passenger,i))

    min_waiting = 9999999
    car = None
    h =  int(round((curr_time%1)*24)%24)
    time_to_dest = OSRM_A2B(event.location,event.returnEventLocation)[0]*time_factor[h,event.station,event.return_station]

    count_cars = 0
######## fix time_factor at the availbilty time and not curr time
    while  close_cars and ((count_cars < Max_cars_to_check) or car==None):
        (d,i) = heapq.heappop(close_cars)
        waiting_time_to_car = OSRM_A2B(Cars[i].location ,event.location)[0]* time_factor[h,Cars[i].station.id,event.station] + max(0,Cars[i].availability_time - curr_time)
        
        if waiting_time_to_car < 0 :
            waiting_time_to_car = OSRM_A2B( event.location, Cars[i].location)[0]* time_factor[h,Cars[i].station.id,event.station] + max(0,Cars[i].availability_time - curr_time)
            if waiting_time_to_car < 0 :
                print(f'no route found from {event.location} to {Cars[i].location}')
                raise SystemExit(-1)
        #if waiting_time_to_car < min_waiting and waiting_time_to_car <= Max_time_to_passenger and waiting_time_to_car >= 0:
        if waiting_time_to_car < min_waiting and waiting_time_to_car <= Max_time_to_passenger:
            min_waiting = waiting_time_to_car
            car = i
        count_cars += 1

    if car != None :
        if int(min_waiting*1440) >= len(waiting_time_by_min) or int(min_waiting*1440) < 0 : 
            print(min_waiting)
            print(Cars[car].location)
            print(event.location)
        waiting_time_by_min[int(min_waiting*1440)] += 1
        
        if Cars[car].on_rebalance:
            return int(car), (min_waiting ,0, max(0,Cars[car].availability_time - event.time)), time_to_dest
        else:
            return int(car), (min_waiting , max(0,Cars[car].availability_time - event.time),0), time_to_dest
    else:
#        print (f'len close cars is {len(close_cars)}')
#        print(f'event location is {event.location}')
#        for i in range(len(Cars)):
#            time_to_passenger = estimate_time_to_passenger(i, event.location, curr_time)
#            print(f' locatio of car {i} is {Cars[i].location} and distance to event is {time_to_passenger}')
        return None, None, None


def Lottery_Car_Locations(n):
    Cars = []
    for i in range(n):
        s = random.randint(0,num_of_stations-1)
        Cars.append(Car(Stations[s].location,0,Stations[s]))

    return Cars


def new_period():
    global upper_bound
    global lower_bound
    hour = math.floor((curr_time % 1)*24)
    print(hour)
    for origin in range(num_of_stations):
        for dest in range(num_of_stations):
            event_amount = np.random.poisson((period*24)*demands[hour][origin][dest]*demand_factor)
            if event_amount:
                event_list = curr_time + np.sort(np.random.rand(event_amount))/(24*period_per_hour)

                for i in range(event_amount):
                    orig_location = random.choice( Stations[origin].locations)
                    dest_location = random.choice( Stations[dest].locations)
                    Event(event_list[i],"request",-1,orig_location ,dest_location ,origin , dest)

    Event(curr_time + period,"new_period")

    if Rebalance_on == True:
        if rebalance_method in ['TR','NE']:
            x, bound = SolveRebalanceModel_TR()
        else:
            x, bound = SolveRebalanceModel_P()  
        upper_bound += bound[0]
        lower_bound += bound[1]

        #rebalance the system
        global total_rebalancing_time
        global total_rebalancing_trips
        for origin in range(num_of_stations):
            for dest in range(num_of_stations):
                amount = x[origin][dest]
                if amount > 0 : #check if there are cars to move from origin to destenation
                    close_cars = []
                    for i in range(len(Cars)):
                        if Cars[i].availability_time <= curr_time:
                            if Cars[i].station.id == origin:
                                time_to_dest_station = estimate_time_to_passenger(i, Stations[dest].location, curr_time)
                                heapq.heappush(close_cars, (time_to_dest_station,i))
                    total_rebalancing_trips += amount
                    for i in range(amount):
                        if close_cars:
                            (d,car) = heapq.heappop(close_cars)
                            travel_time, dist = OSRM_A2B(Cars[car].location,Stations[dest].location)
                            travel_time *= time_factor[hour,Cars[car].station.id,dest]
                            total_rebalancing_time += travel_time
                            Cars[int(car)].location = Stations[dest].location
                            Cars[int(car)].station = Stations[dest]
                            Cars[int(car)].availability_time = curr_time + travel_time
                            Cars[int(car)].on_rebalance = True
                            Event(Cars[int(car)].availability_time,"end_rebalance_for_car", car)

def update_stats():
    global rejection_last_period
    global rejection_per_period
    global car_occupied_by_period
    
    slot = math.floor(curr_time/period)%96
    rejection_per_period [slot] += rejection_last_period 
    rejection_last_period = 0
    num1 = 0
    for car in Cars:
        if car.availability_time > curr_time:
            num1 += 1
    car_occupied_by_period [slot] = num1
    Event(curr_time+ period,"update_stats")
    

Cars = Lottery_Car_Locations(NumOfCars)

if Rebalance_on:
    create_initial_dat_file()
    
#raise SystemExit(0)

Event(0 ,"new_period")
Event(1 ,"new_day")
Event(period,"update_stats")

while curr_time <= T :

    event = heapq.heappop(P)
    curr_time = event.time

    curr_day = curr_time//1

    if event.eventType == "request" :  # rent request event
        car , waiting_time, in_vehicle_time = FindCar(curr_time, event)
        #if verbal:
        #    print (NearestCars)
        if car != None:
            total_duration = waiting_time[0] + in_vehicle_time
            createReturnEvent = Event(event.time + total_duration,"return",car,event.returnEventLocation,())
            NumOfTrips += 1
            total_waiting_time += waiting_time[0] 
            total_waiting_time_dead_headig += (waiting_time[0] - waiting_time[1] - waiting_time[2])
            total_waiting_time_car_occupied += waiting_time[1]
            total_waiting_time_car_on_rebalance += waiting_time[2]
            total_in_vehicle_time += in_vehicle_time
            if verbal:
                print(time.ctime(event.time), " Car", car, "rented from location ", Cars[int(car)])
            Cars[int(car)].location = event.returnEventLocation
            Cars[int(car)].station = FindStation(event.returnEventLocation)
            Cars[int(car)].availability_time = curr_time + total_duration

        else:
            if verbal:
                print (time.ctime(event.time)," No Availble Car")
            
            NumOfRejection += 1
            rejection_last_period +=1

    elif event.eventType == "return" :   # return event
        if verbal:
            print ("Time:",time.ctime(event.time)," Car", event.car, "returned to location ", Cars[event.car].location)


    elif event.eventType == "new_period" :   # return event
        new_period()
        if verbal:
            print ("********* Time:",time.ctime(event.time),"creation of new requests for the next period")
            
    elif event.eventType == "update_stats" :  
        update_stats()

    elif event.eventType == "new_day":  # update the files
        if Rebalance_on == True:
            mean_gap = (upper_bound-lower_bound)/upper_bound
        else:
            mean_gap = 0
        row = str([program_name ,file_name ,rebalance_method,T, cplex_running_time,demand_factor, NumOfCars, Max_time_to_passenger, Max_cars_to_check, period ,planning_horizon , alpha, beta, rho, num_of_stations , start_time_num, str(start_time), day ,NumOfTrips,NumOfRejection , total_waiting_time,total_waiting_time_dead_headig,total_waiting_time_car_occupied, total_waiting_time_car_on_rebalance, total_in_vehicle_time, total_rebalancing_trips, total_rebalancing_time, mean_gap,str(rejection_per_period),str(car_occupied_by_period),str(waiting_time_by_min)]).replace("'","")
        f =  open(f'ResultsByDays-{program_name}.csv', 'a')
        print(row[1:-1])
        f.write(row[1:-1]+"\n")
        f.close()
        NumOfTrips = 0
        NumOfRejection = 0
        total_waiting_time = 0
        total_waiting_time_dead_headig = 0
        total_waiting_time_car_occupied =0
        total_waiting_time_car_on_rebalance = 0
        total_in_vehicle_time = 0
        total_rebalancing_trips = 0
        total_rebalancing_time = 0
        upper_bound = 0
        lower_bound = 0
        rejection_per_period = [0]*period_per_day
        car_occupied_by_period = [0]*period_per_day
        waiting_time_by_min = [0]*int(Max_time_to_passenger*1440+1)
        day = curr_day
        Event(day + 1 ,"new_day")
        print (curr_time)


    elif event.eventType == "end_rebalance_for_car" :   # return event
        Cars[int(event.car)].on_rebalance = False


