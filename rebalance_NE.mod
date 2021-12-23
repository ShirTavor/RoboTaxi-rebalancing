/*********************************************
 * OPL 12.10.0.0 Model
 * Author: shirk
 * Creation Date: 29.6.2021
 *********************************************/

 //INPUT
float alpha = ...;
float beta = ...;
float rho = ...;
float gamma = ...; 
int n = ...; // number of stations
int planning_horizon = ...; // number of periods at planning horizion
int period_per_hour = ...; // how many periods in 1 hour
int cplex_running_time = ...; //running time limit
range Stations = 0..n-1;
range Horizon = 1..planning_horizon;


tuple triplets{
  key int orig;
  key int dest;
  key int neighbor;		
}


float tau [Horizon, Stations , Stations]=...; //time from station i to station j when departing at time t in secondes
int tauT [Horizon, Stations , Stations]=...; //time from station i to station j when arriving at time t in periods

{triplets} Triplets = ...;//{<1,2,3>, <1,3,1>}; //{<orig,dest,neighbor>,...}
int Tau3[Horizon, Triplets] =...; // [1,2]; // travel time from i to j + from k to i when arriveng at period t
{int} Neighbors[Stations] =...; //

// Dynamic data from sim.dat
float D[Horizon,Stations , Stations ] = ...;
float S[Stations , Horizon ] = ...;


dvar int+ x[Stations, Stations, Horizon] ;
dvar float+ y[Triplets, Horizon];
dvar float+ I[Stations,0..planning_horizon] ;
dexpr int car_moved = sum(i in Stations, j in Stations )  x[i,j,1];

execute { cplex.tilim = cplex_running_time;}

minimize sum(t in Horizon, i in Stations, j in Stations)(alpha*tau[t,i,j]*x[i,j,t]+ 
pow(rho,t-1)*(beta*(D[t,i,j]-sum(k in Neighbors[i]) y[<i,j,k>,t])+ 
(alpha+gamma)*sum(k in Neighbors[i])tau[t,k,i]*y[<i,j,k>,t]) );

subject to {

forall (i in Stations, t in Horizon) I[i,t] == I[i,t-1]+S[i,t] +
sum(j in Stations, k in Neighbors[j] : Tau3[t,<j,i,k>]< t) y[<j,i,k>,(t-Tau3[t,<j,i,k>])]+
sum(j in Stations: tauT[t,j,i] < t ) x[j,i,t-tauT[t,j,i]] -
sum(j in Stations)(x[i,j,t]+ sum(k in Stations : i in Neighbors[k]) y[<k,j,i>,t]);

forall(i in Stations) I[i,0] == 0;

forall(i in Stations, j in Stations, t in Horizon )
      D[t,i,j]>= sum(k in Neighbors[i]) y[<i,j,k>,t] ;

}

execute { writeln("Number of cars moved:", car_moved);

	var f =new IloOplOutputFile("export.txt");
	f.write("[")
	for(var i in Stations) {
		f.write("[");			
		for (var j in Stations) {
			f.write (x[i][j][1]); 
			if (j < n-1) f.write(",");
		}				
		f.write("]");
		if (i < n-1) f.write(",");			
	}
	f.write("]\n");
	f.close();
	
	var f =new IloOplOutputFile("export2.txt");
	f.writeln("(",cplex.getObjValue(),",",cplex.getBestObjValue(),")");
	f.close();
	
	var f =new IloOplOutputFile("obj_values.txt","a");
	f.writeln("Lower bound = ",cplex.getBestObjValue(), 
	", Objective value = ",cplex.getObjValue(), 
	", gap = ", (cplex.getObjValue()-cplex.getBestObjValue())/cplex.getObjValue());
	f.close();
}						


