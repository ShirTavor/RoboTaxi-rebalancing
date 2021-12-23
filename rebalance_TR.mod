/*********************************************
 * OPL 12.10.0.0 Model
 * Author: shirk
 * Creation Date: 29.6.2021
 *********************************************/

 //INPUT
float alpha = ...;
float beta = ...;
float rho = ...;
int n = ...; // number of stations
int planning_horizon = ...; // number of periods at planning horizion
int period_per_hour = ...; // how many periods in 1 hour
int cplex_running_time = ...; //running time limit
range Stations = 0..n-1;
range Horizon = 1..planning_horizon;

float tau [Horizon, Stations , Stations]=...; //time from station i to station j when departing at time t in secondes
int tauT [Horizon, Stations , Stations]=...; //time from station i to station j when arriving at time t in periods

// Dynamic data from sim.dat
float D[Horizon,Stations , Stations ] = ...;
float S[Stations , Horizon ] = ...;


dvar int+ x[Stations, Stations, Horizon] ;
dvar float+ y[Stations, Stations, Horizon] ;
dvar float+ I[Stations,0..planning_horizon] ;
dexpr int car_moved = sum(i in Stations, j in Stations )  x[i,j,1];

execute { cplex.tilim = cplex_running_time;}

minimize sum(t in Horizon, i in Stations, j in Stations)(alpha*tau[t,i,j]*x[i,j,t]+ 
beta*pow(rho,t-1)*(D[t,i,j]-y[i,j,t]));

subject to {


forall (i in Stations, t in Horizon) I[i,t] == I[i,t-1]+S[i,t] -
sum(j in Stations) (y[i,j,t]+x[i,j,t]) +
sum(j in Stations : tauT[t,j,i] < t ) (y[j,i,t-tauT[t,j,i]]+x[j,i,t-tauT[t,j,i]]);


forall(i in Stations, j in Stations, t in Horizon ) D[t,i,j]>= y[i,j,t] ;

forall(i in Stations) I[i,0] == 0;

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


