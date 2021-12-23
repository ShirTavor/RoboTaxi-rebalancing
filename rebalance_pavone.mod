/*********************************************
 * OPL 12.10.0.0 Model
 * Author: shirk
 * Creation Date: Sep 19, 2021 at 11:07:01 AM
 *********************************************/

  //INPUT
int n = ...; // number of stations
int cplex_running_time = ...; //running time limit
range Stations = 0..n-1;

float tau [ Stations , Stations]=...; //time from station i to station j when departing at time t in secondes

// Dynamic data from sim.dat
float S[Stations] = ...; // nuber of vehicles ideling at each station 
int v_desired[Stations] = ...; // nuber of desierd vehicles at each station 


dvar int+ x[Stations, Stations] ;
dexpr int car_moved = sum(i in Stations, j in Stations )  x[i,j];

execute { cplex.tilim = cplex_running_time;}



minimize sum(i in Stations, j in Stations)(tau[i,j]*x[i,j]);

subject to {


forall (i in Stations) S[i] +sum(j in Stations) (x[j,i]-x[i,j]) >= v_desired[i];


}


execute { writeln("Number of cars moved:", car_moved);

	var f =new IloOplOutputFile("export.txt");
	f.write("[")
	for(var i in Stations) {
		f.write("[");			
		for (var j in Stations) {
			f.write (x[i][j]); 
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
	
}						

