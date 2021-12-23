/*********************************************
 * OPL 12.10.0.0 Model
 * Author: shirk
 * Creation Date: 22 באפר׳ 2020 at 16:35:15
 *********************************************/

tuple couple { 
	int a ;		// station1 
	int b ;		// station2
}

{couple} two_min = ... ;

tuple station { 
	key int id ;
	float lat ;		 
	float lon ;		
}

{station} stations = ... ;

dvar boolean x[stations];
 

maximize sum (i in stations) x[i];

subject to
{
	forall (cop in two_min)  x[<cop.a>] + x[<cop.b>] <= 1;

}


execute {
  var f =new IloOplOutputFile("export.py")
  
  f.write("station_list = [")
  var flag = true
  for (var  s in stations) {
    if (x[s] > 0.99) 
    {  
    	if (flag) {
    		f.write("(",s.lat,",", s.lon,")");
    	
    		flag = false; 
    		} 
    	else 
    		f.write(",(",s.lat,",", s.lon,")");
    		
    }    		
  }
  f.writeln("]")
  
  f.close();
}
 