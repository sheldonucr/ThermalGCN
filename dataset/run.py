import os
import time



for i in range(400):
    ChipletCoreFile = "Chiplet_Core"+str(i)+".flp"
    os.rename(ChipletCoreFile, "Chiplet_Core.flp")
    for j in range(20):
        cmd = "../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p Chiplet_Core"+str(i)+"_Power"+str(j)+".ptrace -steady_file Chiplet.steady  -model_type grid -detailed_3D on -grid_layer_file Chiplet.lcf -grid_steady_file Chiplet.grid.steady"
        tmr_start = time.time()
        os.system(cmd)
        tmr_end = time.time()
        print(tmr_end - tmr_start)

       # cmd = "../grid_thermal_map.pl Chiplet_Core.flp Chiplet.grid.steady 64 64 > Chiplet" + str(i) + str(j) + ".svg"
        
       # os.system(cmd)

        print(i,j)

        
       EdgeFile = "./data/Edge"+"_"+str(i)+"_"+str(j)+".csv"
       os.rename("./data/Edge.csv", EdgeFile)
       TempFile = "./data/Temperature"+"_"+str(i)+"_"+str(j)+".csv"
       os.rename("./data/Temperature.csv", TempFile)
       PowerFile = "./data/Power"+"_"+str(i)+"_"+str(j)+".csv"
       os.rename("./data/Power.csv", PowerFile)


    os.rename("Chiplet_Core.flp", ChipletCoreFile)
    
