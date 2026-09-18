import datetime
import time 
starttime = datetime.datetime.now()
time.sleep(3)
#long running
endtime = datetime.datetime.now()
during = ((endtime -starttime).seconds * 1000 + (endtime -starttime).microseconds / 1000)
print (during)