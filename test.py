import os
import subprocess
file = open('hello.py')
for line in file:
    if 'Sequential' in line:
        print('sequential')
        subprocess.Popen("sudo docker run -it -v /ajproject:/root -v /Email:/root --name myrpd anantj1/rcnn:v1",shell=True)
        subprocess.Popen("sudo docker exec -d -w /root myrpd -c 'python3 newmail.py'",shell=True)
        #subprocess.Popen("hello.py 1", shell=True)
        #subprocess.Popen("newmail.py 1",shell=True)
        break
    elif 'Regression' in line:
        print('Regression yol')
        break

