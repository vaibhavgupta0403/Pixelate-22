import gym
import pixelate_arena
import time
import pybullet as p
import pybullet_data
import cv2
import numpy as np
from cv2 import aruco
import math

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
size = 13
gr=[[] for i in range(13)]
allcosts=[]
sta=[]
ver = []
old={}
numrow=[7,8,9,10,11,12,13,12,11,10,9,8,7]
imap=[[] for i in range(13)]
c = np.zeros((127, 6), int)
env = gym.make("pixelate_arena-v0")
# time.sleep(2)
circ1,circ2=[],[]
tria1,tria2=[],[]
sq1,sq2=[],[]
# allcenters=[]
redx,redy,pinkx,pinky=[],[],[],[]
rowsta=[]
cellpos=[]
rowsta.append(0)
for i in range(len(numrow)):
      rowsta.append(rowsta[len(rowsta)-1]+numrow[i])

def getContours(img, imgblur,img2,z,co):
    x = y = w = h = 0
    centerx, centery= ([] for i in range(2))
    global ond, t, q,tria1,circ1,tria2,circ2,sq1,sq2,redx,redy,pinkx,pinky
    cont, a = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for con in cont:
        area = cv2.contourArea(con)
        peri=cv2.arcLength(con,True)
        ep = 0.048 * cv2.arcLength(con, True) - 0.05
        app = cv2.approxPolyDP(con, ep, True)
        x, y, w, h = cv2.boundingRect(app)
        
        if len(app)==6 and area<3000 and z==0 and co!=5:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            centerx.append(x + w / 2)
            centery.append(y + h / 2)
            # allcenters.append((y + h / 2,x+w/2))
            # print(area)
        elif len(app) == 4 and co==5:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if z==0:
              if abs(peri**2/area-16)<0.5:
                 sq1.append((x + w / 2,y+h/2))
              else:
                  tria1.append((x + w / 2,y+h/2))
            else:
              if abs(peri**2/area-16)<0.5: 
                 sq2.append((x+w/2,y+h/2))
              else:
                 tria2.append((x + w / 2,y+h/2))
            # print(area)
        elif len(app) == 3:
             cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)
             if z==0:
               tria1.append((x+w/2,y+h/2))
             else:
               tria2.append((x+w/2,y+h/2))
        elif len(app) > 4:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if z==0:
              circ1.append((x+w/2,y+h/2))
            else:
              circ2.append((x+w/2,y+h/2))
    if(co==6):
       redx=centerx
       redy=centery
    if co==7:
       pinkx=centerx
       pinky=centery
    return centerx, centery,img2

def grcell(x):
    ind=(-1,-1)
    for i in range(len(rowsta)-1):
       if x>=rowsta[i] and x<rowsta[i+1]:
          ind=(i,x-rowsta[i])
    if ind[0]==-1:
        ind=(len(rowsta)-1,x-rowsta[len(rowsta)-1])
    return ind

def gen():
  global sta,imap,cellpos
  img=env.camera_feed()
  img2=img.copy()
  x=7
  for i in range(6):
    for j in range(x):
       gr[i].append(0)
    x+=1
  for i in range(6,13):
      for j in range(x):
         gr[i].append(0)
      x-=1
  ccell=0
  x=7
  for i in range(6):
      s=166-22.5*i
      for j in range(x):
         imap[i].append((s+j*45,66.5+i*39))
         cellpos.append((imap[i][j][1],imap[i][j][0]))
         ccell+=1
      sta.append(s)
      x+=1

  for i in range(6,13):
      s=31+22.5*(i-6)
      for j in range(x):
         imap[i].append((s+j*45,66.5+i*39))
         cellpos.append((imap[i][j][1],imap[i][j][0]))
         ccell+=1
      sta.append(s)
      x-=1
  # print(imap)

gen()

def color(lower, upper, lower2, upper2, img, cost, t,co):
    y = []
    z = []
    img2=img.copy()
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(imghsv, lower, upper)
    mask2 = cv2.inRange(imghsv, lower2, upper2)
    mask = mask1 + mask2
    kern1 = np.ones((5, 5), np.uint8)
    kern2 = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kern2)
    mask = cv2.dilate(mask, kern1)
    imggreen = cv2.bitwise_and(img, img, mask)
    imggreen[mask == 0] = (255, 255, 255)
    imgblur = cv2.GaussianBlur(imggreen, (3, 3), 0)
    centx, centy,imgx= getContours(mask,imgblur,img2,t,co)
    cv2.imshow('box',img2)
    row=[]
    if(co!=5 and t==0):
       for i in range(len(centx)):
            ind=round((centy[i]-66.5)/39)
            xind=round((centx[i]-sta[ind])/45)
            gr[ind][xind]=cost


def camera(z):
    img = env.camera_feed()
    global img2
    img2 = img.copy()
    color(np.array([48, 166, 0]), np.array([65, 255, 255]), np.array([1, 0, 0]), np.array([0, 0, 0]), img, 4, z,1) #green
    color(np.array([11, 0, 0]), np.array([27, 255, 255]), np.array([1, 0, 0]), np.array([0, 0, 0]), img, 2, z,2)   #yellow
    color(np.array([122, 0, 0]), np.array([156, 255, 255]), np.array([1, 0, 0]), np.array([0, 0, 0]), img, 3, z,3) # purple
    color(np.array([0, 0, 168]), np.array([0, 0, 255]), np.array([1, 0, 0]), np.array([0, 0, 0]), img, 1, z,4) #white
    color(np.array([106, 230, 80]), np.array([179, 255, 255]), np.array([1, 0, 0]), np.array([0, 0, 0]), img, 6, z,5) #blue
    color(np.array([0, 6, 135]), np.array([6, 255, 255]), np.array([169, 74, 0]), np.array([180, 255, 255]), img, 1, z,6) # red
    color(np.array([148, 0, 187]), np.array([179, 255, 255]), np.array([1, 0, 0]), np.array([0, 0, 0]), img, 1, z,7) #pink

env.remove_car()
time.sleep(1)
camera(0)
env.respawn_car()

def exist(i, j):
    if i < 0 or j < 0 or i > 12 or j > numrow[i] - 1:
        return False
    else:
        return True

def updatec(gr):
    x = 0
    global c
    for i in range(0, size):
        for j in range(0, len(gr[i])):
            if gr[i][j] == 0:
                c[x][:] = 0
            else:
                if exist(i, j + 1):
                    c[x][2] = gr[i][j + 1]
                if exist(i, j-1):
                    c[x][5] = gr[i][j-1]
                if i<6:
                  if exist(i - 1, j-1):
                    c[x][0] = gr[i - 1][j-1]
                  if exist(i - 1, j):
                    c[x][1] = gr[i - 1][j]
                  if exist(i+1, j+1):
                    c[x][3] = gr[i+1][j+1]
                  if exist(i+1, j):
                    c[x][4] = gr[i+1][j]
                elif i>6:
                   if exist(i - 1, j):
                    c[x][0] = gr[i - 1][j]
                   if exist(i - 1, j+1):
                    c[x][1] = gr[i - 1][j+1]
                   if exist(i+1, j):
                    c[x][3] = gr[i+1][j]
                   if exist(i+1, j-1):
                    c[x][4] = gr[i+1][j-1]
                else:
                  if exist(i - 1, j-1):
                    c[x][0] = gr[i - 1][j-1]
                  if exist(i - 1, j):
                    c[x][1] = gr[i - 1][j]
                  if exist(i+1, j):
                    c[x][3] = gr[i+1][j]
                  if exist(i+1, j-1):
                    c[x][4] = gr[i+1][j-1]
                
            x += 1
def camera3():
    global orient
    img = env.camera_feed()
    imgx=img.copy()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imggray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    aruco.drawDetectedMarkers(imgx, corners, ids)
    pos = [(corners[0][0][0][0] + corners[0][0][2][0]) / 2, (corners[0][0][0][1] + corners[0][0][2][1]) / 2]
    a = complex(corners[0][0][0][0] - corners[0][0][3][0], corners[0][0][0][1] - corners[0][0][3][1])
    b=math.atan2(corners[0][0][0][0] - corners[0][0][3][0], corners[0][0][0][1] - corners[0][0][3][1])
    if a.imag == 0:
        if a.real > 0:
            orient = 3
        else:
            orient = 2
    elif a.real == 0:
        if a.imag > 0:
            orient = 1
        else:
            orient = 0
    return pos, a,b

def unvisit(x):
    flag = True
    for i in range(len(ver)):
        if x == ver[i]:
            flag = False
            break
    return flag

def getnum(current,dir):
  global numrow,rowsta
  # print("Hello",rowsta)
  cr=-1000
  for i in range(len(rowsta)-1):
     if current>=rowsta[i] and current<=rowsta[i+1]:
        cr=i
  # print(cr)
  dict1={0:-(numrow[cr-1]+1),1:-(numrow[cr-1]),2:1,3:numrow[cr]+1,4:numrow[cr],5:-1}
  dict2={0:-13,1:-12,2:1,3:13,4:12,5:-1}
  dict3={0:-(numrow[cr-1]),1:-(numrow[cr-1]-1),2:1,3:numrow[cr],4:numrow[cr]-1,5:-1}
  if(cr<6):
    y=dict1.get(dir)
  elif cr==6:
    y=dict2.get(dir)
  else:
    y=dict3.get(dir)
  return y

def exist2(x):
  if x<0 or x>126:
     return False
  else:
     return True

def djikstra(g, start, end):
    min = 99999
    global ver
    l = time.time()
    z = time.time()
    dist = np.zeros(127, int)
    parent = np.zeros(127, int)
    path = []
    for i in range(0, len(dist)):
        if i != start:
            dist[i] = 99999
    current = start
    ver.append(start)
    while unvisit(end) and z - l < 1:
        z = time.time()
        for i in range(6):
            if g[current][i] != 0:
                x = g[current][i] + dist[current]
                y=getnum(current,i)
                if exist2(current+y) and x < dist[current + y]:
                    dist[current + y] = x
                    parent[current + y] = current
        for i in range(0, len(dist)):
            if dist[i] < min and dist[i] != 0 and unvisit(i):
                min = dist[i]
                current = i

        ver.append(current)
        min = 99999
    x = end
    while x != start and z - l < 1:
        x = parent[x]
        path.append(x)
    path.reverse()
    path.append(end)
    ver = []
    return path, dist[end]

updatec(gr)
for i in range(len(gr)):
   for j in range(len(gr[i])):
       allcosts.append(gr[i][j])
# allcenters.sort()
# print(cellpos)
# cv2.waitKey(0)

########### MOVER CODE ##############
py=math.pi
rv=2
lv=10
rm=0.1
lm=30
ri=45
li=30
rci=100000

def show(a):
    print(a)
    b=get_pos()
    img=env.camera_feed().copy()
    color = (0, 255, 0)
    thickness = 9
    start=(round(cellpos[a][1]),round(cellpos[a][0]))
    end=(round(b[0]),round(b[1]))
    img = cv2.line(img, start,end, color, thickness)
    # cv2.imshow("img", img)
    cv2.waitKey(1)

def get_dir(a):
    pos=get_pos()
    x=cellpos[a][0]-pos[1]
    y=cellpos[a][1]-pos[0]
    ang=math.atan2(y,x)
    if ang<0:
      return (ang,ang+py)
    else:
      return (ang,ang-py)

def get_ort():
  _,_,ort=camera3()
  return ort

def get_pos():
  pos,_,_=camera3()
  return pos

def get_cell(pos):
   ind=round((pos[1]-66.5)/39)
   y=rowsta[ind]+round((pos[0]-sta[ind])/45)
   return y

def lndf(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def agdf(a,b):
    if(a-b>=py):
        return 2*py+b-a
    elif(a-b<=-py):
        return -(2*py+a-b)
    else:
        return b-a

def get_ortiented(ang):
    co=get_ort()
    i=0
    if(abs(agdf(ang[0],co))<=abs(agdf(ang[1],co))):
        if(agdf(ang[0],co)>0):
            while(agdf(ang[0],co)>rm):
                env.move_husky(rv,-rv,rv,-rv)
                p.stepSimulation()
                if i%ri==0:
                  co=get_ort()
                i+=1
            return 1
        else:
            while(agdf(ang[0],co)<-rm):
                env.move_husky(-rv,rv,-rv,rv)
                p.stepSimulation()
                if i%ri==0:
                  co=get_ort()
                i+=1
            return 1
    else:
        if(agdf(ang[1],co)>0):
            while(agdf(ang[1],co)>rm):
                env.move_husky(rv,-rv,rv,-rv)
                p.stepSimulation()
                if i%ri==0:
                  co=get_ort()
                i+=1
            return -1
        else:
            while(agdf(ang[1],co)<-rm):
                env.move_husky(-rv,rv,-rv,rv)
                p.stepSimulation()
                if i%ri==0:
                  co=get_ort()
                i+=1
            return -1

def block(x):
   cell=grcell(x)
   gr[cell[0]][cell[1]]=0

def unblock(x):
   cell=grcell(x)
   gr[cell[0]][cell[1]]=old[x]

def mover(nxt):
    ang=get_dir(nxt)
    mul=get_ortiented(ang)
    cor=[cellpos[nxt][1],cellpos[nxt][0]]
    prev=int(lndf(cor,get_pos()))
    i=0
    curr=lndf(cor,get_pos())
    while((int(curr)<=prev) or (abs(curr)>lm)) and  (abs(curr)>=8):
        env.move_husky(lv*mul,lv*mul,lv*mul,lv*mul)
        p.stepSimulation()
        if i%li==0:
          # show(nxt)
          prev=curr
          curr=lndf(cor,get_pos())
        if i%rci==0:
          ang=get_dir(nxt)
          get_ortiented(ang)
        i+=1
        # print("c = ",curr)
        # print("p = ",prev)
    # get=input("continue?")

def pathfollow(a,b):
   pt,dis=djikstra(c,a,b)
   for i in range(1,len(pt)):
     mover(pt[i])
curr=1
pos=get_pos()
curr=get_cell(pos)
spidy=[]
for i in range(len(redx)):
   rcell=get_cell((redx[i],redy[i]))
   if(rcell!=63):
      spidy.append(rcell)
print(curr)
print(spidy)

elecpos=get_cell((circ1[0][0],circ1[0][1]))
sandpos=get_cell((tria1[0][0],tria1[0][1]))
gobpos=get_cell((sq1[0][0],sq1[0][1]))
old[elecpos]=allcosts[elecpos]
old[sandpos]=allcosts[sandpos]
old[gobpos]=allcosts[gobpos]
block(elecpos)
block(sandpos)
block(gobpos)
updatec(gr)

p1,dis1=djikstra(c,curr,spidy[0])
p2,dis2=djikstra(c,curr,spidy[1])
if(dis1<dis2):
    pathfollow(curr,spidy[0])
    pathfollow(spidy[0],spidy[1])
else:
    pathfollow(curr,spidy[1])
    pathfollow(spidy[1],spidy[0])

env.unlock_antidotes()
camera(1)

for i in range(len(circ2)):
    if(not circ2[i] in circ1):
       elecanti=get_cell((circ2[i][0],circ2[i][1]))
    else:
       elecpos=get_cell((circ2[i][0],circ2[i][1]))
for i in range(len(tria2)):
    if(not tria2[i] in tria1):
       sandanti=get_cell((tria2[i][0],tria2[i][1]))
    else:
       sandpos=get_cell((tria2[i][0],tria2[i][1]))
for i in range(len(sq2)):
    if(not sq2[i] in sq1):
       gobanti=get_cell((sq2[i][0],sq2[i][1]))
    else:
       gobpos=get_cell((sq2[i][0],sq2[i][1]))

vildict={elecanti:elecpos,sandanti:sandpos,gobanti:gobpos}
pos=get_pos()
curr=get_cell(pos)
antis=[elecanti,sandanti,gobanti]
antidis,antidis2=[],[]

for i in range(3):
    pa,dis=djikstra(c,curr,antis[i])
    antidis.append((dis,antis[i]))
antidis.sort()
pathfollow(curr,antidis[0][1])


for i in range(3):
    if antis[i]!=antidis[0][1]:
       pa,dis=djikstra(c,antidis[0][1],antis[i])
       antidis2.append((dis,antis[i]))
antidis2.sort()
pathfollow(antidis[0][1],antidis2[0][1])

for i in range(3):
  if antis[i]!=antidis[0][1] and antis[i]!=antidis2[0][1]:
      last=antis[i]
pathfollow(antidis2[0][1],last)

order=[antidis[0][1],antidis2[0][1],last]
vilorder=[]
for i in range(3):
    vilorder.append(vildict[order[i]])

for i in range(3):
    old[vilorder[i]]=allcosts[vilorder[i]]

unblock(elecpos)
unblock(sandpos)
unblock(gobpos)
updatec(gr)

block(vilorder[1])
block(vilorder[2])
updatec(gr)
pathfollow(last,vilorder[0])

unblock(vilorder[1])
updatec(gr)
pathfollow(vilorder[0],vilorder[1])

block(vilorder[0])
unblock(vilorder[2])
updatec(gr)
pathfollow(vilorder[1],vilorder[2])
env.move_husky(0,0,0,0)
cv2.waitKey(0)