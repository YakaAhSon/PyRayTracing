# ideal diffused reflecting surface

from PIL import Image
import numpy as np
import math
import random


BLACK = np.array((0,0,0))
RED = np.array((255,0,0))
GREEN = np.array((0,255,0))
BLUE = np.array((0,0,255))
WHITE = np.array((200,200,200))
GREY = np.array((128,128,128))
RESOLUTION = 800
MAX_DEPTH = 1

NUMBER_REFLECTION = 64

DIFFUSE_RATE = 0.8

def normalize(v):
    return v/math.sqrt(np.dot(v,v))

class Ray:
    # vo: vector(x,y,z), origin
    # vd: vector(x,y,z), direction
    
    def __init__(self,vo,vd, depth = 0):
        self.vo = np.array( vo )
        self.vd = np.array( vd )
        self.depth = depth
    def findPoint(self,alpha):
        return self.vo + self.vd*alpha

class Plane:
    # given the coordinate of a point on the plane, and the normal
    def __init__(self,pos, normal, color = WHITE):
        self.pos = np.array(pos)
        self.normal = normalize( np.array(normal) )
        self.color = np.array(color)
    def intersect(self, ray):
        # (vo + alpha*vd) is on the plane
        # ((vo + alpha*vd)-pos) is perpendiculer to normal
        # alpha = (pos-vo)*normal / (vd*normal)
        t1 = np.dot( self.pos-ray.vo, self.normal)
        t2 = np.dot(ray.vd, self.normal)

        # if the ray is pointing to the same direction as normal, return inf
        if t2>-1e-6:
            return np.inf, None, None
        alpha = t1/t2
        if alpha<0:
            return np.inf, None, None
        else:
            p = ray.findPoint(alpha)
            if p[0]>1.2 or p[1]>2.0001 or p[2]>1.6 or p[0]<-0.0001 or p[1]<-0.0001 or p[2]<-0.0001:
                return np.inf, None,None
            return alpha, self.normal, self.getColor(p)
    def getColor(self,p):
        p = 5*p# 0..10
        x = int(p[0])
        y = int(p[1])
        z = int(p[2])
        if (x+y+z )%2==0:
            return self.color
        else:
            return WHITE-self.color
    
class Sphere:
    # given the coordinate of the center, and the radiant
    def __init__(self, pos, r, color = WHITE):
        self.pos = np.array(pos)
        self.r = r
        self.color = np.array(color)

    def intersect(self,ray):
        # (vo + alpha*vd - pos)^2 = r^2
        # vd^2 * alpha^2 + 2*dot(vd,vo-pos) * alpha + (vo-pos)^2-r^2 = 0
        # return the smaller root only if two roots are both larger than 0
        vo_pos = ray.vo-self.pos
        # a*alpha^2 + b*alpha + c = 0
        a = np.dot(ray.vd, ray.vd)
        b = 2*np.dot(ray.vd, vo_pos)
        c = np.dot(vo_pos,vo_pos)-self.r*self.r
        
        delta = b*b-4*a*c
        if delta<=0:
            return np.inf, None, self.color
        if -b/a<0.0001:
            return np.inf, None, self.color
        if c/a<0.0001:
            return np.inf, None, self.color
        alpha = (-b-math.sqrt(delta))/(2*a)
        normal = ray.findPoint(alpha) - self.pos
        return alpha, normalize(normal), self.color

class Scene:

    def __init__(self):
        self._object_list_ = []
        self.setLight((1,0,0))

    def addObject(self,obj):
        self._object_list_.append(obj)

    # only one parallel light source is supported
    def setLight(self,v):
        # _light_ is acctually in the oppisite direction of light beam
        self._light_ = normalize(-np.array(v))

    # return true if v is in shadow
    # v is np.array
    def inShadow(self,v):
        ray = Ray(v,self._light_)
        l,normal,color = self.intersect(ray)
        if l<np.inf:
            return True
        else:
            return False

    def intersect(self,ray):
        l,normal, color = np.inf, None, None
        for obj in self._object_list_:
            nl,nnormal,ncolor = obj.intersect(ray)
            if nl<l:
                l, normal, color = nl, nnormal, ncolor
        return l, normal, color

    def trace(self,ray):
        alpha,normal,color = self.intersect(ray)

        # return black if the  ray is not intersected with any object
        # Potential problem: light source invisible
        if alpha == np.inf:
            return BLACK
        
        pos = ray.findPoint(alpha)

        result = BLACK
        # check if the intersection point is in the shadow
        if not self.inShadow(pos):
            result=color*np.dot(normal,self._light_)
        if ray.depth>=MAX_DEPTH:
            return result

        diffuse = BLACK
        for i in range(NUMBER_REFLECTION):
            # random diffused reflection
            d = np.array((random.random()-0.5, random.random()-0.5, random.random()-0.5))
            x = np.dot(d,normal)
            while x<0.001:
                d = np.array((random.random()-0.5, random.random()-0.5, random.random()-0.5))
                x = np.dot(d,normal)
            new_ray = Ray(pos,d,ray.depth+1)
            diffuse = diffuse+x*self.trace(new_ray)/float(NUMBER_REFLECTION)
        
        mirror = BLACK
        ref_dir = ray.vd-np.dot(ray.vd,normal)*2*normal
        new_ray = Ray(pos,ref_dir,ray.depth+1)
        mirror = self.trace(new_ray)
        return (diffuse)*DIFFUSE_RATE+mirror*(1-DIFFUSE_RATE)+result

# always square
class Camera:
    def __init__(self, resolution, angle):
        self.pos = np.array((0,0,0))
        self.direction = np.array((-1,0,0))
        self.resolution = resolution
        self.angle = angle
        self.up = np.array((0,0,1))
        self.right = np.array((0,1,0))
        self.upTo(self.up)
        
    def moveTo(self,v):
        self.pos = np.array(v)
    def lookAt(self,direction):
        self.direction = normalize(np.array(direction))
        self.upTo(self.up)
    def upTo(self,direction):
        self.right = normalize(np.cross(self.direction, direction))
        self.up = normalize(np.cross(self.right,self.direction))
        self.x = math.tan(self.angle/2)*self.right/(self.resolution/2)
        self.y = math.tan(self.angle/2)*self.up/(self.resolution/2)

    def generateRay(self, x , y):
        d = self.direction+ (x - self.resolution/2) * self.x + (self.resolution/2-y) * self.y
        return Ray(self.pos,d, depth = 0)
        


scene = Scene()
scene.setLight((-1,1,-2))

scene.addObject(Plane((0,0,0),(0,0,1),WHITE))
scene.addObject(Plane((0,0,0),(1,0,0), GREEN))
scene.addObject(Plane((0,2,0),(0,-1,0), BLUE))

scene.addObject(Sphere((0.5,0.5,0.3),0.25,RED))
scene.addObject(Sphere((0.5,1,0.3),0.25,BLUE))
scene.addObject(Sphere((0.5,1.5,0.3),0.25,GREEN))

scene.addObject(Sphere((0.5,0.5,0.8),0.25,BLUE))
scene.addObject(Sphere((0.5,1,0.8),0.25,WHITE))
scene.addObject(Sphere((0.5,1.5,0.8),0.25,RED))

camera = Camera(RESOLUTION,math.pi/3)
camera.moveTo((2,1.05,0.5))
camera.lookAt((-2,0,0))
camera.upTo((0,0,1))

img = Image.new("RGB",(RESOLUTION,RESOLUTION),"BLACK")
pixels = img.load()

for i in range(RESOLUTION):
    print((i/float(RESOLUTION)*100)),"%"
    for j in range(RESOLUTION):
        ray = camera.generateRay(i,j)
        c = scene.trace(ray)
        def tz(c):
            return min(255,int(c))
        pixels[i,j] = (tz(c[0]),tz(c[1]), tz(c[2]))
img.save("r%d_d%d_i%d.png"%(int(DIFFUSE_RATE*10),MAX_DEPTH,NUMBER_REFLECTION))
img.show()