import numpy as np
from graphics import *
class VectorField:
    def __init__(self, xLength, yLength, zLength, xCount, yCount, zCount, initialStateVector):
        self.xLength  = xLength
        self.yLength  = yLength
        self.zLength  = zLength

        self.xCount = xCount
        self.yCount = yCount
        self.zCount = zCount
        self.initialStateVector = initialStateVector


    def draw(self):
        xd2=self.xLength/2
        xdc= (self.xLength)/self.xCount
        if xdc==0: xdc=1

        yd2=self.yLength/2
        ydc= (self.yLength)/self.yCount
        if ydc==0: ydc=1
            
        zd2=self.zLength/2
        zdc= (self.zLength)/self.zCount
        if zdc==0: zdc=1
        
        verticiesLine = np.zeros(12*64,dtype=np.float32)
        indexLine = 0

        verticiesTriangle = np.zeros(18*64,dtype=np.float32)
        indexTriangle = 0
        count = 0
        for x in np.arange(-1, 0, .25):
            for y in np.arange(-1, 0, .25):
                for z in np.arange(-1, .75, .25):
                    dx, dy, dz = self.initialStateVector(x, y, z)
          
                    
                    #x,y,z
                    verticiesLine[indexLine]=x
                    verticiesLine[indexLine+1]=y
                    verticiesLine[indexLine+2]=z
                    #color of x,y,z
                    verticiesLine[indexLine+3]=1
                    verticiesLine[indexLine+4]=1
                    verticiesLine[indexLine+5]=1
                    #x',y',z'
                    verticiesLine[indexLine+6]=x+dx
                    verticiesLine[indexLine+7]=y+dy
                    verticiesLine[indexLine+8]=z+dz
                    #color of x',y',z'
                    verticiesLine[indexLine+9]=1
                    verticiesLine[indexLine+10]=1
                    verticiesLine[indexLine+11]=1
                    indexLine+=12

                    
                    magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
                    ux, uy, uz = dx / magnitude, dy / magnitude, dz / magnitude


                    
                    # Arrowhead (triangle) size and position
                    arrow_size = 0.1  # Constant size of the arrowhead
                    arrow_length = 0.9  # Position arrowhead close to the end of the vector

                    # Arrowhead base position
                    tx, ty, tz = x + dx * arrow_length, y + dy * arrow_length, z + dz * arrow_length

                    # Perpendicular vector for arrowhead orientation (rotation based on cross product)
                    perp = np.cross([ux, uy, uz], [0, 1, 0])
                    if np.linalg.norm(perp) < 1e-6:  # Handle parallel vectors
                        perp = np.cross([ux, uy, uz], [0, 0, 1])
                    perp = perp / np.linalg.norm(perp) * arrow_size



                    #arrowtip 
                    verticiesTriangle[indexTriangle]=x + dx
                    verticiesTriangle[indexTriangle+1]=y + dy
                    verticiesTriangle[indexTriangle+2]=z + dz
                    #color of arrowtip 
                    verticiesTriangle[indexTriangle+3]=1
                    verticiesTriangle[indexTriangle+4]=1
                    verticiesTriangle[indexTriangle+5]=1
                    # arrow_base1
                    verticiesTriangle[indexTriangle+6]=tx + perp[0]
                    verticiesTriangle[indexTriangle+7]=ty + perp[1]
                    verticiesTriangle[indexTriangle+8]=tz + perp[2]
                    #color of  arrow_base1
                    verticiesTriangle[indexTriangle+9]=1
                    verticiesTriangle[indexTriangle+10]=1
                    verticiesTriangle[indexTriangle+11]=1
                    # arrow_base2
                    verticiesTriangle[indexTriangle+12]=tx - perp[0]
                    verticiesTriangle[indexTriangle+13]=ty - perp[1]
                    verticiesTriangle[indexTriangle+14]=tz - perp[2]
                    #color of  arrow_base2
                    verticiesTriangle[indexTriangle+15]=1
                    verticiesTriangle[indexTriangle+16]=1
                    verticiesTriangle[indexTriangle+17]=1

                    indexTriangle+=18
       
        draw_VectorField(verticiesLine, verticiesTriangle)         










    