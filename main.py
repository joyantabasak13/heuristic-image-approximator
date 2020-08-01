import numpy as np
import cv2
from random import seed
from random import uniform
from random import randint
from scipy.stats import truncnorm
from datetime import datetime
import copy
#seed(1)

###### CONSTANTS ###############
POLYGON_COUNT  = 50
POPULATION_SIZE = 1
INTRA_GEN_POP   = 10
POLYGON_ANGLES  = 10

##### Functions ################

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def generateRandomPolygon(polygon):
    area = 511*511
    angles = randint(3, POLYGON_ANGLES)
    originPt = [int(uniform(0, 511)), int(uniform(0,511))]
    originPt[0] = originPt[0] if ((originPt[0]+50) < 511) else (511-50)
    originPt[0] = originPt[0] if ((originPt[0]-50) > 0) else 50
    originPt[1] = originPt[1] if ((originPt[1] + 50) < 511) else (511 - 50)
    originPt[1] = originPt[1] if ((originPt[1] - 50) > 0) else 50
    polygon = []
    for x in range(angles):
        pt = [int(uniform(originPt[0] - 50 , originPt[0] + 50)), int(uniform(originPt[1] - 50 , originPt[1] + 50))]
        polygon.append(pt)
    polygon = np.asarray(polygon)
    np.reshape(polygon, (len(polygon), 2))
    polygon = np.int32(polygon)
    polygon = cv2.convexHull(polygon)
    polygon = np.asarray(polygon)
    polygon = polygon.flatten().reshape(-1,2)
    n = len(polygon)
    for i in range(n):
        i1 = (i + 1) % n
        area += polygon[i][0] * polygon[i1][1] - polygon[i1][0] * polygon[i][1]
    area *= 0.5
    area = abs(area)
    #Area =.5*[(x0y1 - x1y0) + ...+ (x(n-1)y0 - x0y(n-1))]
    print(area)
    color = randint(0,255)
    polygon = np.array(polygon).tolist()
    polygon.append(color)
    return polygon

def generateRandomPolygons(NumOfPolygon):
    randomPolygons = []
    for n in range(NumOfPolygon):
        polygon = []
        polygon = generateRandomPolygon(polygon)
        randomPolygons.append(polygon)
        #print(triangle)
    return randomPolygons

def polygonsToImg(polygons):
    # Create a black image
    img = np.zeros((512, 512), np.uint8)
    for i in range(len(polygons)):
        polygon = polygons[i]
        t_points = []
        t_points = polygon[:-1]
        t_points = np.asarray(t_points)
        np.reshape(t_points,(len(polygon)-1,2))
        t_points = np.int32(t_points)
        img = cv2.fillConvexPoly(img, cv2.convexHull(t_points), polygon[-1])
        #print("Colour for ",i," is ",triangle[3])
        #print("Shape for ", i, " is ", t_points)
    return img

def evaluateImage(test_Image, ref_Image):
    fitness_val = 0
    fitness_mat = np.absolute(test_Image - ref_Img)
    fitness_val = np.sum(fitness_mat)
    return fitness_val

def tweakPolygon(polygon, percentOfAngles):
    angleFraction = int (100/percentOfAngles)
    X = get_truncated_normal(mean=0, sd=3, low=-5, upp=5)
    change = []
    pts = []
    pointsToBeMutated = int ((len(polygon)-1)/angleFraction)
    for i in range((pointsToBeMutated*2)+1):
        change.append(X.rvs())
    change = np.array(change)
    for i in range(pointsToBeMutated):
        x = int(uniform(0,len(polygon)-1))
        for j in range(2):
            polygon[x][j] = int (polygon[x][j]+change[i*2+j])
            if polygon[x][j]>511:
                polygon[x][j] = 511
            if polygon[x][j] <0:
                polygon[x][j] = 0

    polygon[-1] = int (polygon[-1] + change[-1])
    if polygon[-1] < 0 :
        polygon[-1] = 0
    if polygon[-1] > 255:
        polygon[-1] = 255
    #print(change)
    return polygon

def evaluatePolygon(polygon, ref_Image):
    # Create a black image
    temp_img = np.zeros((512, 512), np.uint8)
    t_points = []
    t_points = polygon[:-1]
    t_points = np.asarray(t_points)
    np.reshape(t_points, (len(polygon) - 1, 2))
    t_points = np.int32(t_points)
    img = cv2.fillConvexPoly(temp_img, cv2.convexHull(t_points), polygon[-1])
    combined = temp_img[:, :]
    rows, cols = np.where(combined > 0)
    indices = list([rows,cols])
    fitness_val = 0
    fitness_val = int(abs(np.sum(ref_Image[tuple(indices)])- (polygon[-1]*len(rows)))/len(rows))
    #print("Polygon Fitness Val -->",fitness_val)
    #cv2.imshow(str(t_points[0,0]), temp_img)
    return fitness_val

def randomPolygonMutation(polygons, percentOfAngles):
    a= randint(0, POLYGON_COUNT - 1)
    tweakPolygon(polygons[a], percentOfAngles)
    return polygons

def tournamentSelectMutation(polygons, ref_Image, percentOfAngles):
    a = randint(0, POLYGON_COUNT - 1)
    b = randint(0, POLYGON_COUNT - 1)
    while (a == b):
        b = randint(0, POLYGON_COUNT - 1)
    a_val = evaluatePolygon(polygons[a], ref_Image) #evaluateTriangle(polygons[a], ref_Image)
    b_val = evaluatePolygon(polygons[b], ref_Image) #evaluateTriangle(polygons[b], ref_Image)
    if a_val < b_val :
        polygons[b] = tweakPolygon(polygons[b], percentOfAngles)
    else:
        polygons[a] = tweakPolygon(polygons[a], percentOfAngles)
    return polygons

def generateOffspings(population, ref_Image):
    off_Pop = []
    percentOfAngles = 50
    off_Pop = copy.deepcopy(population)  # ADD Parent
    for x in range(POPULATION_SIZE):
        for y in range(int(INTRA_GEN_POP/POPULATION_SIZE)):
            child = []
            child = copy.deepcopy(population[x])
            for z in range(int(POLYGON_COUNT / 2)):
                #child = randomPolygonMutation(child, percentOfAngles)
                child = tournamentSelectMutation(child, ref_Img, percentOfAngles)
            off_Pop.append(copy.deepcopy(child))
    return off_Pop

def selectSuccessorPop(population, ref_Image):
    offspring = generateOffspings(population,ref_Image)
    off_Image = []
    off_fitness = []
    successor_Pop = []
    successor_fitness = []
    successor_Image = []
    for x in range(len(offspring)):
        off_Image.append(polygonsToImg(offspring[x]))
        off_fitness.append(evaluateImage(off_Image[x],ref_Image))
    ind = np.argpartition(np.asarray(off_fitness), range(POPULATION_SIZE))[:POPULATION_SIZE]
    #print("Successor Fitness is \n")
    #print(off_fitness)
    #print("Successor is -->",str(off_fitness[ind[0]]))
    #print(ind)
    for x in range(len(ind)):
        successor_Pop.append(offspring[ind[x]])
        successor_fitness.append(off_fitness[ind[x]])
        successor_Image.append(off_Image[ind[x]])
    return successor_Pop, successor_fitness, successor_Image


##### Initialization ###########
best_fitness = 512*512*256
ref_Img = cv2.imread('ref.jpg', cv2.IMREAD_GRAYSCALE)
pop = []
pop_images = []
pop_fitness = []
generation = 0
for x in range(POPULATION_SIZE):
    randTri = generateRandomPolygons(POLYGON_COUNT)
    pop.append(randTri)
    print("Initial Image Created for Pop: ",x)
    #temp_image = trianglesToImg(randTri)
    #Pop_Images.append(temp_image)

while(1):
    pop, pop_fitness, pop_images = selectSuccessorPop(pop,ref_Img)
    ind = np.argmin(np.asarray(pop_fitness))
    gen_best = pop[ind]
    gen_best_fitness = pop_fitness[ind]
    gen_best_image = pop_images[ind]
    text_tag = str(datetime.now()) + " --> Gen: " + str(generation) + "--> Fitness -->" + str(gen_best_fitness)
    print(text_tag)
    f = open("fitness.txt", "a")
    f.write(str(gen_best_fitness))
    f.write("\n")
    f.close()
    if best_fitness > gen_best_fitness:
        text = "New Best at gen " + str(generation) + " with Fitness " + str(gen_best_fitness)
        print(text)
        best_fitness = gen_best_fitness
        with open('bestPolygons.txt', 'w') as f:
            for item in gen_best:
                f.write("%s\n" % item)
            f.close()
    if (generation%100 == 0):
        text = str(generation) + ".jpg"
        cv2.imwrite(text, gen_best_image)
    generation = generation + 1

#text = "Init_Image_No " + str(x)
#cv2.imshow(text,temp_image)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv.imwrite('refCopy.png',img)
 #   cv.destroyAllWindows()


