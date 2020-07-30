import numpy as np
import cv2
from random import seed
from random import uniform
from random import randint
from scipy.stats import truncnorm
from datetime import datetime
#seed(1)

###### CONSTANTS ###############
TRIANGLE_COUNT  = 50
POPULATION_SIZE = 1
INTRA_GEN_POP   = 5

##### Functions ################

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def generateRandomTriangle(triangle):
    area = 399*399
    for x in range(2):
        pt = [uniform(0, 399), uniform(0,399)]
        triangle.append(pt)
    while(area>8000):      #5% of total area
        pt = [uniform(0, 399), uniform(0,399)]
        area = int(0.5*(triangle[0][0]*(triangle[1][1] - pt[1])
                        + triangle[1][0]*(pt[1] - triangle[0][1]) + pt[0]*(triangle[0][1] - triangle[1][1])))
        area = abs(area)
        print(area)
    #Area =.5*[x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)]
    triangle.append(pt)
    color = randint(0,255)
    triangle.append(color)
    return triangle

def generateRandomTriangles(NumOfTri):
    randomTriangles = []
    for n in range(NumOfTri):
        triangle = []
        triangle = generateRandomTriangle(triangle)
        randomTriangles.append(triangle)
        #print(triangle)
    return randomTriangles

def trianglesToImg(triangles):
    # Create a black image
    img = np.zeros((400, 400), np.uint8)
    for i in range(len(triangles)):
        triangle = triangles[i]
        t_points = np.array([triangle[0], triangle[1], triangle[2]], dtype= np.int32)
        img = cv2.fillConvexPoly(img, t_points, triangle[3])
        #print("Colour for ",i," is ",triangle[3])
        #print("Shape for ", i, " is ", t_points)
    return img

def colorDiff(colorX,colorY):
    colorX_val = np.sum((np.array(colorX)).astype(int))
    colorY_val = np.sum((np.array(colorY)).astype(int))
    return abs(colorX_val - colorY_val)

def evaluateTriangle(triangle, ref_Image):
    # Create a black image
    temp_img = np.zeros((400, 400), np.uint8)
    t_points = np.array([triangle[0], triangle[1], triangle[2]], dtype='int32')
    #print(type(triangle[3]))
    #print(type(int(triangle[3])))
    #print(triangle[3])
    temp_img = cv2.fillConvexPoly(temp_img, t_points, triangle[3])
    combined = temp_img[:, :]
    rows, cols = np.where(combined > 0)
    fitness_val = 0
    for p in range(len(rows)):
        fitness_val = fitness_val + colorDiff(ref_Image[rows[p],cols[p]], triangle[3])
    #cv2.imshow(str(t_points[0,0]), temp_img)
    return fitness_val

def evaluateImage(test_Image, ref_Image):
    fitness_val = 0
    for r in range(400):
        for c in range(400):
            fitness_val = fitness_val + colorDiff(test_Image[r,c], ref_Image[r,c])
    return fitness_val

def tweakTriangle(triangle):
    X = get_truncated_normal(mean=0, sd=5, low=-20, upp=20)
    change = []
    for i in range(7):
        change.append(X.rvs())
    change = np.array(change)
    for i in range(3):
        for j in range(2):
            triangle[i][j] = int (triangle[i][j]+change[i*2+j])
            if triangle[i][j]>399:
                triangle[i][j] = 399
            if triangle[i][j] <0:
                triangle[i][j] = 0

    triangle[3] = int (triangle[3] + change[6])
    if triangle[3] < 0 :
        triangle[3] = 0
    if triangle[3] > 255:
        triangle[3] = 255
    #print(change)
    return triangle

def tournamentSelectMutation(triangles, ref_Image):
    a = randint(0,TRIANGLE_COUNT-1)
    b = randint(0,TRIANGLE_COUNT-1)
    while (a == b):
        b = randint(0, TRIANGLE_COUNT - 1)
    a_val = evaluateTriangle(triangles[a], ref_Image)
    b_val = evaluateTriangle(triangles[b], ref_Image)
    if a_val > b_val :
        triangles[b] = tweakTriangle(triangles[b])
    else:
        triangles[a] = tweakTriangle(triangles[a])
    return triangles

def generateOffspings(population, ref_Image):
    off_Pop = []
    for x in range(POPULATION_SIZE):
        off_Pop.append(population[x]) #ADD Parent
        for y in range(int(INTRA_GEN_POP/POPULATION_SIZE)):
            for z in range(int(TRIANGLE_COUNT/2)):
                population[x] = tournamentSelectMutation(population[x], ref_Image)
            off_Pop.append(population[x])
    return off_Pop

def selectSuccessorPop(population, ref_Image):
    offspring = generateOffspings(population,ref_Image)
    off_Image = []
    off_fitness = []
    successor_Pop = []
    successor_fitness = []
    successor_Image = []
    for x in range(len(offspring)):
        off_Image.append(trianglesToImg(offspring[x]))
        off_fitness.append(evaluateImage(off_Image[x],ref_Image))
    ind = np.argpartition(np.asarray(off_fitness), POPULATION_SIZE)[:POPULATION_SIZE]
    for x in range(len(ind)):
        successor_Pop.append(offspring[ind[x]])
        successor_fitness.append(off_fitness[ind[x]])
        successor_Image.append(off_Image[ind[x]])
    return successor_Pop, successor_fitness, successor_Image


##### Initialization ###########
best_image = np.zeros((400,400), np.uint8)
best_fitness = 400*400*256
ref_Img = cv2.imread('ref.jpg', cv2.IMREAD_GRAYSCALE)
file = open("Fitness.txt", "w")
pop = []
pop_images = []
pop_fitness = []
generation = 0
for x in range(POPULATION_SIZE):
    randTri = generateRandomTriangles(TRIANGLE_COUNT)
    pop.append(randTri)
    print("Initial Image Created for Pop: ",x)
    #temp_image = trianglesToImg(randTri)
    #Pop_Images.append(temp_image)

while(1):
    generation = generation + 1
    print(datetime.now())
    print(generation)
    pop, pop_fitness, pop_images = selectSuccessorPop(pop,ref_Img)
    file.writelines("\n%d\n" %(generation))
    file.writelines("%s " % item for item in pop_fitness)
    ind = np.argmin(np.asarray(pop_fitness))
    gen_best = pop[ind]
    gen_best_fitness = pop_fitness[ind]
    gen_best_image = pop_images[ind]
    text = "Gen_Best_" + str(generation) +".jpg"
    cv2.imwrite(text, gen_best_image)
    if best_fitness > gen_best_fitness:
        text = "New Best at gen " + str(generation) + " with Fitness " + str(gen_best_fitness)
        print(text)
        best_fitness = gen_best_fitness
        best_image = gen_best_image


file.close()
#text = "Init_Image_No " + str(x)
#cv2.imshow(text,temp_image)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv.imwrite('refCopy.png',img)
 #   cv.destroyAllWindows()


