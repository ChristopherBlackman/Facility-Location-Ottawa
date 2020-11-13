import re
import numpy as np
import itertools as it
from pprint import pprint
from sklearn.neighbors import NearestNeighbors
KML_ROAD_FILE = "./data/Road_Centrelines.kml"


def main():
    print("Extracting Data")
    pre_data = organize_data()
    print("Data Extracted")

    print("Converting To PointSet")
    point_set_x, point_set_y = to_points(pre_data)

    pprint(point_set_x)
    print("Cal. Intersections")
    #point_intersections(pre_data,point_set_x,point_set_y)

    print("Seg. Intersections")
    segment_intersections(pre_data,point_set_x,point_set_y)

def to_points(pre_data):
    tmp_x = []
    tmp_y = []
    
    for road in pre_data:
        for point in road['coordinates']:
            tmp_x.append(point)
            tmp_y.append(road['object_id'])
    return (np.array(tmp_x),np.array(tmp_y))

# get the 10 NN of a point
# assumptions : use Euclidian Distance for NN calculation, but test with spherical (curvature is almost plane like)
# assumptions : Nearest Radius is 0.0000000000001 -> least Sig. Digit in longitued, latitutde
# ToDo : Determine Output Data Format for graph
def point_intersections(pre_data,point_set_x,point_set_y):
    #neigh = NearestNeighbors(n_neighbors=10,metric=spherical_dist, radius=0.4) 
    neigh = NearestNeighbors(n_neighbors=10, radius=0.0000000000001) 
    neigh.fit(point_set_x)

    for road in pre_data:
            Y = neigh.kneighbors(road['coordinates'],10,return_distance=False)

            for point, nn in zip(road['coordinates'],Y):
                for index in nn:
                    if (spherical_dist(point,point_set_x[index]) < 10 and  not road['object_id'] == point_set_y[index]):
                        print(spherical_dist(point,point_set_x[index]))
                        print(road['object_id'], point_set_y[index])
                        

def segment_intersections(pre_data,point_set_x,point_set_y):
    for road in pre_data:
        start = road['coordinates'][0]
        end = road['coordinates'][-1]
        print(start)
        print(end)
    pass


# 2d only
# A is from t1 to point
# B is from t1 to t2
# returns the projection in terms of how much it is scaled by B
from numpy import linalg as la
def cal_projections(A,B):
    t = np.apply_along_axis(la.norm,1,B)
    return np.einsum('ij, ij->i',A,(B/t.reshape((t.shape[0],1))))/t

# returns the othogonal length of p to segment
# t1 is segment endpoint
# t2 is segment endpoint
# p  is point we are calculating orthogonal distance
def cal_length(t1,t2,p):
    l = cal_projections(t1-p,t1-t2)
    f = np.vectorize(lambda_condition)
    l = f(l)
    
    pp = t1*(1-l.reshape((l.shape[0],1))) + t2*(l.reshape((l.shape[0],1)))
    
    return np.apply_along_axis(la.norm,1,pp-p)

# checks if line hits segment
# return x if in range
# return inf if out of range
def lambda_condition(x):
    if x < 0.0 or x > 1.0:
        return np.inf
    return x


# Credit : Jaime
# https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
# Note : not matching up with values from ottawa, but may be a good estimate
# Note : note taking into account depth of terrain
def spherical_dist(pos1, pos2, r=6370994):
    pos1 = pos1 * np.pi / 180.0
    pos2 = pos2 * np.pi / 180.0
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

# Not The Right Way to Parse A XML/KML File
# Proper way would be to define a recursive Context Free Grammer, but like, Python (this is faster)
def organize_data():
    data_points = []


    road = None
    count = 0

    with open(KML_ROAD_FILE,'rt',encoding="utf-8") as f:
        btwn = re.compile(">[\w\s,-\.'\(\)]+<")
        asci = re.compile("[\w\s,-\.'\(\)]+")

        for line in f.readlines():
            # check for new placemark
            if re.search("<Placemark>",line):
                road = {'coordinates':None,'object_id':None,'municipality':None,'name':None,'coordinates':[]}
            if re.search("</Placemark>",line):
                data_points.append(road)

            # check for object_id
            if re.search("<SimpleData name=\"OBJECTID\">",line):
                road['object_id'] = asci.search(btwn.search(line).group(0)).group(0)
                pass

            # check for object_id
            if re.search("<SimpleData name=\"ROAD_NAME\">",line):
                road['name'] = asci.search(btwn.search(line).group(0)).group(0)
                pass

            # check for municipality
            if re.search("<SimpleData name=\"MUNICIPALITY\">",line):
                #print(re.search(">[\w\s]+<",line))
                road['municipality'] = asci.search(btwn.search(line).group(0)).group(0)
                pass

            # check for longitute, and latitude of path
            if re.search("<coordinates>",line):
                coordinates = re.findall("-{0,1}\d+\.\d+",line)
                temp = []
                for i in range(int(len(coordinates)/2)):
                    temp.append((float(coordinates[2*i]),float(coordinates[2*i+1])))
                road['coordinates'] = np.array(temp)
                count = count + len(temp)

    return data_points










if __name__ == "__main__":
    print("Running Main File")
    main()
